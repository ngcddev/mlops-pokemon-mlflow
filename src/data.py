"""Cargar, limpiar y partir los datos de Pokémon.

- elimina las columnas que no sirven como feature (`COLUMNS_TO_DROP`);
- quita duplicados y filas incompletas;
- codifica con `LabelEncoder` las columnas de `CATEGORICAS` (ninguna
  por ahora: todas las features son numéricas);
- escala las features con `StandardScaler`, sin tocar el objetivo;
- parte entrenamiento y prueba con `TEST_SIZE` y `RANDOM_SEED`.

El objetivo es `Type1`. El escalador se ajusta con todo el conjunto
limpio y después se hace la partición.

`registrar_preprocesamiento` deja en el run activo los conteos de cada
filtro, el mapa de cada `LabelEncoder`, la media y la escala de
`StandardScaler`, y una muestra antes y después de transformar.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

MUESTRA = 10


def _leer_csv() -> pd.DataFrame:
    if not config.DATA_PATH.is_file():
        raise FileNotFoundError(
            f"No está el conjunto en {config.DATA_PATH}. "
            "Revisa DATA_PATH en .env."
        )
    datos = pd.read_csv(config.DATA_PATH)
    datos.columns = [c.strip() for c in datos.columns]
    faltan = [c for c in (config.TARGET, *config.CATEGORICAS) if c not in datos.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en {config.DATA_PATH.name}: {faltan}")
    return datos


def _limpiar(datos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica los filtros del notebook y cuenta cuántas filas quita cada uno."""
    resumen = {
        "archivo": config.DATA_PATH.name,
        "filas_leidas": int(len(datos)),
        "columnas_leidas": list(datos.columns),
    }
    datos = datos.drop(columns=list(config.COLUMNS_TO_DROP))
    resumen["columnas_eliminadas"] = list(config.COLUMNS_TO_DROP)

    n = len(datos)
    datos = datos.drop_duplicates()
    resumen["filas_duplicadas"] = int(n - len(datos))

    n = len(datos)
    datos = datos.dropna()
    resumen["filas_incompletas"] = int(n - len(datos))

    datos = datos.reset_index(drop=True)
    resumen["filas_limpias"] = int(len(datos))
    return datos, resumen


def cargar_limpio() -> pd.DataFrame:
    """Lee `DATA_PATH` y aplica los filtros del notebook."""
    datos, _ = _limpiar(_leer_csv())
    return datos


def ajustar_preprocesador(datos: pd.DataFrame) -> dict:
    """Ajusta el codificador y el escalador. No incluye el objetivo (`Type1`)."""
    trabajo = datos.copy()
    encoders = {}
    for columna in config.CATEGORICAS:
        encoder = LabelEncoder()
        trabajo[columna] = encoder.fit_transform(trabajo[columna].astype(str))
        encoders[columna] = encoder

    numericas = [
        columna
        for columna in trabajo.columns
        if columna not in config.CATEGORICAS and columna != config.TARGET
    ]
    scaler = StandardScaler()
    scaler.fit(trabajo[numericas])
    features = [columna for columna in datos.columns if columna != config.TARGET]
    return {
        "encoders": encoders,
        "scaler": scaler,
        "numericas": numericas,
        "features": features,
    }
    
def aplicar_preprocesador(datos: pd.DataFrame, preprocesador: dict) -> pd.DataFrame:
    """Devuelve las features codificadas y escaladas, en el orden de entrenamiento."""
    trabajo = datos.copy()
    if config.TARGET in trabajo.columns:
        trabajo = trabajo.drop(columns=[config.TARGET])

    faltan = [c for c in preprocesador["features"] if c not in trabajo.columns]
    if faltan:
        raise ValueError(f"Faltan features para predecir: {faltan}")

    for columna, encoder in preprocesador["encoders"].items():
        trabajo[columna] = encoder.transform(trabajo[columna].astype(str))

    numericas = preprocesador["numericas"]
    escaladas = pd.DataFrame(
        preprocesador["scaler"].transform(trabajo[numericas]),
        columns=numericas,
        index=trabajo.index,
    )
    trabajo[numericas] = escaladas
    return trabajo[preprocesador["features"]]


def _transformaciones(preprocesador: dict) -> dict:
    """Mapa legible de codificadores y del escalador, sin el objeto de sklearn."""
    codificacion = {}
    for columna, encoder in preprocesador["encoders"].items():
        codificacion[columna] = {
            str(clase): int(codigo) for codigo, clase in enumerate(encoder.classes_)
        }

    scaler = preprocesador["scaler"]
    escalado = {
        columna: {
            "media": float(scaler.mean_[i]),
            "escala": float(scaler.scale_[i]),
        }
        for i, columna in enumerate(preprocesador["numericas"])
    }
    return {
        "objetivo": config.TARGET,
        "test_size": config.TEST_SIZE,
        "random_seed": config.RANDOM_SEED,
        "codificacion_label": codificacion,
        "escalado_standard": escalado,
        "features": list(preprocesador["features"]),
        "pasos": [
            f"eliminar columnas: {', '.join(config.COLUMNS_TO_DROP)}",
            "quitar duplicados",
            "quitar filas incompletas",
            "LabelEncoder en las columnas categóricas (ninguna)",
            "StandardScaler en el resto de features, sin el objetivo",
            f"partición test_size={config.TEST_SIZE}, random_state={config.RANDOM_SEED}",
        ],
    }


def cargar_particion():
    """Devuelve train, test, el preprocesador y el material para MLflow."""
    crudo = _leer_csv()
    datos, resumen = _limpiar(crudo)
    preprocesador = ajustar_preprocesador(datos)
    features = aplicar_preprocesador(datos, preprocesador)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        datos[config.TARGET],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    material = {
        "resumen": resumen,
        "transformaciones": _transformaciones(preprocesador),
        "muestra_cruda": crudo.head(MUESTRA),
        "muestra_limpia": datos.head(MUESTRA),
        "muestra_transformada": features.head(MUESTRA),
        "filas_train": int(len(x_train)),
        "filas_test": int(len(x_test)),
    }
    return x_train, x_test, y_train, y_test, preprocesador, material


def registrar_preprocesamiento(material: dict) -> None:
    """Escribe filtros, transformaciones y muestras en el run activo."""
    import mlflow

    if mlflow.active_run() is None:
        raise RuntimeError("registrar_preprocesamiento necesita un run de MLflow activo.")

    resumen = material["resumen"]
    mlflow.log_params(
        {
            "archivo": resumen["archivo"],
            "filas_leidas": resumen["filas_leidas"],
            "filas_duplicadas": resumen["filas_duplicadas"],
            "filas_incompletas": resumen["filas_incompletas"],
            "filas_limpias": resumen["filas_limpias"],
            "filas_train": material["filas_train"],
            "filas_test": material["filas_test"],
            "columnas_eliminadas": ",".join(resumen["columnas_eliminadas"]),
            "columnas_categoricas": ",".join(config.CATEGORICAS) or "ninguna",
        }
    )
    mlflow.log_dict(
        {"limpieza": resumen, **material["transformaciones"]},
        "preprocesamiento/transformaciones.json",
    )
    mlflow.log_table(material["muestra_cruda"], "preprocesamiento/muestra_cruda.json")
    mlflow.log_table(material["muestra_limpia"], "preprocesamiento/muestra_limpia.json")
    mlflow.log_table(
        material["muestra_transformada"],
        "preprocesamiento/muestra_transformada.json",
    )


def fila_cruda_ejemplo() -> pd.DataFrame:
    """Una observación sin el objetivo, para probar `predict` sin un archivo extra."""
    datos = cargar_limpio()
    return datos.drop(columns=[config.TARGET]).head(1)
