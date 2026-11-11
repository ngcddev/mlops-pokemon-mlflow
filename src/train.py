"""Entrena las cuatro estrategias del notebook y las deja en MLflow.

Cada modelo es un run del experimento `MLFLOW_EXPERIMENT_NAME`:
regresión logística, SVM, árbol de decisión y random forest. En todos
se registra accuracy sobre la partición de prueba, los parámetros y el
clasificador. El artefacto `model` es solo ese estimador: la API espera
las columnas ya codificadas y escaladas. La columna Dataset del run
apunta a la partición de entrenamiento y a la de prueba.

Es el entry point `main` de `MLproject` (`mlflow run .`).
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from data import cargar_particion, registrar_preprocesamiento


def _modelos():
    semilla = config.RANDOM_SEED
    return [
        ("regresion_logistica", LogisticRegression(random_state=semilla)),
        ("svm", SVC(random_state=semilla)),
        ("arbol_decision", DecisionTreeClassifier(random_state=semilla)),
        ("random_forest", RandomForestClassifier(random_state=semilla)),
    ]


def _datasets(x_train, y_train, x_test, y_test):
    """Arma los dos conjuntos que muestra la columna Dataset de cada run."""
    origen = str(config.DATA_PATH)

    def conjunto(features, objetivo, nombre):
        tabla = features.copy()
        tabla[config.TARGET] = objetivo
        return mlflow.data.from_pandas(
            tabla,
            source=origen,
            targets=config.TARGET,
            name=nombre,
        )

    return (
        conjunto(x_train, y_train, "pokemon_train"),
        conjunto(x_test, y_test, "pokemon_test"),
    )


def _parametros(modelo) -> dict:
    parametros = {}
    for clave, valor in modelo.get_params().items():
        if isinstance(valor, (int, float, str, bool)):
            parametros[clave] = valor
        else:
            parametros[clave] = "None" if valor is None else str(valor)
    return parametros


def entrenar():
    """Ajusta los cuatro modelos y abre un run por cada uno."""
    # `mlflow run` ya abrió un run padre. Los modelos van en runs propios
    # del experimento configurado, para que `register.py` pueda compararlos.
    os.environ.pop("MLFLOW_RUN_ID", None)
    config.activar_seguimiento()
    mlflow.sklearn.autolog(disable=True)

    x_train, x_test, y_train, y_test, _, material = cargar_particion()
    with mlflow.start_run(run_name="preprocesamiento"):
        registrar_preprocesamiento(material)
    print(
        f"Partición: train {x_train.shape[0]} filas, "
        f"test {x_test.shape[0]} filas, {x_train.shape[1]} features"
    )

    datos_train, datos_test = _datasets(x_train, y_train, x_test, y_test)
    ejemplo = x_train.head(1)

    for nombre, modelo in _modelos():
        with mlflow.start_run(run_name=nombre):
            mlflow.log_input(datos_train, context="training")
            mlflow.log_input(datos_test, context="evaluation")
            modelo.fit(x_train, y_train)
            prediccion = modelo.predict(x_test)
            accuracy = accuracy_score(y_test, prediccion)

            mlflow.log_params(_parametros(modelo))
            mlflow.log_param("estrategia", nombre)
            mlflow.log_param("test_size", config.TEST_SIZE)
            mlflow.log_param("random_seed", config.RANDOM_SEED)
            mlflow.log_metric(config.METRIC, accuracy)
            mlflow.sklearn.log_model(
                modelo,
                name="model",
                input_example=ejemplo,
                serialization_format="pickle",
            )
            print(f"{nombre}: {config.METRIC}={accuracy:.6f}")


if __name__ == "__main__":
    entrenar()
