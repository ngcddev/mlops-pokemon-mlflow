"""Carga el modelo por alias y predice.

Resuelve `models:/<MLFLOW_MODEL_NAME>@<MLFLOW_MODEL_ALIAS>`. El artefacto
es solo el clasificador, así que las columnas del CSV ya tienen que venir
codificadas y escaladas, en el mismo orden del entrenamiento.

Es el entry point `predict` de `MLproject` (`mlflow run . -e predict`).
Sin `--input` usa el `input_example` guardado en el modelo.
"""

from __future__ import annotations

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import Model

import config


def _uri() -> str:
    return f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_MODEL_ALIAS}"


def cargar():
    """Devuelve el clasificador del alias."""
    config.activar_seguimiento()
    return mlflow.sklearn.load_model(_uri())


def predecir(datos: pd.DataFrame):
    """Predice una observación o un lote ya transformado."""
    return cargar().predict(datos)


def main():
    parser = argparse.ArgumentParser(
        description="Predice el tipo (Type1) con columnas ya codificadas y escaladas."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="CSV ya transformado. Si se omite, se usa el ejemplo guardado en el modelo.",
    )
    args = parser.parse_args()
    config.activar_seguimiento()
    if args.input:
        datos = pd.read_csv(args.input)
    else:
        datos = Model.load(_uri()).load_input_example(_uri())
        if isinstance(datos, pd.DataFrame):
            datos = datos.head(1)
    if datos is None:
        raise SystemExit("El modelo no tiene input_example. Pasa un CSV con --input.")
    print(predecir(datos).tolist())


if __name__ == "__main__":
    main()
