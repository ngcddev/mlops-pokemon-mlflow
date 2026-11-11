"""Nombres del experimento y del modelo, y semilla.

`src/` lee estos valores desde aquí. Salen de `.env`; si falta el
archivo, se usan los de `.env.example`.

Conjunto de Pokémon: se clasifica el tipo principal (`Type1`) a partir
de las estadísticas base y se elige el run con mayor accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / ".env")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ejemplo-mlops")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ejemplo-modelo")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "challenger")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

_ruta_datos = os.getenv("DATA_PATH", "data/pokemon.csv")
DATA_PATH = (
    Path(_ruta_datos)
    if Path(_ruta_datos).is_absolute()
    else RAIZ / _ruta_datos
)

# Decisiones cerradas en el notebook. No se repiten en el resto de src/.
TARGET = "Type1"
TEST_SIZE = 0.3
METRIC = "accuracy"
# Todas las features son numéricas: no hay columnas para LabelEncoder.
CATEGORICAS: tuple[str, ...] = ()

# Identificadores, texto libre, `Total` (suma de las seis estadísticas) y
# `Type2` (casi determina el tipo y tiene muchos vacíos).
COLUMNS_TO_DROP = (
    "ID",
    "Name",
    "Form",
    "Type2",
    "Total",
)

def activar_seguimiento():
    """Apunta MLflow al almacén y al experimento de este proyecto."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
