"""
utils.py — Immo Predictor
=========================
Business logic layer: model loading, input validation, and prediction.
All Streamlit-specific code is kept in app.py; this module has no Streamlit imports.

Usage
-----
    from utils import load_models, predict_price, predict_building_type, FIELD_CONFIG
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Paths
_ROOT = Path(__file__).resolve().parent          # project root
MODELS_DIR = _ROOT / "models"
REG_MODEL_PATH = MODELS_DIR / "model_regression.pkl"
CLF_MODEL_PATH = MODELS_DIR / "model_classification.pkl"


# Field configuration — drives the Streamlit UI
# Each entry maps a feature name to its display metadata.

NEIGHBORHOOD_OPTIONS: list[str] = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "NAmes",   "NoRidge", "NPkVill", "NridgHt", "NWAmes",  "OldTown",
    "SWISU",   "Sawyer",  "SawyerW", "Somerst", "StoneBr", "Timber",
    "Veenker",
]

NEIGHBORHOOD_LABELS: dict[str, str] = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale":  "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR":  "Iowa DOT & Railroad",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "NAmes":   "North Ames",
    "NoRidge": "Northridge",
    "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights",
    "NWAmes":  "Northwest Ames",
    "OldTown": "Old Town",
    "SWISU":   "South & West of Iowa State University",
    "Sawyer":  "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber":  "Timberland",
    "Veenker": "Veenker",
}

HOUSESTYLE_OPTIONS: list[str] = [
    "1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl",
]

HOUSESTYLE_LABELS: dict[str, str] = {
    "1Story": "1 étage",
    "1.5Fin": "1,5 étage — 2e niveau aménagé",
    "1.5Unf": "1,5 étage — 2e niveau brut",
    "2Story": "2 étages",
    "2.5Fin": "2,5 étages — 2e niveau aménagé",
    "2.5Unf": "2,5 étages — 2e niveau brut",
    "SFoyer": "Split Foyer",
    "SLvl":   "Split Level",
}

BLDGTYPE_LABELS: dict[str, str] = {
    "1Fam":   "Maison individuelle (1Fam)",
    "2fmCon": "Conversion 2 familles (2fmCon)",
    "Duplx":  "Duplex",
    "TwnhsE": "Maison de ville — bout de rangée (TwnhsE)",
    "Twnhs":  "Maison de ville — milieu de rangée (Twnhs)",
}

# Regression field configuration

REG_FIELD_CONFIG: dict[str, dict[str, Any]] = {
    "GrLivArea": {
        "label":   "Surface habitable (sq ft)",
        "help":    "Surface au-dessus du sous-sol, hors garage.",
        "type":    "number",
        "min":     300,
        "max":     6000,
        "default": 1500,
        "step":    50,
    },
    "TotalBsmtSF": {
        "label":   "Surface sous-sol (sq ft)",
        "help":    "Surface totale du sous-sol. Entrez 0 si pas de sous-sol.",
        "type":    "number",
        "min":     0,
        "max":     3000,
        "default": 800,
        "step":    50,
    },
    "LotArea": {
        "label":   "Surface du terrain (sq ft)",
        "help":    "Superficie totale de la parcelle.",
        "type":    "number",
        "min":     1000,
        "max":     50000,
        "default": 9000,
        "step":    500,
    },
    "BedroomAbvGr": {
        "label":   "Chambres (hors sous-sol)",
        "help":    "Nombre de chambres au-dessus du niveau du sol.",
        "type":    "slider",
        "min":     0,
        "max":     8,
        "default": 3,
    },
    "FullBath": {
        "label":   "Salles de bain complètes",
        "help":    "Salles de bain complètes au-dessus du sous-sol.",
        "type":    "slider",
        "min":     0,
        "max":     4,
        "default": 2,
    },
    "TotRmsAbvGrd": {
        "label":   "Pièces totales (hors sous-sol)",
        "help":    "Nombre total de pièces, hors salles de bain et sous-sol.",
        "type":    "slider",
        "min":     2,
        "max":     14,
        "default": 7,
    },
    "OverallQual": {
        "label":   "Qualité générale (1–10)",
        "help":    "Note globale des matériaux et finitions (1 = Très mauvais, 10 = Excellent).",
        "type":    "slider",
        "min":     1,
        "max":     10,
        "default": 6,
    },
    "OverallCond": {
        "label":   "État général (1–10)",
        "help":    "État général de la maison (1 = Très mauvais, 10 = Excellent).",
        "type":    "slider",
        "min":     1,
        "max":     10,
        "default": 5,
    },
    "YearBuilt": {
        "label":   "Année de construction",
        "help":    "Année de la construction originale.",
        "type":    "number",
        "min":     1872,
        "max":     2010,
        "default": 1990,
        "step":    1,
    },
    "YearRemodAdd": {
        "label":   "Année de rénovation",
        "help":    "Dernière année de rénovation (= construction si jamais rénové).",
        "type":    "number",
        "min":     1872,
        "max":     2010,
        "default": 2000,
        "step":    1,
    },
    "GarageCars": {
        "label":   "Capacité garage (voitures)",
        "help":    "Nombre de voitures que le garage peut accueillir. 0 = pas de garage.",
        "type":    "slider",
        "min":     0,
        "max":     4,
        "default": 2,
    },
    "GarageArea": {
        "label":   "Surface garage (sq ft)",
        "help":    "Superficie du garage. Entrez 0 si pas de garage.",
        "type":    "number",
        "min":     0,
        "max":     1500,
        "default": 480,
        "step":    20,
    },
    "PoolArea": {
        "label":   "Surface piscine (sq ft)",
        "help":    "Superficie de la piscine. Entrez 0 si pas de piscine.",
        "type":    "number",
        "min":     0,
        "max":     800,
        "default": 0,
        "step":    10,
    },
    "Fireplaces": {
        "label":   "Cheminées",
        "help":    "Nombre de cheminées.",
        "type":    "slider",
        "min":     0,
        "max":     4,
        "default": 1,
    },
    "Neighborhood": {
        "label":   "Quartier",
        "help":    "Localisation dans la ville d'Ames, Iowa.",
        "type":    "select",
        "options": NEIGHBORHOOD_OPTIONS,
        "labels":  NEIGHBORHOOD_LABELS,
        "default": "NAmes",
    },
}

# Classification field configuration

CLF_FIELD_CONFIG: dict[str, dict[str, Any]] = {
    "GrLivArea": {
        "label":   "Surface habitable (sq ft)",
        "help":    "Surface au-dessus du sous-sol, hors garage.",
        "type":    "number",
        "min":     300,
        "max":     6000,
        "default": 1500,
        "step":    50,
    },
    "TotRmsAbvGrd": {
        "label":   "Pièces totales (hors sous-sol)",
        "help":    "Nombre total de pièces, hors salles de bain et sous-sol.",
        "type":    "slider",
        "min":     2,
        "max":     14,
        "default": 7,
    },
    "OverallQual": {
        "label":   "Qualité générale (1–10)",
        "help":    "Note globale des matériaux et finitions.",
        "type":    "slider",
        "min":     1,
        "max":     10,
        "default": 6,
    },
    "YearBuilt": {
        "label":   "Année de construction",
        "help":    "Année de la construction originale.",
        "type":    "number",
        "min":     1872,
        "max":     2010,
        "default": 1990,
        "step":    1,
    },
    "GarageCars": {
        "label":   "Capacité garage (voitures)",
        "help":    "Nombre de voitures que le garage peut accueillir. 0 = pas de garage.",
        "type":    "slider",
        "min":     0,
        "max":     4,
        "default": 2,
    },
    "Neighborhood": {
        "label":   "Quartier",
        "help":    "Localisation dans la ville d'Ames, Iowa.",
        "type":    "select",
        "options": NEIGHBORHOOD_OPTIONS,
        "labels":  NEIGHBORHOOD_LABELS,
        "default": "NAmes",
    },
    "HouseStyle": {
        "label":   "Style architectural",
        "help":    "Style de la maison (nombre de niveaux et finition).",
        "type":    "select",
        "options": HOUSESTYLE_OPTIONS,
        "labels":  HOUSESTYLE_LABELS,
        "default": "1Story",
    },
}


# Model loading — cached via a module-level dict

_MODEL_CACHE: dict[str, Any] = {}


def load_models() -> tuple[dict, dict]:
    """
    Load both model bundles from disk (once per process).

    Returns
    -------
    reg_bundle : dict with keys ``pipeline`` and ``meta``
    clf_bundle : dict with keys ``pipeline`` and ``meta``
    """
    if "reg" not in _MODEL_CACHE:
        if not REG_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle de régression introuvable : {REG_MODEL_PATH}\n"
                "Exécutez d'abord le notebook pour générer les fichiers .pkl."
            )
        with open(REG_MODEL_PATH, "rb") as f:
            _MODEL_CACHE["reg"] = pickle.load(f)

    if "clf" not in _MODEL_CACHE:
        if not CLF_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modèle de classification introuvable : {CLF_MODEL_PATH}\n"
                "Exécutez d'abord le notebook pour générer les fichiers .pkl."
            )
        with open(CLF_MODEL_PATH, "rb") as f:
            _MODEL_CACHE["clf"] = pickle.load(f)

    return _MODEL_CACHE["reg"], _MODEL_CACHE["clf"]


# Prediction helpers

def predict_price(user_inputs: dict[str, Any]) -> tuple[float, dict]:
    """
    Predict the sale price of a property.

    Parameters
    ----------
    user_inputs : dict
        Keys must match REG_FEATURES. Values may be scalars.

    Returns
    -------
    predicted_price : float
        Predicted sale price in USD.
    meta : dict
        Model metadata (model_name, test_mae, test_rmse, test_r2).
    """
    reg_bundle, _ = load_models()
    pipeline = reg_bundle["pipeline"]
    meta = reg_bundle["meta"]

    features = meta["features"]
    X = pd.DataFrame([{k: user_inputs.get(k, np.nan) for k in features}])
    predicted = float(pipeline.predict(X)[0])
    return predicted, meta


def predict_building_type(user_inputs: dict[str, Any]) -> tuple[str, np.ndarray, list[str], dict]:
    """
    Predict the building type (BldgType) of a property.

    Parameters
    ----------
    user_inputs : dict
        Keys must match CLF_FEATURES. Values may be scalars.

    Returns
    -------
    predicted_class : str
        Most likely BldgType code (e.g. "1Fam").
    probabilities : np.ndarray
        Class probability vector (shape: n_classes).
    classes : list[str]
        Ordered class labels corresponding to probabilities.
    meta : dict
        Model metadata (model_name, test_accuracy, test_f1_weighted).
    """
    _, clf_bundle = load_models()
    pipeline = clf_bundle["pipeline"]
    meta = clf_bundle["meta"]

    features = meta["features"]
    X = pd.DataFrame([{k: user_inputs.get(k, np.nan) for k in features}])

    predicted_class = pipeline.predict(X)[0]
    probabilities = pipeline.predict_proba(X)[0]
    classes = list(pipeline.classes_)

    return predicted_class, probabilities, classes, meta


# Input sanitisation

def validate_year_remod(year_built: int, year_remod: int) -> int:
    """YearRemodAdd must be >= YearBuilt."""
    return max(year_remod, year_built)


def format_price(value: float) -> str:
    """Format a USD price with thousand separators."""
    return f"${value:,.0f}"


def confidence_band(mae: float, price: float) -> tuple[float, float]:
    """Return a ±MAE confidence interval around the predicted price."""
    return max(0.0, price - mae), price + mae
