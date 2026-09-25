from __future__ import annotations

import math


FINAL_MODEL_VERSION = "final_dual_har_x_20260831"
FINAL_HORIZONS = (5, 20, 40)
FINAL_MODELS = {
    "main": {
        "model_name": "har__mse_log__macro+gvz+slv_iv+us_epu",
        "family": "har",
        "objective": "mse_log",
        "blocks": ["macro", "gvz", "slv_iv", "us_epu"],
        "alpha": 10.0,
    },
    "robust": {
        "model_name": "har__qlike__macro+us_epu",
        "family": "har",
        "objective": "qlike",
        "blocks": ["macro", "us_epu"],
        "alpha": 0.01,
    },
}


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)


def validate_final_model_settings(settings: dict) -> None:
    project = settings.get("project", {})
    if project.get("version") != FINAL_MODEL_VERSION:
        raise ValueError(
            f"Frozen model version must be {FINAL_MODEL_VERSION!r}; create a new "
            "version instead of mutating this package."
        )
    if tuple(settings.get("training", {}).get("horizons", [])) != FINAL_HORIZONS:
        raise ValueError(f"Frozen horizons must be {list(FINAL_HORIZONS)}.")
    if int(project.get("annualization_days", 0)) != 252:
        raise ValueError("Annualization must remain 252 trading days.")
    configured = settings.get("models", {})
    alphas = settings.get("training", {}).get("alpha_by_model", {})
    for role, expected in FINAL_MODELS.items():
        model = configured.get(role, {})
        for field in ("model_name", "family", "objective", "blocks"):
            if model.get(field) != expected[field]:
                raise ValueError(
                    f"Frozen {role}.{field} must equal {expected[field]!r}; "
                    f"found {model.get(field)!r}."
                )
        if not _same(alphas.get(role), expected["alpha"]):
            raise ValueError(
                f"Frozen alpha for {role} must equal {expected['alpha']}."
            )
