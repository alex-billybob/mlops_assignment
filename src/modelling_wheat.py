import json
from pathlib import Path
import pandas as pd
from hydra import main
from omegaconf import DictConfig, OmegaConf

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import mlflow, mlflow.sklearn
from pycaret.classification import (
    setup, compare_models, tune_model, blend_models, stack_models,
    finalize_model, predict_model, plot_model, pull, save_model, load_model, get_config
)
@main(config_path="config", config_name="config_wheat", version_base=None)
def run(cfg: DictConfig):
    print("Hydra loaded. CWD:", Path.cwd())
    print(OmegaConf.to_yaml(cfg))

    PROJ_ROOT = Path.cwd()               # stays at repo root due to hydra.job.chdir=false
    DATA_PATH       = PROJ_ROOT / cfg.paths.data_csv
    MODEL_SAVE_STEM = PROJ_ROOT / cfg.paths.model_stem
    SCHEMA_PATH     = PROJ_ROOT / cfg.paths.schema_json
    EXAMPLE_PATH    = PROJ_ROOT / cfg.paths.example_json
    BATCH_PATH      = PROJ_ROOT / cfg.paths.batch_csv
    ARTIFACT_DIR    = PROJ_ROOT / cfg.paths.artifacts_dir
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    (PROJ_ROOT / "models").mkdir(parents=True, exist_ok=True)
    (PROJ_ROOT / "data" / "wheatseeds").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    assert "Type" in df.columns
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=cfg.train.random_state, stratify=df["Type"]
    )
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    (ARTIFACT_DIR / "split_info.json").write_text(json.dumps({
        "random_state": cfg.train.random_state, "test_size": 0.2,
        "stratify": True, "train_count": len(train_df), "test_count": len(test_df)
    }, indent=2))

    # batch from TEST set
    feat_cols = [c for c in test_df.columns if c != "Type"]
    test_df[feat_cols].sample(n=min(10, len(test_df)), random_state=cfg.train.random_state)\
        .reset_index(drop=True).to_csv(BATCH_PATH, index=False)

    setup_kwargs = dict(
        data=train_df, target="Type", session_id=cfg.train.random_state,
        normalize=True, feature_selection=True, remove_multicollinearity=True,
        multicollinearity_threshold=0.95, fold=cfg.train.folds,
        fold_strategy="stratifiedkfold", fix_imbalance=False,
        log_experiment=False, verbose=False, html=False,
    )
    if cfg.train.polynomial_features:
        setup_kwargs.update(
            polynomial_features=True,
            polynomial_degree=cfg.train.polynomial_degree,
            polynomial_threshold=cfg.train.polynomial_threshold,
        )
    if cfg.train.bin_numeric_features:
        setup_kwargs.update(bin_numeric_features=list(cfg.train.bin_numeric_features))
    _ = setup(**setup_kwargs)

    top_models = compare_models(sort="Accuracy", n_select=cfg.train.compare_n_select)
    pull().to_csv(ARTIFACT_DIR / "baseline_compare_results.csv", index=False)

    tuned_models = []
    for m in (top_models if isinstance(top_models, list) else [top_models]):
        tuned_models.append(tune_model(
            m, optimize=cfg.train.optimize, fold=cfg.train.folds,
            search_library="optuna", choose_better=True, n_iter=cfg.train.tune_iter
        ))
    pull().to_csv(ARTIFACT_DIR / "tuned_results.csv", index=False)

    scores = []
    for tm in tuned_models:
        hold = predict_model(tm)
        f1m = f1_score(hold["Type"], hold["prediction_label"], average="macro")
        acc = accuracy_score(hold["Type"], hold["prediction_label"])
        scores.append((tm, float(f1m), float(acc)))
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_tuned, best_holdout_f1, _ = scores[0]

    candidate = best_tuned
    if len(tuned_models) > 1:
        try:
            b = blend_models(estimator_list=tuned_models, optimize=cfg.train.optimize,
                             choose_better=True, fold=cfg.train.folds)
            f1b = f1_score(*(lambda d=(predict_model(b)): (d["Type"], d["prediction_label"]))(), average="macro")
            if f1b > best_holdout_f1: candidate, best_holdout_f1 = b, float(f1b)
        except Exception: pass
        try:
            s = stack_models(estimator_list=tuned_models, meta_model=None,
                             optimize=cfg.train.optimize, choose_better=True, fold=cfg.train.folds)
            f1s = f1_score(*(lambda d=(predict_model(s)): (d["Type"], d["prediction_label"]))(), average="macro")
            if f1s > best_holdout_f1: candidate, best_holdout_f1 = s, float(f1s)
        except Exception: pass

    final_model = finalize_model(candidate)
    test_pred = predict_model(final_model, data=test_df.copy())
    test_acc = accuracy_score(test_pred["Type"], test_pred["prediction_label"])
    test_f1  = f1_score(test_pred["Type"], test_pred["prediction_label"], average="macro")
    print("Test Accuracy:", round(float(test_acc), 5), "Test Macro F1:", round(float(test_f1), 5))

    save_model(final_model, str(MODEL_SAVE_STEM))
    X_df, y_series = get_config("X"), get_config("y")
    schema = {
        "X_columns": list(X_df.columns),
        "X_dtypes": {c: str(t) for c, t in X_df.dtypes.items()},
        "classes": sorted(set(y_series)),
        "folds": cfg.train.folds,
        "random_state": cfg.train.random_state,
        "pipeline_repr": str(get_config("pipeline")),
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    EXAMPLE_PATH.write_text(json.dumps(test_df.drop(columns=["Type"]).iloc[[0]].to_dict(orient="records")[0], indent=2))

    if cfg.mlflow.tracking_uri: mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)
    model_pkl = MODEL_SAVE_STEM.with_suffix(".pkl")
    with mlflow.start_run(run_name="wheat_seeds_task2_hydra"):
        mlflow.log_param("random_state", cfg.train.random_state)
        mlflow.log_param("folds", cfg.train.folds)
        mlflow.log_param("tune_iter", cfg.train.tune_iter)
        mlflow.log_param("optimize", cfg.train.optimize)
        mlflow.log_metric("holdout_macro_f1_selected", float(best_holdout_f1))
        mlflow.log_metric("test_accuracy", float(test_acc))
        mlflow.log_metric("test_macro_f1", float(test_f1))
        mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="notebook_artifacts")
        for f in [model_pkl, SCHEMA_PATH, EXAMPLE_PATH, BATCH_PATH]:
            mlflow.log_artifact(str(f), artifact_path="inference_assets")
        loaded = load_model(str(MODEL_SAVE_STEM))
        mlflow.sklearn.log_model(loaded, artifact_path="model")
        try:
            run_id = mlflow.active_run().info.run_id
            mlflow.register_model(model_uri=f"runs:/{run_id}/model", name="WheatSeedsClassifier_Sklearn")
        except Exception:
            pass


if __name__ == "__main__":
    run()
