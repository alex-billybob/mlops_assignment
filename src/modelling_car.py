import pandas as pd
import mlflow
import warnings
from pycaret.regression import *
import hydra
from omegaconf import DictConfig

warnings.filterwarnings('ignore')

def train_and_register_model(cfg: DictConfig):
    # Load cleaned data
    df = pd.read_csv(cfg.dataset.processed_path)

    # PyCaret setup
    exp = setup(
        data=df,
        target='Price',
        normalize=cfg.model.normalize,
        rare_to_value=cfg.model.rare_to_value,
        bin_numeric_features=cfg.model.bin_numeric_features,
        verbose=False,
        session_id=cfg.model.session_id
    )

    # Compare top models
    top_models = compare_models(n_select=cfg.model.n_select_top_models)

    # MLflow experiment setup
    mlflow.set_tracking_uri(cfg.model.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.model.mlflow_experiment)

    for model in top_models:
        model_name = str(model).split("(")[0]
        eval_result = predict_model(model)
        metrics = pull()  # metrics from PyCaret table
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("target", "Price")
            mlflow.log_param("normalize", cfg.model.normalize)
            mlflow.log_param("rare_to_value", cfg.model.rare_to_value)
            mlflow.log_param("bin_numeric_features", cfg.model.bin_numeric_features)
            mlflow.log_param("session_id", cfg.model.session_id)
            mlflow.log_metric("R2", metrics["R2"])
            mlflow.log_metric("MAE", metrics["MAE"])
            mlflow.log_metric("RMSE", metrics["RMSE"])
            mlflow.sklearn.log_model(model, "model")

    # Finalize best model
    best_model = top_models[0]
    tuned_model = tune_model(best_model, optimize=cfg.model.tune_metric, fold=5)
    final_model = finalize_model(tuned_model)

    # Save locally
    save_model(final_model, cfg.model.save_path)

    predict_model(final_model, data=df)
    final_metrics = pull()
    
    # Register in MLflow with input example
    input_example = df.head(1)
    from mlflow.models.signature import infer_signature
    signature = infer_signature(input_example, final_model.predict(input_example))

    with mlflow.start_run(run_name="Final_Best_Model"):
        mlflow.log_param("model", str(best_model).split("(")[0])
        mlflow.log_param("tuned", True)
        mlflow.log_param("optimize_metric", cfg.model.tune_metric)
    
        mlflow.log_metric("R2", final_metrics["R2"])
        mlflow.log_metric("MAE", final_metrics["MAE"])
        mlflow.log_metric("RMSE", final_metrics["RMSE"])

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name="UsedCarPriceModel",
            input_example=input_example,
            signature=signature
        )

    print("Training complete. Model saved and registered in MLflow.")

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    train_and_register_model(cfg)

if __name__ == "__main__":
    main()
