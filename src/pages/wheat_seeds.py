# wheat_seeds.py
# Streamlit app for real-time single and batch predictions (Task 3)
# Requirements: pycaret==3.3.2, streamlit==1.49.0, pandas==2.1.4, scikit-learn==1.4.2, numpy
# Files expected in the working directory:
# - wheat_seeds_pipeline.pkl              (PyCaret saved model pipeline)
# - pycaret_setup_config.json             (contains X_columns and classes)
# - example_request.json                  (one example row for the form defaults)
# - wheat_seeds_batch_examples.csv        (test-set feature batch for demo)


import json
from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

APP_TITLE = "Wheat Seeds Classifier"
MODEL_STEM = "models/wheat_seeds_pipeline"  # load_model will read wheat_seeds_pipeline.pkl
SCHEMA_FILE = "src/config/pycaret_setup_config.json"
EXAMPLE_FILE = "data/wheatseeds/example_request.json"
BATCH_SAMPLE_FILE = "data/wheatseeds/wheat_seeds_batch_examples.csv"

# Optional mapping for readability
CLASS_NAME_MAP = {1: "Kama", 2: "Rosa", 3: "Canadian"}


@st.cache_resource(show_spinner=False)
def _load_pipeline_and_schema():
    pipe = load_model(MODEL_STEM)

    schema = {}
    if Path(SCHEMA_FILE).exists():
        with open(SCHEMA_FILE, "r") as f:
            schema = json.load(f)
    cols = schema.get("X_columns", None)
    classes = schema.get("classes", None)

    example = None
    if Path(EXAMPLE_FILE).exists():
        with open(EXAMPLE_FILE, "r") as f:
            example = json.load(f)

    return pipe, cols, classes, example


def _coerce_single_row(feature_order, inputs_dict):
    row = {k: inputs_dict.get(k, None) for k in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def _predict_single(df_one_row, pipe):
    preds = predict_model(pipe, data=df_one_row.copy(), raw_score=True)
    label = preds.loc[0, "prediction_label"]
    score = float(preds.loc[0, "prediction_score"]) if "prediction_score" in preds.columns else None
    prob_cols = [c for c in preds.columns if c.lower().startswith("score_")]
    probs = preds.loc[0, prob_cols].to_dict() if prob_cols else None
    return label, score, probs, preds


def _validate_and_fix_batch(df_in, required_cols):
    df = df_in.copy()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[required_cols].copy()


def _bytes_download(df, filename):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button("Download predictions CSV", buffer, file_name=filename, type="primary")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="collapsed")
    st.title(APP_TITLE)
    st.caption("Real-time single and batch predictions. Saved PyCaret pipeline with consistent preprocessing.")

    pipe, feature_cols, classes, example = _load_pipeline_and_schema()
    if feature_cols is None:
        st.error("Schema not found. Ensure pycaret_setup_config.json with 'X_columns' is present.")
        st.stop()

    tabs = st.tabs(["Single prediction", "Batch prediction", "Model card"])

    # Single prediction tab
    with tabs[0]:
        st.subheader("Single prediction")
        col_left, col_right = st.columns([1, 1])

        defaults = example or {c: 0.0 for c in feature_cols}

        with col_left:
            st.write("Feature inputs")
            inputs = {}
            for c in feature_cols:
                val = defaults.get(c, 0.0)
                inputs[c] = st.number_input(label=c, value=float(val) if val is not None else 0.0, format="%.5f")

            if st.button("Predict", type="primary"):
                try:
                    one = _coerce_single_row(feature_cols, inputs)
                    label, score, probs, preds = _predict_single(one, pipe)

                    st.success("Prediction complete.")
                    st.write("**Predicted class (numeric):**", int(label))
                    st.write("**Predicted class (name):**", CLASS_NAME_MAP.get(int(label), str(label)))
                    if score is not None:
                        st.write("**Predicted class probability:**", f"{score:.4f}")

                    if probs is not None:
                        pretty = {}
                        for k, v in probs.items():
                            try:
                                cls = int(str(k).split("_")[-1])
                                pretty[CLASS_NAME_MAP.get(cls, k)] = float(v)
                            except Exception:
                                pretty[k] = float(v)
                        st.write("**Class probabilities:**")
                        st.json(pretty)

                    with st.expander("Raw prediction row"):
                        st.dataframe(preds)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        with col_right:
            st.info("Tip: The form is prefilled with an example row from `example_request.json` if present.")

    # Batch prediction tab
    with tabs[1]:
        st.subheader("Batch prediction")
        st.write("Upload a CSV with columns exactly matching the training schema.")

        use_sample = st.toggle("Use packaged batch sample", value=Path(BATCH_SAMPLE_FILE).exists())

        df_in = None
        if use_sample and Path(BATCH_SAMPLE_FILE).exists():
            try:
                df_in = pd.read_csv(BATCH_SAMPLE_FILE)
                st.caption(f"Loaded {BATCH_SAMPLE_FILE}")
            except Exception as e:
                st.warning(f"Sample file load failed: {e}")

        uploaded = st.file_uploader("Or upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"CSV read failed: {e}")

        if df_in is not None:
            try:
                clean = _validate_and_fix_batch(df_in, feature_cols)
                st.write("Preview (first 10 rows):")
                st.dataframe(clean.head(10))

                if st.button("Run batch prediction", type="primary"):
                    preds = predict_model(pipe, data=clean.copy(), raw_score=True)
                    preds["prediction_name"] = preds["prediction_label"].apply(lambda x: CLASS_NAME_MAP.get(int(x), str(x)))
                    st.success("Batch prediction complete.")
                    st.write("Preview with predictions:")
                    st.dataframe(preds.head(20))
                    _bytes_download(preds, "wheat_seeds_predictions.csv")

            except Exception as e:
                st.error(f"Batch validation/prediction failed: {e}")
        else:
            st.info("Provide a CSV or toggle the packaged sample.")

    # Model card tab (also hosts About)
    with tabs[2]:
        st.subheader("Model card")
        st.info(
            "About: This app serves a multiclass model that predicts wheat variety "
            "from kernel geometry using a saved PyCaret 3.3.2 pipeline. "
            "Inputs are validated against the training schema. "
            "Classes: 1 → Kama, 2 → Rosa, 3 → Canadian. "
            "Artifacts: `wheat_seeds_pipeline.pkl`, `pycaret_setup_config.json`."
        )
        st.markdown(
            """**Use-case**  
Predict wheat variety (Kama, Rosa, Canadian) from kernel geometry.

**Inputs**  
Seven numeric features: Area, Perimeter, Compactness, Length, Width, AsymmetryCoeff, Groove.

**Preprocessing**  
Saved PyCaret pipeline handles normalization, feature selection, and multicollinearity removal.

**Metrics**  
Reported in Task 2 on a held-out test set: Accuracy and Macro-F1. Use MLflow run for exact values.

**Caveats**  
Small dataset. Avoid out-of-distribution inputs. Probabilities may be overconfident.
            """
        )


if __name__ == "__main__":
    main()
