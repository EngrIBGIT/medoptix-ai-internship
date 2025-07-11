import os
import joblib
import matplotlib.pyplot as plt
import shap
import logging

# Create folders if they don't exist
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# Save model using joblib
def save_model(model, name, model_type="classifier"):
    ensure_dir("models/saved_models")
    filename = f"models/saved_models/{name.lower().replace(' ', '_')}_{model_type}.joblib"
    joblib.dump(model, filename)
    logging.info(f"Saved model → {filename}")
    return filename

# Save SHAP summary plot
def save_shap_summary(model, X, name, model_type="classifier"):
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        ensure_dir("reports/figures")
        plt.title(f"SHAP Summary for {name}")
        shap.summary_plot(shap_values, X, show=False)
        fig_path = f"reports/figures/shap_{name.lower().replace(' ', '_')}_{model_type}.png"
        plt.savefig(fig_path)
        plt.close()
        logging.info(f"Saved SHAP plot → {fig_path}")
        return fig_path
    except Exception as e:
        logging.warning(f"SHAP failed for {name}: {e}")
        return None

# Load model
def load_model(path):
    return joblib.load(path)

# Print top feature importances (e.g., from RandomForest)
def print_feature_importance(model, feature_names, top_n=10):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:top_n]
        print("Top Features:")
        for i in indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
