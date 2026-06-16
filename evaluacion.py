import os
import sys
import mlflow
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

#  Configurar MLflow Tracking URI desde variable de entorno
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    print("ERROR: MLFLOW_TRACKING_URI no está configurada.")
    sys.exit(1) # Falla el pipeline de GitHub

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()
MODEL_NAME = "Iris_RF"

# Preparar datos de validación (idealmente, deberías bajarlos de un S3/Blob Storage)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Función para evaluar un modelo dado su URI
def evaluate_model_by_uri(model_uri):
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    except Exception as e:
        print(f"Aviso: No se pudo cargar {model_uri}. Error: {e}")
        return -1.0

if __name__ == "__main__":
    
    # Obtener el Retador (último modelo registrado)
    latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not latest_versions:
        print("No hay modelos registrados para evaluar.")
        sys.exit(1)
        
    challenger_version = max(latest_versions, key=lambda v: int(v.version))
    challenger_uri = f"models:/{MODEL_NAME}/{challenger_version.version}"
    champion_uri = f"models:/{MODEL_NAME}@champion"
    
    print("Iniciando evaluación Champion-Challenger en CI/CD...")
    champion_acc = evaluate_model_by_uri(champion_uri)
    challenger_acc = evaluate_model_by_uri(challenger_uri)
    
    print(f"Accuracy Champion: {champion_acc:.4f}")
    print(f"Accuracy Challenger (v{challenger_version.version}): {challenger_acc:.4f}")
    
    # Lógica de CI/CD
    if challenger_acc > champion_acc:
        print("✅ ¡El Challenger es mejor! Promoviendo a @champion.")
        client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version.version)
        # Salida 0 = Éxito para GitHub Actions
        sys.exit(0) 
    else:
        print("⚠️ El Challenger no superó al Champion. No se promueve.")
        # Salida 0 = Sigue siendo éxito, el pipeline corrió bien, solo no promovió nada.
        # (Cámbialo a sys.exit(1) si quieres que el Pull Request marque una "X" roja si el modelo empeora).
        sys.exit(0)