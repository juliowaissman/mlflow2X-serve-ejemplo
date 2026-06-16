# Ejemplo de uso de puesta en producción de modelos con MLflow 2.X

Implementar un patrón Champion-Challenger (Campeón-Retador) utilizando el sabor `pyfunc` de MLflow es una buena prácticas para asegurar que solo los modelos que realmente mejoran tus métricas lleguen a producción, dándote además la flexibilidad de incluir lógica personalizada de pre y post procesamiento.

Para mantener todo organizado, vamos a tener un pequeño ejemplito en tres partes:

- Entrenamiento y Registro (`PyFunc`): Una libreta de Jupyter para entrenar el modelo, la cual se puede entrenar desde Colab, utilizando los secretos de Colab.

- Evaluación Champion-Challenger: Un script donde utilizamos los modelos en modo *batch*, y utlizamos tanto el modelo en producción como un modelo retador. Si el modelo retador es mejor, entonces lo conertimos en campeón. ara hacer este paso, vamos a ejemplificarlo de como hacerlo desde las *github actions* de son herramientas de integración continua (CI).

- Puesta en Producción (usando *mlflow serve*): Ya sabemos como mandar llamar modelos para ejecutarlos en modo batch, así que vamos a ver como ejecutar modelos como *Model As A Service (MaaS)*. Esto son solo indicaciones en un archivo *markdown*.

Vamos a utilizar la cuenta de *Databricks community* y los valores los vamos a guardar en los secretos, tanto los de *Colab* como los del repositorio de *github*. Si no te acuerdas como se hace, te recomiendo que revises de nuevo [esta libreta](https://colab.research.google.com/github/mcd-unison/aaa-curso/blob/main/ejemplos/mlflow3-et.ipynb).


## 1. Entrenamiento del modelo

Ejecutar en *Colab* (o en forma local) la libreta `entrenamiento.ipynb` para entrenar el modelo y registrarlo en MLflow.

## 2. Evaluación Champion-Challenger

Esto lo vamos a hacer usando *github actions* para ejemplificar como se puede hacer en un entorno de integración continua (CI). El script de evaluación se encuentra en `evaluacion.py`. Este script va a cargar el modelo campeón y el modelo retador, los va a evaluar con un conjunto de datos de prueba, y va a comparar sus métricas. Si el modelo retador es mejor que el campeón, entonces lo va a promover a campeón.

para esto necesitamos una estructura de carpetas como la siguiente:

```
mi-proyecto-ml/
├── .github/
│   └── workflows/
│       └── mlflow_cicd.yml      # El flujo de GitHub Actions
├── entrenamiento.ipynb          # (Se ejecuta en Colab)
├── evaluacion.py                # Para CI/CD
├── requirements.txt             # Dependencias del proyecto
└── README.md
```

## 3. Puesta en Producción

Para poner el modelo en producción, podemos usar `mlflow serve` para servir el modelo como una API REST. Esto nos permitirá hacer peticiones de inferencia al modelo desde cualquier aplicación cliente. Para esto, primero necesitamos exportar el modelo registrado en MLflow a un formato que pueda ser servido, y luego usar `mlflow serve` para levantar el servicio de inferencia.

Para esto vamos a correrlo en forma local (ya que si no necesitaríamos un servidor para correrlo), pero la idea es que este proceso se pueda automatizar también usando CI/CD, de manera que cada vez que se registre un nuevo modelo campeón, se actualice automáticamente el servicio de inferencia con el nuevo modelo.

Antes de ejecutar cualquier comando de Python o de la terminal, debes exponer tus credenciales de Databricks en tu entorno local. Reemplaza los valores de ejemplo con los de tu cuenta:

```bash
# Cambiar el tracking URI al modo especial de Databricks
export MLFLOW_TRACKING_URI="databricks"

# Tu URL de Databricks (ejemplo: https://adb-123456789.0.azuredatabricks.net)
export DATABRICKS_HOST="https://<tu-espacio-de-trabajo>.azuredatabricks.net"

# Tu Token de Acceso Personal (PAT) generado en Settings -> Developer
export DATABRICKS_TOKEN="dapi********************************"
````

Nota para Windows (Command Prompt): Usa `set` en lugar de `export`. Para PowerShell, usa `$env:VAR="valor"`.

Una vez que las variables de entorno anteriores están configuradas en tu terminal, la CLI de MLflow puede comunicarse directamente con el Model Registry de Databricks. Para levantar el servidor web local utilizando el modelo campeón que definimos previamente, ejecuta:

```bash
mlflow models serve -m "models:/Custom_Iris_Model@champion" -p 5000 --env-manager local
````

Si prefieres apuntar a una versión específica en lugar de un alias, simplemente cambia el URI del modelo:
```bash
mlflow models serve -m "models:/Custom_Iris_Model/1" -p 5000 --env-manager local
```

Recuerda que necesitas que todas las librerías que utilizas estén instaladas localmente, ya que todo se ejecuta local (aunque recuperes el modelo de un servidor de modelos externo). Una vez que el servidor esté levantado, podemos hacer peticiones de inferencia al modelo usando `curl` o cualquier cliente HTTP. Por ejemplo:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"dataframe_split": {"columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], "data": [[5.1, 3.5, 1.4, 0.2]]}}' http://127.0.0.1:5000/invocations
```
Esto enviará una petición de inferencia al modelo servido, y deberíamos recibir una respuesta con la predicción del modelo para los datos de entrada proporcionados.



