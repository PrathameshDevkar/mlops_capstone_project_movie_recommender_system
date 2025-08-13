# promote model

import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "PrathameshDevkar"
    repo_name = "mlops_capstone_project_movie_recommender_system"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()

    model_name = "my_model_1"
    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version_staging,
        stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == "__main__":
    promote_model()


# import os
# import mlflow
# import pickle
# import pandas as pd
# from sklearn.metrics import f1_score

# def evaluate_model(model_uri, vectorizer, test_data):
#     model = mlflow.pyfunc.load_model(model_uri)
#     X = test_data.iloc[:, :-1]
#     y = test_data.iloc[:, -1]
#     y_pred = model.predict(X)
#     return f1_score(y, y_pred)

# def promote_best_model():
#     # Set up credentials
#     dagshub_token = os.getenv("CAPSTONE_TEST")
#     if not dagshub_token:
#         raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

#     os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
#     os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

#     # Connect to MLflow
#     dagshub_url = "https://dagshub.com"
#     repo_owner = "PrathameshDevkar"
#     repo_name = "mlops_capstone_project_movie_recommender_system"
#     mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

#     client = mlflow.MlflowClient()
#     model_name = "my_model_1"

#     # Load vectorizer and test data
#     vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
#     test_data = pd.read_csv('data/processed/test_bow.csv')

#     # Find all staging models
#     staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

#     best_f1 = 0
#     best_version = None

#     for v in staging_versions:
#         model_uri = f"models:/{model_name}/{v.version}"
#         try:
#             f1 = evaluate_model(model_uri, vectorizer, test_data)
#             print(f"Version {v.version} F1 Score: {f1}")
#             if f1 > best_f1:
#                 best_f1 = f1
#                 best_version = v.version
#         except Exception as e:
#             print(f"Failed to evaluate version {v.version}: {e}")

#     if best_version:
#         # Archive all production models
#         prod_versions = client.get_latest_versions(model_name, stages=["Production"])
#         for v in prod_versions:
#             client.transition_model_version_stage(model_name, v.version, stage="Archived")

#         # Promote best model
#         client.transition_model_version_stage(model_name, best_version, stage="Production")
#         print(f"Promoted model version {best_version} with F1 score {best_f1}")
#     else:
#         print("No valid model found for promotion")

# if __name__ == "__main__":
#     promote_best_model()
