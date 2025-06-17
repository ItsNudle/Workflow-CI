import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_username     = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password     = os.getenv("MLFLOW_TRACKING_PASSWORD")

os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Model ML Eksperimen")
mlflow.sklearn.autolog()

X = pd.read_csv("spam_ham_emails_preprocessing/tfidf.csv")
y = pd.read_csv("spam_ham_emails_preprocessing/labels.csv")["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run(nested=True):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print("Akurasi:", acc)
