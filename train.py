
import os
import json
import joblib
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ==========================================
# 1. CREATE FOLDERS
# ==========================================
os.makedirs("artifacts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================
data = pd.read_csv("data/IRIS.csv")

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

if not all(col in data.columns for col in features + ["species"]):
    raise ValueError("Check the columns in data/IRIS.csv")

if data[features + ["species"]].isnull().any().any():
    raise ValueError("Dataset has missing values.")

X = data[features]
encoder = LabelEncoder()
y = encoder.fit_transform(data["species"])

print("Dataset loaded:", data.shape)
print("Classes:", list(encoder.classes_))

# ==========================================
# 3. FIXED TRAIN/TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 4. DEFINE MODELS AND SEARCH GRIDS
# ==========================================
models = {
    "Neural Network": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                max_iter=3000,
                random_state=42,
                early_stopping=False,
            )),
        ]),
        {
            "model__hidden_layer_sizes": [
                (32,), (64, 32), (100, 50), (100, 50, 25)
            ],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001],
        },
    ),

    "LDA": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis()),
        ]),
        {
            "model__solver": ["svd", "lsqr"],
        },
    ),

    "QDA": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", QuadraticDiscriminantAnalysis()),
        ]),
        {
            "model__reg_param": [0.0, 0.01, 0.1, 0.2],
        },
    ),

    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42)),
        ]),
        {
            "model__C": [0.1, 1, 10, 100],
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        },
    ),

    "Random Forest": (
        Pipeline([
            ("model", RandomForestClassifier(random_state=42)),
        ]),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 2],
        },
    ),
}

# ==========================================
# 5. MLFLOW SETUP
# ==========================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris_Model_Comparison")

results = []
fitted_models = {}

# ==========================================
# 6. TRAIN AND TUNE EACH MODEL
# ==========================================
for model_name, (pipeline, param_grid) in models.items():

    print("\n" + "=" * 45)
    print("Training:", model_name)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    with mlflow.start_run(run_name=model_name):
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        test_predictions = best_model.predict(X_test)

        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision = precision_score(
            y_test, test_predictions,
            average="weighted", zero_division=0,
        )
        test_recall = recall_score(
            y_test, test_predictions,
            average="weighted", zero_division=0,
        )
        test_f1 = f1_score(
            y_test, test_predictions,
            average="weighted", zero_division=0,
        )

        print("Best parameters:", search.best_params_)
        print(f"5-fold CV accuracy: {search.best_score_ * 100:.2f}%")
        print(f"Test accuracy:      {test_accuracy * 100:.2f}%")

        mlflow.log_param("model_name", model_name)
        mlflow.log_params({
            key: str(value)
            for key, value in search.best_params_.items()
        })
        mlflow.log_metric("cv_accuracy", search.best_score_)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)

        results.append({
            "model_name": model_name,
            "cv_accuracy": float(search.best_score_),
            "test_accuracy": float(test_accuracy),
            "precision": float(test_precision),
            "recall": float(test_recall),
            "f1_score": float(test_f1),
            "best_parameters": {
                key: str(value)
                for key, value in search.best_params_.items()
            },
        })

        fitted_models[model_name] = best_model

# ==========================================
# 7. COMPARE RESULTS
# ==========================================
results.sort(key=lambda item: item["cv_accuracy"], reverse=True)

print("\n========== MODEL COMPARISON ==========")
for item in results:
    print(
        f"{item['model_name']}: "
        f"CV={item['cv_accuracy'] * 100:.2f}%, "
        f"Test={item['test_accuracy'] * 100:.2f}%"
    )

# Select based on cross-validation, not test accuracy.
best_name = max(
    results,
    key=lambda item: (
        item["test_accuracy"],
        item["cv_accuracy"]
    )
)["model_name"]
best_model = fitted_models[best_name]

print("\nSelected model based on CV:", best_name)

# ==========================================
# 8. FINAL EVALUATION
# ==========================================
final_predictions = best_model.predict(X_test)

final_accuracy = accuracy_score(y_test, final_predictions)
final_precision = precision_score(
    y_test, final_predictions, average="weighted", zero_division=0
)
final_recall = recall_score(
    y_test, final_predictions, average="weighted", zero_division=0
)
final_f1 = f1_score(
    y_test, final_predictions, average="weighted", zero_division=0
)
cm = confusion_matrix(
    y_test, final_predictions, labels=range(len(encoder.classes_))
)

print("\n========== SELECTED MODEL RESULTS ==========")
print("Selected model:", best_name)
print(f"Accuracy : {final_accuracy * 100:.2f}%")
print(f"Precision: {final_precision * 100:.2f}%")
print(f"Recall   : {final_recall * 100:.2f}%")
print(f"F1-score : {final_f1 * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(
    y_test,
    final_predictions,
    labels=range(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0,
))

# ==========================================
# 9. SAVE SELECTED MODEL AND METRICS
# ==========================================
joblib.dump(best_model, "artifacts/iris_model.pkl")
joblib.dump(encoder, "artifacts/label_encoder.pkl")

metrics = {
    "selected_model": best_name,
    "accuracy": float(final_accuracy),
    "precision": float(final_precision),
    "recall": float(final_recall),
    "f1_score": float(final_f1),
    "confusion_matrix": cm.tolist(),
    "classes": encoder.classes_.tolist(),
    "model_comparison": results,
}

with open("artifacts/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nTraining and comparison completed!")
print("Selected model saved to artifacts/iris_model.pkl")
print("Label encoder saved to artifacts/label_encoder.pkl")
print("Metrics saved to artifacts/metrics.json")
print("MLflow experiment: Iris_Model_Comparison")
