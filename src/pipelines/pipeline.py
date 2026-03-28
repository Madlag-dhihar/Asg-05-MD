"""
Pipeline Runner
Orchestrates:
ingestion → feature engineering → preprocess → train → evaluate
"""

from src.data.ingestion import ingest_data
from src.features.feature_en import feature_engineering
from src.preproses.preproses import preprocess_data
from src.models.train import split_data, train_baseline, tune_model, save_model
from src.models.eval import evaluate


ACCURACY_THRESHOLD = 0.75


def run_pipeline():

    print("=" * 50)
    print("STEP 1: Data Ingestion")

    train_df, test_df = ingest_data()

    print("\nSTEP 2: Feature Engineering")

    train_df = feature_engineering(train_df)

    print("Feature engineering completed")
    print("Dataset shape:", train_df.shape)

    print("\nSTEP 3: Preprocessing")

    X, y, feature_columns, encoder = preprocess_data(train_df, is_train=True)

    print("Features:", len(feature_columns))

    print("\nSTEP 4: Train Test Split")

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nSTEP 5: Baseline Training")

    baseline_model, run_id = train_baseline(X_train, y_train)

    print("MLflow Run ID:", run_id)

    print("\nSTEP 6: Hyperparameter Tuning")

    best_model = tune_model(X_train, y_train)

    print("\nSTEP 7: Save Model")

    save_model(best_model, feature_columns, encoder, X_train)

    print("\nSTEP 8: Evaluation")

    acc, prec, rec, f1 = evaluate(X_test, y_test, run_id)

    print("\n" + "=" * 50)

    if acc >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED (accuracy={acc:.4f})")
    else:
        print(f"Model REJECTED (accuracy={acc:.4f})")


if __name__ == "__main__":
    run_pipeline()