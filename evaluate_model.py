from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def load_model(models_dir: Path) -> object:
    model_path = models_dir / 'sleep_model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
    return joblib.load(model_path)


def generate_eval_data(n: int = 1000, seed: int = 123) -> tuple[pd.DataFrame, pd.Series]:
    # Prefer real CSV if present
    base_dir = Path(__file__).resolve().parent
    data_csv = base_dir / 'data' / 'sleep.csv'
    if data_csv.exists():
        df = pd.read_csv(data_csv)
        if 'gender' in df.columns and df['gender'].dtype == object:
            df['gender'] = df['gender'].str.lower().map({'male': 1, 'm': 1, 'female': 0, 'f': 0}).fillna(0).astype(int)
        if 'work_type' in df.columns and df['work_type'].dtype == object:
            mapping = {'sedentary': 0, 'light': 1, 'moderate': 2, 'heavy': 3}
            df['work_type'] = df['work_type'].str.lower().map(mapping).fillna(0).astype(int)
        if 'label' in df.columns and df['label'].dtype == object:
            label_map = {'healthy': 0, 'insomnia': 1, 'sleep apnea': 2, 'sleep_apnea': 2}
            df['label'] = df['label'].str.lower().map(label_map).astype(int)
    else:
        from train_model import generate_synthetic_dataset  # type: ignore
        df = generate_synthetic_dataset(num_rows=n, random_state=seed)

    X = df[['age','gender','bmi','sleep_duration','stress_level','physical_activity','heart_rate','systolic_bp','diastolic_bp','work_type']]
    y = df['label']
    return X, y


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / 'models'
    model = load_model(models_dir)
    X, y_true = generate_eval_data()
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall model accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()


