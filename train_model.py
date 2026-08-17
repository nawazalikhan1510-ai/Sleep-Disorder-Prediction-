import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib


def generate_synthetic_dataset(num_rows: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 80, size=num_rows)
    gender = rng.integers(0, 2, size=num_rows)  # 0 female, 1 male
    bmi = rng.normal(26, 4.5, size=num_rows).clip(14, 55)
    sleep_duration = rng.normal(7, 1.5, size=num_rows).clip(2, 12)
    stress_level = rng.integers(0, 11, size=num_rows)
    physical_activity = rng.integers(0, 300, size=num_rows)
    heart_rate = rng.integers(55, 100, size=num_rows)
    systolic_bp = rng.integers(100, 160, size=num_rows)
    diastolic_bp = rng.integers(60, 100, size=num_rows)
    work_type = rng.integers(0, 4, size=num_rows)  # 0 sedentary, 1 light, 2 moderate, 3 heavy

    risk_score = (
        (sleep_duration < 6).astype(int) * 2
        + (bmi > 30).astype(int)
        + (stress_level > 6).astype(int)
        + (systolic_bp > 140).astype(int)
        + (heart_rate > 90).astype(int)
    )

    # 0 Healthy, 1 Insomnia, 2 Sleep Apnea
    y = np.select(
        [risk_score <= 1, (risk_score == 2) | (stress_level > 7), risk_score >= 3],
        [0, 1, 2]
    )

    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'sleep_duration': sleep_duration,
        'stress_level': stress_level,
        'physical_activity': physical_activity,
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'work_type': work_type,
        'label': y.astype(int)
    })
    return df


def load_or_generate_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        # Attempt to normalize common string fields to numeric encodings
        if 'gender' in df.columns and df['gender'].dtype == object:
            df['gender'] = df['gender'].str.lower().map({'male': 1, 'm': 1, 'female': 0, 'f': 0}).fillna(0).astype(int)
        if 'work_type' in df.columns and df['work_type'].dtype == object:
            mapping = {'sedentary': 0, 'light': 1, 'moderate': 2, 'heavy': 3}
            df['work_type'] = df['work_type'].str.lower().map(mapping).fillna(0).astype(int)
        # If label provided as string
        if 'label' in df.columns and df['label'].dtype == object:
            label_map = {'healthy': 0, 'insomnia': 1, 'sleep apnea': 2, 'sleep_apnea': 2}
            df['label'] = df['label'].str.lower().map(label_map).astype(int)
        return df
    # Fallback: synthetic
    return generate_synthetic_dataset()


def train_and_save_model() -> None:
    base_dir = Path(__file__).resolve().parent
    data_csv = base_dir / 'data' / 'sleep.csv'
    if not data_csv.exists():
        data_csv.parent.mkdir(parents=True, exist_ok=True)
        df_tmp = generate_synthetic_dataset(num_rows=3000, random_state=42)
        df_tmp.to_csv(data_csv, index=False)
        print(f"No CSV found. Generated sample dataset at {data_csv}")
    df = load_or_generate_dataset(data_csv)
    X = df[[
        'age', 'gender', 'bmi', 'sleep_duration', 'stress_level', 'physical_activity',
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'work_type']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Models to compare
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=400, C=0.8))
        ]),
        'SVC_rbf': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=0.8, gamma='scale', probability=True))
        ]),
        'RandomForest': RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=8,
            random_state=42,
            class_weight='balanced_subsample'
        )
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    MIN_ACCURACY = 0.85
    cv_results: dict[str, float] = {}
    for name, m in models.items():
        scores = cross_val_score(m, X_train, y_train, cv=skf, scoring='accuracy')
        cv_results[name] = float(scores.mean())
        status = "OK" if scores.mean() >= MIN_ACCURACY else "LOW"
        print(f"{name} CV accuracy: {scores.mean():.4f} ± {scores.std():.4f} [{status}]")

    # Pick best by CV
    best_name = max(cv_results, key=cv_results.get)
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nBest model: {best_name} | Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Insomnia', 'Sleep Apnea']))

    # Store confusion matrix for inspection
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix:\n', cm)

    models_dir = base_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / 'sleep_model.joblib')
    (models_dir / 'metrics.txt').write_text(
        f"best={best_name}\ncv={cv_results}\naccuracy={acc:.4f}\nthreshold={MIN_ACCURACY}\nconfusion_matrix={cm.tolist()}\n",
        encoding='utf-8'
    )
    print(f"Saved best model to {models_dir / 'sleep_model.joblib'} and metrics.txt")


if __name__ == '__main__':
    train_and_save_model()


