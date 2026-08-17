Sleep Disorder Prediction (Django + scikit-learn)

Quick start

1) Create and activate venv (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2) Install deps:
   python -m pip install --upgrade pip
   pip install -r requirements.txt

3) Train model (compares LR, SVC, RandomForest via CV and picks best):
   python train_model.py

4) Use real CSV (optional, preferred for authenticity):
   - Place your dataset at `data/sleep.csv`
   - Required columns: age, gender, bmi, sleep_duration, stress_level, physical_activity, heart_rate, systolic_bp, diastolic_bp, work_type, label
   - Accepted values:
     - gender: male/female or 1/0
     - work_type: sedentary/light/moderate/heavy or 0–3
     - label: healthy/insomnia/sleep apnea or 0/1/2
   - Then run: `python train_model.py`

5) Evaluate model accuracy:
   python evaluate_model.py

6) Run server:
   python manage.py migrate
   python manage.py runserver

Open http://127.0.0.1:8000/

Admin:
   python manage.py createsuperuser

Notes:
- Replace synthetic data with real dataset when available
- Review security settings before production (ALLOWED_HOSTS, SECRET_KEY)

