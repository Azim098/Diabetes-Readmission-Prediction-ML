# Diabetes Readmission Prediction - Flask + Docker

A local Flask web application for the diabetes hospital readmission ML project.

## Project structure

```text
diabetes_readmission_app/
├── app.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── data/
│   └── diabetic_data_clean.csv
├── templates/
│   └── index.html
└── static/
    └── style.css
```

After training, these files are generated automatically:

```text
model_pipeline.pkl
feature_metadata.json
model_metrics.json
```

## 1. Put the dataset in the data folder

Copy the cleaned dataset to:

```text
data/diabetic_data_clean.csv
```

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell activation is blocked, use Command Prompt:

```cmd
.venv\Scripts\activate
```

## 3. Install packages

```bash
pip install -r requirements.txt
```

## 4. Train and save the deployment pipeline

```bash
python train_model.py
```

This creates the complete pipeline, metadata and test metrics.

## 5. Start Flask

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

or:

```text
http://localhost:5000
```

## 6. Docker

Build:

```bash
docker build -t diabetes-readmission-app .
```

Run:

```bash
docker run --rm -p 5000:5000 diabetes-readmission-app
```

Open:

```text
http://localhost:5000
```

## Important

The model is an academic/research prediction system and not a medical diagnostic tool.
