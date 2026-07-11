# Disease Prediction System — Team 2 (iDatalytics / RP2)

ML-based disease prediction from symptoms with a Generative AI recipe recommendation layer, deployed as a Flask web application.

---

## Team

| Name | Role |
|---|---|
| Milan TJ | Team Lead |
| Sona Santhosh | Member |
| Nandana Venugopal | Member |
| Ajul ND | Member |
| P Padm Sarang | Member |

---

## Project Structure

```
├── app/                        # Flask application (Ticket 3)
├── models/
│   ├── best_model.pkl          # Primary model — Bernoulli NB (weights NOT committed — see CONTRIBUTING.md)
│   ├── bernoulli_nb.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── svm_rbf.pkl
│   ├── xgboost.pkl
│   ├── symptom_cols.json       # Ordered 132-column feature list
│   ├── disease_label_mapping.csv
│   ├── model_metadata.json
│   └── pipeline.py             # Import-ready prediction + GenAI pipeline
├── data/
│   ├── Training_cleaned.csv    # Cleaned training set (4,920 rows)
│   ├── Training_dedup.csv      # Deduplicated unique records (304 rows)
│   ├── Testing_cleaned.csv     # Held-out test set (42 rows)
│   ├── disease_label_mapping.csv
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   └── Symptom-severity.csv
│   └── sample/                 # Small sample data only
├── notebooks/
│   ├── Ticket1_Data_Collection_EDA.ipynb
│   ├── Ticket2_Model_Development.ipynb
│   └── outputs/                # EDA and model comparison charts (PNG)
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd <repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running Locally

```bash
python app/app.py
```

---

## Sprint Deadlines

| Ticket | Focus | Deadline | Status |
|---|---|---|---|
| Ticket 1 | Data Collection & EDA | 02 Jul 2026 | ✅ Complete |
| Ticket 2 | Model Development & GenAI Integration | 09 Jul 2026 | ✅ Complete |
| Ticket 3 | Optimization, Flask Deployment & Presentation | 16 Jul 2026 | 🔄 In Progress |

---

## Ticket 1 — Data Collection & EDA

**Owner:** Milan TJ (lead), Nandana Venugopal, Sona Santhosh | **Completed:** 01 Jul 2026

### Datasets

**Primary:** [Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) (kaushil268, Kaggle)
4,920 records · 132 binary symptom features · 41 disease labels

**Supplementary:** [Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) (itachi9604, Kaggle)
Disease descriptions, precautions, and symptom severity weights — mapped to all 41 diseases and 132 symptoms. Used to enrich the GenAI prompt layer.

### Pipeline

1. **Schema Validation** — confirmed 132 symptom columns + 1 target; removed spurious trailing empty column (`Unnamed: 133`) present in raw export.
2. **Cleaning** — 0 nulls after column drop; all 132 features validated as clean binary 0/1; 4,616 duplicate rows identified (dataset is balanced at 120 rows/disease but only 304 truly unique symptom-combinations exist — documented for Ticket 2 overfitting check).
3. **Encoding** — `prognosis` label-encoded via `sklearn.LabelEncoder`; mapping saved to `disease_label_mapping.csv` for reuse in inference.
4. **NLP Preprocessing** — symptom column names normalised into natural-language phrases (e.g. `burning_micturition` → "burning micturition"); per-record `symptom_text` field generated as a GenAI prompt template.
5. **EDA** — 7 visualisations produced covering disease frequency, symptom co-occurrence heatmap, top symptoms, disease share pie chart, word cloud, average symptoms per disease, and severity distribution.

### Key Findings

- High-frequency symptoms (fatigue, vomiting, high fever, headache) appear across many diseases — supports ensemble models over linear ones.
- Jaundice-cluster symptoms (`yellowish_skin`, `dark_urine`, `yellowing_of_eyes`) show strong co-occurrence in the heatmap.
- Average symptom count varies noticeably by disease — Hepatitis family presents with fewer, more specific symptoms; Gastroenteritis and Typhoid present broader clusters.

### Outputs

| File | Description |
|---|---|
| `data/Training_cleaned.csv` | Full cleaned training set with encoding + symptom_text |
| `data/Training_dedup.csv` | 304 unique rows for overfitting comparison |
| `data/Testing_cleaned.csv` | Cleaned 42-row test set |
| `data/disease_label_mapping.csv` | LabelEncoder class ↔ integer mapping |
| `data/symptom_Description.csv` | Disease descriptions (41 diseases) |
| `data/symptom_precaution.csv` | 4 precautions per disease (41 diseases) |
| `data/Symptom-severity.csv` | Severity weight per symptom (132 symptoms, scale 1–7) |
| `notebooks/outputs/01–07_*.png` | 7 EDA visualisations |

---

## Ticket 2 — Model Development & GenAI Integration

**Owner:** Milan TJ (lead), Ajul ND, Sona Santhosh | **Completed:** 08 Jul 2026

### Model Results

| Model | Val Acc | Test Acc | Val F1 | Test F1 | CV Mean | ROC-AUC |
|---|---|---|---|---|---|---|
| Decision Tree | 1.0000 | 0.9762 | 1.0000 | 0.9837 | 0.9995 | 0.9936 |
| Random Forest | 1.0000 | 0.9762 | 1.0000 | 0.9837 | 1.0000 | 1.0000 |
| **Bernoulli NB** ✅ | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| SVM (RBF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 0.9762 | 1.0000 | 0.9837 | 0.9995 | 1.0000 |

**Selected Model: Bernoulli NB** — 100% test accuracy, perfect macro F1 and ROC-AUC, smallest serialised size (88 KB), fastest inference. Ideal for Flask deployment.

> **Overfitting check (deduplicated dataset):** Decision Tree drops to 59% F1 on unique symptom-combinations; Random Forest and SVM hold at 100%, confirming genuine generalisation rather than row memorisation. `random_forest.pkl` is included in the Flask app as a secondary option.

### Pipeline

1. Stratified 80/20 train/val split on full training data; Kaggle `Testing.csv` used as final held-out set.
2. Benchmarked on both full (4,920 rows) and deduplicated (304 rows) datasets.
3. 5-fold cross-validation on all models.
4. ROC-AUC computed via One-vs-Rest (OvR) multi-class approach.
5. Feature importance extracted from Random Forest and XGBoost for documentation.

### GenAI Integration

`get_recipe_recommendation(disease, symptoms)` builds a structured prompt enriched with disease description, precautions, and a computed severity score (average symptom weight from `Symptom-severity.csv`), then calls **Gemini 2.0 Flash** via the `google-genai` SDK to generate a personalised dietary recipe and lifestyle recommendations.

```python
# SDK: google-genai (not the deprecated google-generativeai)
from google import genai
client = genai.Client(api_key=GEMINI_API_KEY)
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt,
)
```

Set `GEMINI_API_KEY` as an environment variable or directly in the notebook. Falls back to a structured mock response if quota is exhausted or key is not set. Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### Outputs

| File | Description |
|---|---|
| `models/best_model.pkl` | Serialised Bernoulli NB model |
| `models/*.pkl` | All 5 trained models |
| `models/symptom_cols.json` | Ordered 132-feature list for inference |
| `models/disease_label_mapping.csv` | Encoded label ↔ disease name |
| `models/model_metadata.json` | Full metrics, model name, class names |
| `models/pipeline.py` | Importable prediction + GenAI recommendation pipeline |
| `notebooks/outputs/08_model_comparison.png` | Accuracy/F1 bar chart |
| `notebooks/outputs/09_cv_scores.png` | 5-fold CV scores with error bars |
| `notebooks/outputs/10_confusion_matrix.png` | Confusion matrix (best model, test set) |
| `notebooks/outputs/11_feature_importance.png` | RF + XGBoost feature importances |
| `notebooks/outputs/12_roc_auc.png` | ROC-AUC comparison |
| `notebooks/outputs/classification_report.csv` | Per-class precision/recall/F1 |

---

## Workflow

See [CONTRIBUTING.md](./CONTRIBUTING.md) for branching strategy, commit conventions, and PR process before pushing any code.

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · NLTK · Google Gemini API (`google-genai`) · Flask · Matplotlib · Seaborn · WordCloud · joblib · GitHub
