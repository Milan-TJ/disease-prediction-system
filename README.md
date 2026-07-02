# Disease Prediction System (Team 2 — iDatalytics/RP2)

ML-based disease prediction with a generative AI recipe recommendation layer, deployed as a Flask app.

## Team
| Name |
|---|
| Milan TJ |
| Sona Santhosh|
| Nandana Venugopal|
| Ajul ND|
|P Padm Sarang |

## Project structure
```
├── app/                # Flask application
├── models/              # Model training code (weights NOT committed — see CONTRIBUTING.md)
├── data/
│   └── sample/          # Small sample data only
├── notebooks/           # Exploration / EDA notebooks
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
└── README.md
```

## Setup
```bash
git clone <repo-url>
cd <repo-name>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running locally
```bash
python app/app.py
```

## Workflow
See [CONTRIBUTING.md](./CONTRIBUTING.md) for branching, commit conventions, and PR process before pushing any code.

## Sprint deadlines
Sprints run through mid-July 2026 — see team board for current sprint tasks.
