import json
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load model
model = joblib.load("models/best_model.pkl")

# Load symptom list
with open("models/symptom_cols.json", "r") as f:
    symptom_cols = json.load(f)

# Load disease label mapping
label_map = pd.read_csv("models/disease_label_mapping.csv")

# Load disease descriptions
description_df = pd.read_csv("data/symptom_Description.csv")

# Load precautions
precaution_df = pd.read_csv("data/symptom_precaution.csv")


def predict(symptom_list):

    features = np.zeros(len(symptom_cols))

    for symptom in symptom_list:
        symptom = symptom.strip().lower().replace(" ", "_")

        if symptom in symptom_cols:
            index = symptom_cols.index(symptom)
            features[index] = 1

    probabilities = model.predict_proba([features])[0]

    prediction = np.argmax(probabilities)

    confidence = probabilities[prediction] * 100

    disease = label_map.loc[
        label_map["encoded_label"] == prediction,
        "disease"
    ].values[0]

    top3_indices = np.argsort(probabilities)[::-1][:3]

    top3 = []

    for idx in top3_indices:

        disease_name = label_map.loc[
            label_map["encoded_label"] == idx,
            "disease"
        ].values[0]

        top3.append({
            "disease": disease_name,
            "confidence": round(probabilities[idx] * 100, 2)
        })

    return disease, round(confidence,2), top3

def get_description(disease):

    row = description_df[
        description_df["Disease"].str.strip() == disease.strip()
    ]

    if len(row) == 0:
        return "Description not available."

    return row.iloc[0]["Description"]


def get_precautions(disease):

    row = precaution_df[
        precaution_df["Disease"].str.strip() == disease.strip()
    ]

    if len(row) == 0:
        return []

    return [
        row.iloc[0]["Precaution_1"],
        row.iloc[0]["Precaution_2"],
        row.iloc[0]["Precaution_3"],
        row.iloc[0]["Precaution_4"],
    ]

def get_ai_recommendation(disease, description):

    prompt = f"""
You are a professional clinical nutritionist.

A patient has been predicted with:

Disease: {disease}

Description:
{description}

Give the response in the following format:

Health Overview:
(2-3 lines)

Foods to Eat:
- item
- item
- item

Foods to Avoid:
- item
- item
- item

Recommended Recipe:
Recipe Name

Ingredients:
- ...

Preparation:
...

Lifestyle Tips:
- ...
- ...
- ...

Keep the response easy to understand.
"""

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt
    )

    return response.text