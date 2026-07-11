import numpy as np
import pandas as pd
import json

def predict_disease(symptom_list, model=best_model, symptom_cols=SYMPTOM_COLS,
                    label_map_df=label_map):
    """
    Given a list of active symptom names, returns:
      - predicted disease name
      - confidence score (max class probability)
      - top-3 disease predictions with probabilities
    """
    # Build feature vector
    vec = np.zeros(len(symptom_cols))
    for sym in symptom_list:
        sym_clean = sym.strip().lower().replace(' ', '_')
        if sym_clean in symptom_cols:
            vec[symptom_cols.index(sym_clean)] = 1

    proba = model.predict_proba([vec])[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    # Map encoded label back to disease name
    le_map = dict(zip(label_map_df['encoded_label'], label_map_df['disease']))
    top3 = [(le_map[i], round(float(proba[i]), 4)) for i in top3_idx]

    return {
        'predicted_disease': top3[0][0],
        'confidence': top3[0][1],
        'top_3_predictions': top3
    }


def get_recipe_recommendation(disease, symptoms,
                               desc_df=desc_df, prec_df=prec_df, sev_df=sev_df):

    disease_stripped = disease.strip()

    desc_row = desc_df[desc_df['Disease'].str.strip() == disease_stripped]
    description = desc_row['Description'].values[0] if len(desc_row) else 'No description available.'

    prec_row = prec_df[prec_df['Disease'].str.strip() == disease_stripped]
    precautions = [prec_row.iloc[0].get(f'Precaution_{i}', '') for i in range(1, 5)] \
                  if len(prec_row) else ['Consult a doctor', 'Rest', 'Stay hydrated', 'Maintain hygiene']

    sym_weights = sev_df.set_index('Symptom')['weight'].to_dict()
    severity_score = np.mean([sym_weights.get(s, 3) for s in symptoms]) if symptoms else 3.0

    prompt = build_recipe_prompt(disease_stripped, symptoms, description, precautions, severity_score)

    if GENAI_AVAILABLE:
        response = client.models.generate_content(
            # model='gemini-2.0-flash',        # ← updated model name
            model='gemini-2.0-flash-lite',
            contents=prompt,
        )
        recommendation = response.text
        source = 'Gemini API (gemini-2.0-flash)'
    else:
        recommendation = f"""**Health Overview**
{disease_stripped} requires dietary adjustments and lifestyle changes.

**Recipe: Nourishing Recovery Broth**
- Ingredients: ginger, turmeric, garlic, lemon, vegetable broth, spinach
- Simmer 15 min, serve warm.
- Why it helps: anti-inflammatory support for {disease_stripped}.

**Lifestyle Tips**
1. Drink 8–10 glasses of water daily.
2. Sleep 7–8 hours for immune restoration.
3. Avoid processed food; prioritise whole grains.

*(MOCK mode — connect Gemini API for personalised responses)*"""
        source = 'Mock (offline)'

    return {
        'disease': disease_stripped,
        'symptoms': symptoms,
        'severity_score': round(severity_score, 2),
        'description': description,
        'precautions': precautions,
        'recommendation': recommendation,
        'source': source,
    }


def build_recipe_prompt(disease, symptoms, description, precautions, severity_score):
    prec_text = ' | '.join([p for p in precautions if p and str(p).lower() != 'nan'])
    return f"""You are a clinical nutritionist and wellness advisor AI.

    A patient has been predicted to have: **{disease}**

    Disease description: {description}

    Symptoms reported: {', '.join(symptoms)}

    Recommended general precautions: {prec_text}

    Average symptom severity score: {severity_score:.1f}/7

    Please provide:
    1. A short health overview for this condition (2-3 sentences).
    2. A specific dietary recipe recommendation that supports recovery or management (include: dish name, key ingredients, brief preparation steps, and why it helps).
    3. Three lifestyle tips specific to managing {disease}.

    Keep the tone professional but approachable. Format clearly with numbered sections.
    """


def disease_prediction_pipeline(symptom_list):
    """
    Full end-to-end pipeline:
      Input  : list of symptom name strings
      Output : dict with disease prediction + confidence + top3 + GenAI recommendation
    """
    pred = predict_disease(symptom_list)
    recipe = get_recipe_recommendation(pred['predicted_disease'], symptom_list)
    return {
        'predicted_disease':  pred['predicted_disease'],
        'confidence':         pred['confidence'],
        'top_3_predictions':  pred['top_3_predictions'],
        'description':        recipe['description'],
        'precautions':        recipe['precautions'],
        'severity_score':     recipe['severity_score'],
        'recommendation':     recipe['recommendation'],
        'genai_source':       recipe['source'],
    }
