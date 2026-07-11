from flask import Flask, render_template, request
from predictor import (
    predict,
    get_description,
    get_precautions,
    get_ai_recommendation,
    symptom_cols
)
app = Flask(__name__)

@app.route("/")
def home():
    return render_template(
    "home.html",
    symptoms=sorted(symptom_cols)
    )

@app.route("/predict", methods=["POST"])
def predict_route():

    symptom_list = request.form.getlist("symptoms")

    #symptom_list = [
    #   s.strip().lower().replace(" ", "_")
    #    for s in symptoms.split(",")
    #   if s.strip()
    #]
    if not symptom_list:
        return render_template(
        "home.html",
        symptoms=sorted(symptom_cols),
        error="Please select at least one symptom."
        )

    disease, confidence, top3 = predict(symptom_list)

    description = get_description(disease)

    ai_response = get_ai_recommendation(
    disease,
    description
    )

    precautions = get_precautions(disease)

    return render_template(
    "result.html",
    symptoms=symptom_list,
    disease=disease,
    confidence=confidence,
    top3=top3,
    description=description,
    precautions=precautions,
    ai_response=ai_response
    )

if __name__ == "__main__":
    app.run(debug=True)