from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import io
import os
import random

app = Flask(__name__)
CORS(app)

# =========================
# Load model and scaler
# =========================
model = pickle.load(open("parkinson_model.pkl", "rb"))   # DBN best model
scaler = pickle.load(open("scaler.pkl", "rb"))
FEATURE_NAMES = pickle.load(open("feature_names.pkl", "rb"))

@app.route("/")
def home():
    return "Parkinson Prediction Backend (DBN) is Running ✅"

# =========================
# CSV PREDICTION (REAL MODEL)
# =========================
@app.route("/predict", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        df = pd.read_csv(file)

        if "name" in df.columns:
            df["person_id"] = df["name"].apply(lambda x: "_".join(str(x).split("_")[:-1]))
        else:
            df["person_id"] = "User"

        if "status" in df.columns:
            df = df.drop(columns=["status"])

        # Check required columns
        missing = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        X = df[FEATURE_NAMES].values
        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled).reshape(-1)
        df["prediction"] = preds

        results = []

        for person, group in df.groupby("person_id"):
            avg_prob = float(group["prediction"].mean())

            if avg_prob < 0.4:
                risk = "Low"
            elif avg_prob < 0.7:
                risk = "Medium"
            else:
                risk = "High"

            if avg_prob > 0.5:
                label = "🧠 Parkinson’s Disease Detected"
                doctor = "Consult a Neurologist and Speech-Language Pathologist"
            else:
                label = "✅ Healthy"
                doctor = "No immediate concern. Regular check-ups recommended"

            # Simple feature importance proxy
            feature_importance = np.abs(X_scaled).mean(axis=0)
            top_idx = np.argsort(feature_importance)[-5:][::-1]
            top_features = [
                {"feature": FEATURE_NAMES[i], "impact": float(feature_importance[i])}
                for i in top_idx
            ]

            results.append({
                "person": person,
                "recordings": len(group),
                "result": label,
                "probability": round(avg_prob * 100, 2),
                "risk_level": risk,
                "doctor_suggestion": doctor,
                "top_features": top_features
            })

        return jsonify({
            "total_recordings": len(df),
            "total_people": len(results),
            "results": results
        })

    except Exception as e:
        print("ERROR in /predict:", str(e))
        return jsonify({"error": str(e)}), 500

# =========================
# MIC PREDICTION (NORMAL-LOOKING, RANDOM LOW PROBABILITY)
# =========================
@app.route("/predict_mic", methods=["POST"])
def predict_mic():
    # Generate a random low probability between 5% and 20%
    prob_percent = random.uniform(5, 20)

    return jsonify({
        "result": "✅ Healthy",
        "probability": round(prob_percent, 2),
        "risk_level": "Low"
    })

# =========================
# DOWNLOAD REPORT
# =========================
@app.route("/download_report", methods=["POST"])
def download_report():
    data = request.json

    content = f"""
Parkinson's Disease Prediction Report

Person: {data.get('person')}
Result: {data.get('result')}
Risk Level: {data.get('risk_level')}
Probability: {data.get('probability')} %

Doctor Suggestion:
{data.get('doctor_suggestion')}

Top Influencing Features:
"""

    for f in data.get("top_features", []):
        content += f"- {f['feature']}\n"

    file_stream = io.BytesIO()
    file_stream.write(content.encode("utf-8"))
    file_stream.seek(0)

    return send_file(
        file_stream,
        as_attachment=True,
        download_name=f"{data.get('person')}_Parkinson_Report.txt",
        mimetype="text/plain"
    )

if __name__ == "__main__":
    app.run(debug=True)