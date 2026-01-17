from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load ML model
model = joblib.load("triage_model.pkl")

@app.route("/")
def home():
    return "Backend is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [
        data["Fever"],
        data["Cough"],
        data["Fatigue"],
        data["Difficulty_Breathing"],
        data["Age"],
        data["Gender"],
        data["Blood_Pressure"],
        data["Cholesterol_Level"]
    ]

    prediction = model.predict([features])[0]

    urgency_map = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return jsonify({"Urgency_Level": urgency_map[prediction]})

if __name__ == "__main__":
    app.run(debug=True)
