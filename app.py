from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_detection_model.pkl")  # Save model after training and push to GitHub

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Expect JSON input
        df = pd.DataFrame([data])  # Convert to DataFrame
        prediction = model.predict(df)[0]  # Make prediction
        return jsonify({"fraud_prediction": int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
