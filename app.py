import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load your model
with open("api.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    disease_mapping = {
        0: "Cough",
        1: "Pneumonia",
        2: "Tuberculosis (TB)",
        3: "Chronic obstructive pulmonary disease (COPD)",
        4: "Bronchitis",
        5: "Flu",
        6: "Asthma",
        # Add more mappings as needed
    }



    # Extract input features from form
    try:
        feat = [
            int(request.form['Gender']),
            int(request.form['Age']),
            int(request.form['Diagnosis Age']),
            int(request.form['Blood Group']),
            int(request.form['Birth Order']),
            int(request.form['Marital Status']),
            int(request.form['Lifestyle']),
            int(request.form['Weight']),
            int(request.form['Obesity']),
            int(request.form['Obesity In Family']),
            int(request.form['Fast Food']),
            int(request.form['Smoking']),
            int(request.form['High BP']),
            int(request.form['Diabetes']),
        ]


        features = np.array(feat).reshape(1, -1)
        prediction = model.predict(features)[0]  # Get the prediction
        
        predicted_disease = disease_mapping.get(prediction, "Unknown Disease")  # Get disease name from mapping
        
        return render_template("result.html", prediction_text=f"Might be possible that you have {predicted_disease}")
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
