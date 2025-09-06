# Step 15: Flask API for real-time prediction
from flask import Flask, request, jsonify,redirect, render_template
import joblib
import pandas as pd
import shap
import csv
import os    

app = Flask(__name__)

@app.route("/")
def home():
    return redirect("/index")


# Load the trained model
model = joblib.load('credit_fraud_xgb_model.pkl')



@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        # ---- API request ----
        json_data = request.get_json(force=True)
        input_df = pd.DataFrame([json_data], dtype=float)
        prediction = model.predict(input_df)

        # Logging
        log_file = "predictions_log.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(list(json_data.keys()) + ["prediction"])
            writer.writerow(list(json_data.values()) + [int(prediction[0])])

        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        top_features = pd.Series(shap_values[0], index=input_df.columns).nlargest(3).to_dict()

        return jsonify({
            "fraud_prediction": int(prediction[0]),
            "top_influences": top_features
        })

    else:
        # ---- Form submission from HTML ----
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data], dtype=float)
        prediction = model.predict(input_df)

        # Logging
        log_file = "predictions_log.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(list(form_data.keys()) + ["prediction"])
            writer.writerow(list(form_data.values()) + [int(prediction[0])])

        return render_template("index.html", result=int(prediction[0]))



@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if os.path.isfile("predictions_log.csv"):
        df = pd.read_csv("predictions_log.csv")
        print("📊 Dashboard DataFrame:\n", df.head())
        tables = df.to_dict(orient="records")
        print("📊 Tables sent to Jinja:", tables[:3])
    else:
        tables = []
        print("⚠️ predictions_log.csv not found")
    
    return render_template("dashboard.html", tables=tables)

@app.route('/get_logs')
def get_logs():
    if os.path.isfile("predictions_log.csv"):
        df = pd.read_csv("predictions_log.csv")
        return df.to_json(orient='records')
    return jsonify([])



if __name__ == '__main__':
    print("Registered routes:", [str(r) for r in app.url_map.iter_rules()])
    app.run(debug=True)
