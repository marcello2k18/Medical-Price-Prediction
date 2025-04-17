from flask import Flask, render_template, request
import pandas as pd
import pickle
from model import MedicalInsuranceModel

# Load the medical insurance model
try:
    model = MedicalInsuranceModel.load_model('medical_insurance_model.pkl')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

app = Flask(__name__)

# Route to render the form
@app.route('/', methods=['GET'])
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form inputs
            age = int(request.form['age'])
            sex = request.form['sex']
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = request.form['smoker']
            region = request.form['region']

            # Prepare input data for prediction
            user_input = {
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'children': children,
                'smoker': smoker,
                'region': region
            }

            # Perform regression and classification predictions
            regression_predictions = model.predict_regression(user_input)
            classification_predictions = model.predict_classification(user_input)

            # Prepare the result for display
            result = {
                'regression_predictions': regression_predictions,
                'classification_predictions': classification_predictions
            }

            # Render predict.html with prediction result
            return render_template('predict.html', result=result)

        except Exception as e:
            return render_template('predict.html', error_message=str(e))

    # Render the initial form (GET request)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
