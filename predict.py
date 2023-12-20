import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'rf_model_diabetes.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('diabetes')

@app.route('/predict', methods=['POST'])
def predict():
    person = request.get_json()

    X = dv.transform([person])
    y_pred = model.predict_proba(X)[0, 1]
    diabetes = y_pred >= 0.5

    result = {
        'diabetes_probability': float(y_pred),
        'diabetes': bool(diabetes)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

