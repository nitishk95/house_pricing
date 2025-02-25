from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load Data and Model
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))  # Correct float conversion
    bath = float(request.form.get('bath'))  # Correct float conversion
    sqft = float(request.form.get('total_sqft'))  # Convert to float for numerical processing

    # Create DataFrame for prediction
    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Predict house price
    prediction = pipe.predict(input_data)[0] * 1e5 # Correct indexing

    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
