from flask import Flask, jsonify, request
import pickle
import pandas as pd
from my_transformers import DenseTransformer

with open('pipeline_svm.pkl', 'rb') as f:
    pipeline_svm = pickle.load(f)
    
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the POST request
    data = request.get_json(force=True)

    
    # Convert data into pandas DataFrame
    data_df = pd.DataFrame(data, index=[0]) 

    # Make predictions using the trained model
    predictions = pipeline_svm.predict(data_df)

    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())



if __name__ == '__main__':
    app.run(debug=True)