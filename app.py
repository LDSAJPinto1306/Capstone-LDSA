import os
import json
import joblib
import pandas as pd
import pickle
from flask import Flask, jsonify, request
from peewee import Model, BooleanField, CharField, TextField
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import logging
import datetime

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# Connect to database
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

# Define Prediction model
class Prediction(Model):
    obs_id = TextField(unique=True)
    observation = TextField()
    pred_class = BooleanField()
    true_class = BooleanField(null=True)

    class Meta:
        database = DB

# Create tables if not exists
DB.create_tables([Prediction], safe=True)

# Load machine learning model
pipeline = joblib.load('pipeline.pickle')

# Define columns expected in incoming data
columns = [
    "id",
    "name",
    "sex",
    "dob",
    "race",
    "juv_fel_count", 
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_case_number",
    "c_charge_degree",
    "c_charge_desc",
    "c_offense_date",
    "c_arrest_date",
    "c_jail_in",
    "is_recid",  # Ensure this matches the key in incoming JSON
]

# Function to process observation (customize as per your requirements)
def process_observation(observation):
    logger.info("Processing observation: %s", observation)
    # Additional processing logic can go here
    return observation

# Endpoint for making predictions
@app.route('/will_recidivate/', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    logger.info('Received observation: %s', obs_dict)
    _id = obs_dict.get('id')

    if not _id:
        logger.warning('No id provided in request')
        return jsonify({'error': 'id is required'}), 400

    # Process the observation
    observation = process_observation(obs_dict)

    # Check if the ID already exists in the database
    if Prediction.select().where(Prediction.obs_id == _id).exists():
        logger.warning('ID %s already exists in the database', _id)
        return jsonify({'error': 'id already exists'}), 400

    try:
        obs_df = pd.DataFrame([observation], columns=columns)
    except ValueError as e:
        logger.error('ValueError occurred: %s', str(e))
        return jsonify({'error': str(e)}), 400

    # Make prediction using the model
    outcome = bool(pipeline.predict(obs_df))
    response = {'id': _id, 'outcome': outcome}
    
    # Save prediction to database
    p = Prediction(obs_id=_id,pred_class=outcome,observation=json.dumps(obs_dict))
    try:
        p.save()
        logger.info('Stored Prediction %s',p)
    except:
        error_msg = '\n\n\n !!! Failed to Store Prediction !!! \n\n\n'
        logger.warning(error_msg)
        return error_msg
    
    return jsonify(response)

# Endpoint for updating true class
@app.route('/recidivism_result', methods=['POST'])
def update():
    obs = request.get_json()
    logger.info('Received update request: %s', obs)
    _id = obs.get('id')
    outcome = obs.get('outcome')

    if not _id:
        logger.warning('No id provided in request')
        return jsonify({'error': 'id is required'}), 400

    # Check if the ID exists in the database
    if not Prediction.select().where(Prediction.obs_id == _id).exists():
        logger.warning('ID %s does not exist in the database', _id)
        return jsonify({'error': 'id does not exist'}), 400

    # Update true class
    p = Prediction.get(Prediction.obs_id == _id)
    p.true_class = outcome
    p.save()
    logger.info('Updated true class for ID %s', _id)

    response = {'id': _id, 'outcome': outcome, 'predicted_outcome': p.pred_class}
    return jsonify(response)

# Endpoint to list database contents
@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8000)
