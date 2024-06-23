import os
import json
import joblib
import pandas as pd
import pickle
from flask import Flask, jsonify, request
from peewee import (
    Model, BooleanField, CharField,
    TextField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import logging
import datetime


class CustomRailwayLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage()
        }
        return json.dumps(log_record)
    

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # this should be just "logger.setLevel(logging.INFO)" but markdown is interpreting it wrong here...
    handler = logging.StreamHandler()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = CustomRailwayLogFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger()

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    id = CharField(primary_key=True, max_length=50)
    observation = TextField()
    pred_class = BooleanField()
    true_class = BooleanField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)


########################################
# Unpickle the previously-trained model

pipeline = joblib.load('pipeline.pickle')

#with open('dtypes.pickle', 'rb') as fh:
#    dtypes = pickle.load(fh)


# End model un-pickling
########################################


# Manually Setting the Columns to be Recieved
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
    "is_recid",
    "r_case_number",
    "r_charge_degree",
    "r_charge_desc",
    "r_offense_date",
    "is_violent_recid",
    "vr_case_number",
    "vr_offense_date",
    "vr_charge_degree",
    "vr_charge_desc",

]

########################################

app = Flask(__name__)


def process_observation(observation):
    logger.info("Processing observation, %s", observation)
    
    observation['c_jail_out'] = datetime.datetime.now()
    # A lot of processing
    return observation


@app.route('/will_recidivate/', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    logger.info('Observation: %s', obs_dict)
    _id = obs_dict['id']
    observation = process_observation(obs_dict)

    if not _id:
        logger.warning('Returning error: no id provided')
        return jsonify({'error': 'id is required'}), 400
    if Prediction.select().where(Prediction.id == _id).exists():
        prediction = Prediction.get(Prediction.id == _id)

        # Update the prediction
        prediction.observation = str(observation)
        prediction.save()

        logger.warning('Returning error: already exists id %s', _id)
        return jsonify({'error': 'id already exists'}), 400

    try:
        observation['c_jail_time'] = None  # Will be calculated by CalculateJailTime transformer    
        
        obs = pd.DataFrame([observation], columns=columns)
        
    except ValueError as e:
        logger.error('Returning error: %s', str(e), exc_info=True)
        default_response = {'id': _id, 'outcome': False}
        return jsonify(default_response), 200
    
    outcome = bool(pipeline.predict(obs))
    response = {'id': _id, 'outcome': outcome}
    p = Prediction(
        id=_id,
        observation=request.data,
        pred_class=outcome,
    )
    p.save()
    logger.info('Saved: %s', model_to_dict(p))
    logger.info('Prediction: %s', response)

    return jsonify(response)


@app.route('/recidivism_result', methods=['POST'])
def update():
    obs = request.get_json()
    logger.info('Observation:', obs)
    _id = obs['id']
    outcome = obs['outcome']

    if not _id:
        logger.warning('Returning error: no id provided')
        return jsonify({'error': 'id is required'}), 400
    
    if not Prediction.select().where(Prediction.id == _id).exists():
        logger.warning(f'Returning error: id {_id} does not exist in the database')
        return jsonify({'error': 'id does not exist'}), 400
    
    p = Prediction.get(Prediction.id == _id)
    p.true_class = outcome
    p.save()
    logger.info('Updated: %s', model_to_dict(p))

    response = {'id': _id, 'outcome': outcome,'predicted_outcome': p.pred_class}
    return jsonify(response)


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8000)
