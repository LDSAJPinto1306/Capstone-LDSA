import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError,BooleanField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect
import warnings
warnings.filterwarnings('ignore')



