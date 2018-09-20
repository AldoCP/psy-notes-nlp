#from flask import Flask
from flask import Flask, render_template, request
from get_letters import get_letters
app = Flask(__name__)
#@app.route('/')
#def hello():
#    return "Draft of psy-notes-nlp app."

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import string
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, accuracy_score, roc_auc_score
import datetime
import xgboost as xgb
from sklearn.metrics import confusion_matrix, auc, accuracy_score, roc_auc_score
import pickle
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, train_test_split
from sklearn import metrics    #Additional scklearn functions
import multiprocessing

def tokenizer_better(text):
        punc_list = string.punctuation+'0123456789'
        text = "".join([ch for ch in text if ch not in punc_list])
        tokens = word_tokenize(text)
        return tokens

def pred_one(anote, prep_model, ml_model, missing_val=-99.0):
	anote = prep_model.transform([anote])
	anote = xgb.DMatrix(anote, missing=missing_val, feature_names=prep_model.get_feature_names())
	return ml_model.predict(anote)[0]

loaded_prep_model = pickle.load(open('vect.pickle', 'rb'))
loaded_ml_model = pickle.load(open('bst.pickle', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    #trends = ''
    letters = []
    result = []
    if request.method == "POST":
        # get url that the user has entered
        try:
            word = request.form['word']
            letters = get_letters(word)
            result = [pred_one(word, loaded_prep_model, loaded_ml_model)]
            #result = [len(word)]# print statements just print to terminal
            #print("word was:")
            #print(word)
        except:
            errors.append(str(type(word))+str(type(loaded_prep_model)) );#"This is an error message. (Unable to get URL. Please make sure it's valid and try again.)")
            #errors.append(pred_one(word, loaded_prep_model, loaded_ml_model) )
            #print("error")
    #return render_template('index.html')
    return render_template('index.html', letters=result, errors=errors)

if __name__ == '__main__':
    app.run()
