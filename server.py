import json
import pandas as pd
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import re 
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from flask import Flask,render_template,url_for,request
import dill as pickle
filename = 'model_v1.pk'

with open(filename ,'rb') as f:
    loaded_model = pickle.load(f)

vectorizer = CountVectorizer(stop_words = 'english',ngram_range=(1,2))
sw = set(stopwords.words('english'))
ps = PorterStemmer()

def clean(rev):    
    rev = rev.lower()
    p = re.findall(r"@\w+",rev)
    try:
        s = rev.replace(p[0],'')
        s = s.replace('rt :','')
    except:
        s = rev
    for i in range(len(s)):
        if s[i:i+5] == 'https':
            s = s[:i]

    t = word_tokenize(s)
    t_ = [token for token in t if token not in sw]

    word = " ".join(t_)
    return word

def preProcess(text):
    text = [clean(text)]
    return text

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)
    
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        data = preProcess(message)
        my_prediction = int(loaded_model.predict(data))
        print(my_prediction)
    if my_prediction == 0:
        return "Neutral"
    elif my_prediction == -1:
        return "Negative"
    elif my_prediction == 1:
        return "Positive"

if __name__ == '__main__':
	app.run(debug=True)

    