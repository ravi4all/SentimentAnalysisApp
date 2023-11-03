from django.shortcuts import render
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle as pkl

# Create your views here.
def index(request):
    return render(request, "index.html")

def textProcessing(df):
    table = str.maketrans('','',string.punctuation)
    for i in range(len(df)):
        df['Review'].iloc[i] = df['Review'].iloc[i].lower().translate(table)
    
    documents = []
    # word tokenization
    for i in range(len(df)):
        documents.append(word_tokenize(df['Review'].iloc[i]))
        
    englishStopwords = stopwords.words("english")
    words = []
    for tokens in documents:
        word = []
        for i in range(len(tokens)):
            if tokens[i] not in englishStopwords:
                word.append(tokens[i])
        words.append(word)
        
    wnet = WordNetLemmatizer()
    for i in range(len(words)):
        for j in range(len(words[i])):
            words[i][j] = wnet.lemmatize(words[i][j], 'v')
            
    for i in range(len(words)):
        words[i] = " ".join(words[i])
    
    return words

def load_pickle(filename):
    file = open(filename, "rb")
    obj = pkl.load(file)
    file.close()
    return obj

def do_prediction(request):
    tfidf = load_pickle("tfidf.pkl")
    logistic = load_pickle("model.pkl")
    # test_sentence = "This movie was very bad"
    test_sentence = request.GET["review"]
    test_df = pd.DataFrame({'Review' : [test_sentence]})
    test_data = textProcessing(test_df)
    test_tf = tfidf.transform(test_data).toarray()
    prediction = logistic.predict(test_tf)
    if prediction[0] == 0:
        msg = "Negative Review"
    else:
        msg = "Positive Review"
    return render(request, "prediction.html", {"prediction" : msg})