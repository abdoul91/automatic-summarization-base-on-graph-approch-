#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.corpus   import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def preprocess(sentence) :
    stop_words = stopwords.words('french')
    words      = word_tokenize(sentence)
    tokens     = [word.lower() for word in words]
    table      = str.maketrans('', '', string.punctuation)
    stripped   = [w.translate(table) for w in tokens]
    sent       =  ' '.join([word for word in stripped if word.isalnum() and 
                            word not in stop_words])
    return sent
    

def tfIdf(corpus) :
    corpus    = sent_tokenize(corpus)
    documents = [preprocess(sent) for sent in corpus]
    tfidf     = TfidfVectorizer()
    tfIdf_mat = tfidf.fit_transform(documents)
    df = pd.DataFrame(tfIdf_mat.todense(), columns=tfidf.get_feature_names())
    df['p_sentence'] = [word_tokenize(sent) for sent in documents]
    df['sentences']  = corpus
    return df
    

