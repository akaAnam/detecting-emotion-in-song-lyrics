#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 01:29:13 2022

@author: anamkhan
"""

# link to dataset: https://github.com/imdiptanu/lyrics-emotion-detection

# libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize # for stylometric features
from nltk.corpus import stopwords
import nltk
import string


# Reading in cleaned data
multiLabel = pd.read_csv("data/cleaned_multiLabel.csv")

multiLabel.head()

# Subset our dataframe to only contain relevant columns for this task
multiLabel = multiLabel.drop(columns=['artist', 'genre', 'song', 'year'])

# dropping activation label - not needed
multiLabel = multiLabel.drop(columns=['activation'])


########################
## FEATURE EXTRACTION ##
########################

# PART 2
# stylometric features: punctuation, function words, digits 

# -----

# create a duplicate of the multiLabel DF for manipulation
style_multiLabel = multiLabel

# 1. 
# replace "\n" with " "
# We don't want the back slashes to be interpreted as punctuation and mess w/
# our analyses. new line characters are only here because of the data format
for i in range(len(style_multiLabel)):
    style_multiLabel['lyrics'][i] = style_multiLabel['lyrics'][i].replace("\n", " ")

# sanity check
style_multiLabel['lyrics'][4]

# 2. 
# tokenize each song's lyrics and save as a new col --> tok_lyrics
# https://machinelearningmastery.com/clean-text-machine-learning-python/
style_multiLabel['tok_lyrics'] = style_multiLabel.apply(lambda row: nltk.word_tokenize(row['lyrics']), axis=1)

# ** potential issues with work_tokenize()** : 
#       - for some reason some punctuation isnt tokenizing 
#          getting "wo" and "n't" and seperate tokens 
#          need to figure out how to isolate apostraphes 
#       - words like "heee-eeee-eey" are also not tokenizing on punctuation
#           need to isolate the hyphens 

# saves a list of tokens in each row 
style_multiLabel['tok_lyrics'][4]






    





