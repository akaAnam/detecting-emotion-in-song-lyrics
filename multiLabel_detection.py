#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:51:13 2022

@author: anamkhan
"""

# libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

## PART 1
##  context: bags of n-grams, tfidf unigrams

# -----
# TFIDF
# -----

# 1. 
# -- Text Preprocessing -- 

# function to pre-process the text. 
# This will do the things we don't do in tfidfVectorizer like removing numbers
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text



# create a duplicate of the multiLabel DF for manipulation
tfidf_multiLabel = multiLabel

# apply our preprocessing function to entire lyrics columns 
tfidf_multiLabel['lyrics'] = tfidf_multiLabel['lyrics'].apply(lambda x:pre_process(x))

# sanity check 
tfidf_multiLabel['lyrics'][2]



# 2. 
# -- Applying TfidfVectorizer() -- 

# instantiate vectorizer 
vectorizer = TfidfVectorizer(lowercase=True, analyzer="word", use_idf=True, stop_words='english')

# fit the transform 
X = vectorizer.fit_transform(tfidf_multiLabel.lyrics)

# see our results 
vectorizer.vocabulary_
X.toarray()


# save our tfidf results into their own DF
# https://stackoverflow.com/questions/45961747/append-tfidf-to-pandas-dataframe
multiLabel_featurenames = vectorizer.get_feature_names()
tfidf_df = pd.DataFrame(X.toarray(), columns=multiLabel_featurenames)

# drop first column since this is style and not context
multiLabel_featurenames[0]
tfidf_df = tfidf_df.drop(columns = ['____________'])


# 3. 
# -- Fitting Multi-label Models --


# Multi- LABEL classification of text 
# https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5


# How many songs have multiple labels? 
rowsums = multiLabel.iloc[:,1:].sum(axis=1)
x=rowsums.value_counts()

#plot
plt.figure(figsize=(8,5))
ax = sns.barplot(x.index, x.values)
plt.title("Multiple categories per song")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)





