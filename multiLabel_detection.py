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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


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

# PART 1
# context: bags of n-grams, tfidf unigrams

# -----
# TFIDF
# -----

# 1.
# -- Text Preprocessing --

# function to pre-process the text.
# This will do the things we don't do in tfidfVectorizer like removing numbers
def pre_process(text):

    # lowercase
    text = text.lower()

    # remove tags
    text = re.sub("", "", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    return text


# create a duplicate of the multiLabel DF for manipulation
tfidf_multiLabel = multiLabel

# apply our preprocessing function to entire lyrics columns
tfidf_multiLabel['lyrics'] = tfidf_multiLabel['lyrics'].apply(
    lambda x: pre_process(x))

# sanity check
tfidf_multiLabel['lyrics'][2]


# 2.
# -- Applying TfidfVectorizer() --

# instantiate vectorizer
vectorizer = TfidfVectorizer(
    lowercase=True, analyzer="word", use_idf=True, stop_words='english')

# fit the transform
X = vectorizer.fit_transform(tfidf_multiLabel.lyrics)

# see our results
vectorizer.vocabulary_
X.toarray()


# save our tfidf results into their own DF to check work
# https://stackoverflow.com/questions/45961747/append-tfidf-to-pandas-dataframe
multiLabel_featurenames = vectorizer.get_feature_names()
tfidf_df = pd.DataFrame(X.toarray(), columns=multiLabel_featurenames)

# drop first column since this is style and not context
multiLabel_featurenames[0]
tfidf_df = tfidf_df.drop(columns=['____________'])


####################
## MODEL TRAINING ##
##  multi-label   ##
####################


# Multi- LABEL classification of text
# https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5


# How many songs have multiple labels?
rowsums = multiLabel.iloc[:, 1:].sum(axis=1)
x = rowsums.value_counts()

# plot
plt.figure(figsize=(8, 5))
ax = sns.barplot(x.index, x.values)
plt.title("Multiple categories per song")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)


# 1.
# -- Format labels for multi label classification--

# Create an empty list
labels = []

# Iterate over each row
for index, rows in tfidf_multiLabel.iterrows():
    # Create list for the current row
    my_list = [rows.amazement, rows.calmness, rows.joyful, rows.nostalgia,
               rows.power, rows.sadness, rows.solemnity, rows.tenderness, rows.tension]

    # append the list to the final list
    labels.append(my_list)


# 2.
# -- Define a pipeline --

lr = LogisticRegression(solver='sag')

# Why one vs rest?
""" The Multi-label algorithm accepts a binary mask over multiple labels.
 The result for each prediction will be an array of 0s and 1s marking which
 class labels apply to each row input sample """

ovr = OneVsRestClassifier(lr)
pipeline = make_pipeline(vectorizer, ovr)


# 3.
# -- Create test & train data --

train, test = train_test_split(
    tfidf_multiLabel, random_state=42, test_size=0.25, shuffle=True)

X_train = train.lyrics
X_test = test.lyrics

print(X_train.shape)
print(X_test.shape)


# # 4.
# # -- train & predict --

emotions = ["amazement", "calmness", "joyful", "nostalgia",
            "power", "sadness", "solemnity", "tenderness", "tension"]

for emotion in emotions:
    print('Processing {}'.format(emotion), "... ")

    # train the model using X_dtm & y
    pipeline.fit(X_train, train[emotion])

    # compute the testing accuracy
    prediction = pipeline.predict(X_test)
    print('Test accuracy is {}'.format(
        accuracy_score(test[emotion], prediction)))
    print('Test F1 is {}'.format(f1_score(test[emotion], prediction)))
    print('\n')


# ALL CATEGORIES AT ONCE ##

# 3.
# -- Create test & train data --

lyrics = tfidf_multiLabel['lyrics']
# sanity check
lyrics[0:4]
labels[0:4]

all_X_train, all_X_test, all_y_train, all_y_test = train_test_split(
    lyrics, labels, test_size=0.25, random_state=42)


# 4.
# -- train & predict --

pipeline.fit(all_X_train, all_y_train)
y_pred = pipeline.predict(all_X_test)


f1_score(all_y_test, y_pred, average="micro")
accuracy_score(all_y_test, y_pred)

# F1: 0.4471256210078069
# Accuracy: 0.06896551724137931
