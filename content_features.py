#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anamkhan, swami venkateswaran, harrison lee
"""

# libraries
import argparse
import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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
from nltk.corpus import stopwords

def main(multiLabel):
    # Reading in cleaned data
    print("Running content_features.py\n")
    multiLabel = pd.read_csv(multiLabel)

    multiLabel.head()

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


    # --------------------------
    # -- Get the top features --
    # --------------------------

    # 1.
    # define function to get top n features
    # https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
    def get_top_n_words(corpus, n=None):

        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                    for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]


    # 2.
    # Remove stop words
    stop = stopwords.words('english')

    # Exclude stopwords with Python's list comprehension and pandas.DataFrame.apply.
    tfidf_multiLabel['lyrics_without_stopwords'] = tfidf_multiLabel['lyrics']
    tfidf_multiLabel['lyrics_without_stopwords'] = tfidf_multiLabel['lyrics_without_stopwords'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    # sanity check
    tfidf_multiLabel["lyrics_without_stopwords"][1]

    # 3.
    # get top n features excluding stopwords
    top10_no_stop = get_top_n_words(
        tfidf_multiLabel['lyrics_without_stopwords'], 10)

    top10_no_stop


    # 4.
    # Plot
    word = []
    freq = []
    for item in top10_no_stop:
        word.append(item[0])
        freq.append(item[1])

    sns.barplot(word, freq, palette="Blues_r")
    plt.title("Top 10 Most Frequent Words in Song Lyrics")
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Words', fontsize=12)

    plt.savefig("visuals/top10_words_nostop.png", dpi=900)
    plt.show()


    ####################
    ## MODEL TRAINING ##
    ##  multi-label   ##
    ####################

    # -------------------
    # -------------------
    # LOGISTIC REGRESSION
    # -------------------
    # -------------------


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

    # for table creation
    LR_accuracy = []
    LR_f1 = []
    print("\n???????????????????????????????????????")
    print("LOGISTICAL REGRESSION RESULTS")
    print("???????????????????????????????????????\n")

    for emotion in emotions:
        print('Processing {}'.format(emotion), "... ")

        # train the model using X_dtm & y
        pipeline.fit(X_train, train[emotion])

        # compute the testing accuracy
        prediction = pipeline.predict(X_test)
        LR_accuracy.append(accuracy_score(test[emotion], prediction))
        LR_f1.append(f1_score(test[emotion], prediction))
        print('Test accuracy is {}'.format(
            accuracy_score(test[emotion], prediction)))

        print('Test F1 is: {}'.format(f1_score(test[emotion], prediction)))
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


    LR_f1_allAtOnce = f1_score(all_y_test, y_pred, average="micro")
    LR_accuracy_allAtOnce = accuracy_score(all_y_test, y_pred)


    # -----------
    # -----------
    # NAIVE BAYES
    # -----------
    # -----------


    # 1.
    # -- Format labels for multi label classification--

    # ~ done above ~


    # 2.
    # -- Define a pipeline --

    nb = MultinomialNB()

    ovr = OneVsRestClassifier(nb)

    pipelineNB = make_pipeline(vectorizer, ovr)


    # 3.
    # -- Create test & train data --

    # ~ done above ~


    # # 4.
    # # -- train & predict --

    # for table creation
    NB_accuracy = []
    NB_f1 = []
    print("???????????????????????????????????????")
    print("NAIVE BAYES RESULTS")
    print("???????????????????????????????????????\n")
    for emotion in emotions:
        print('Processing {}'.format(emotion), "... ")

        # train the model using X_dtm & y
        pipelineNB.fit(X_train, train[emotion])

        # compute the testing accuracy
        prediction = pipelineNB.predict(X_test)
        NB_accuracy.append(accuracy_score(test[emotion], prediction))
        NB_f1.append(f1_score(test[emotion], prediction))
        print('Test accuracy is: {}'.format(
            accuracy_score(test[emotion], prediction)))
        print('Test F1 is: {}'.format(f1_score(test[emotion], prediction)))
        print('\n')


    results = pd.DataFrame({'LR Accuracy': LR_accuracy,
                            'LR F1': LR_f1,
                            'NB Accuracy': NB_accuracy,
                            'NB F1': NB_f1}, index=emotions)


    # ALL CATEGORIES AT ONCE ##

    # 4.
    # -- train & predict --

    pipelineNB.fit(all_X_train, all_y_train)
    y_pred = pipelineNB.predict(all_X_test)


    f1_score(all_y_test, y_pred, average="micro")
    accuracy_score(all_y_test, y_pred)

    NB_f1_allAtOnce = f1_score(all_y_test, y_pred, average="micro")
    NB_accuracy_allAtOnce = accuracy_score(all_y_test, y_pred)

    results_allAtOnce = pd.DataFrame({'Accuracy': [LR_accuracy_allAtOnce, NB_accuracy_allAtOnce],
                                    'F1': [LR_f1_allAtOnce, NB_f1_allAtOnce]},
                                    index=['Logistic Regression', 'Naive Bayes'])


    #print(results)
    # exporting to csv for table formatting outside of python
    # results.to_csv('data/content_results.csv')
    # results_allAtOnce
    # results_allAtOnce.to_csv('data/content_results_allAtOnce.csv')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--multiLabel", type=str, default="data/cleaned_multiLabel.csv",
                        help="multi label data file")
    args = parser.parse_args()
    main(args.multiLabel)