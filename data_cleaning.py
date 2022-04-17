# libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# Links to Datasets:

# multi-label dataset - 1,160 songs with multiple emotion labels per song
# https://github.com/imdiptanu/lyrics-emotion-detection

# single-label - 1,160 songs with 1 emotion label per song
# https://github.com/imdiptanu/lyrics-emotion-detection

# Edmond's Dance Dataset - 524 songs with multiple emotion labels per song
# Dataset proposed - https://aclanthology.org/2021.wassa-1.24.pdf
# https://github.com/BaoChunhui/Generate-Emotional-Music

# "labels" and "emotions" are used interchangeably throughout the code but mean
# the same thing


# Reading in data files
multiLabel = pd.read_csv("data/MultiLabel.csv")
singleLabel = pd.read_csv("data/SingleLabel.csv")

multiLabel.head()
singleLabel.head()
len(multiLabel.index)   # 1,160 songs
len(singleLabel.index)   # 1,160 songs


#---------------------#
# MULTI LABEL DATASET #
#---------------------#

# Splitting labels column so that its not a list of randomly arranged emotions:
    # Will create a new column for each emotion class
    # Will code a 0 if the emotion is not present and 1 if it is 
    # calmness, tenderness ----> calmness: 1   tenderness: 1 

# 1. Instantiate count vectorizer
cv = CountVectorizer()

# 2. Fit CV to our labels column
freq = cv.fit_transform(multiLabel.labels)

# 3. Create a DF 
# https://stackoverflow.com/questions/45905722/python-access-labels-of-sklearn-countvectorizer
dtm = pd.DataFrame(freq.toarray(), columns=cv.get_feature_names())
dtm.head(10)

# left join our new labels DF to our original multiLabel DF
multiLabel = multiLabel.join(dtm)
multiLabel.head()

