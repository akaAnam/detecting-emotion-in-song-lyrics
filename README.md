# Detecting Emotions in Song Lyrics

## Introduction
Song lyrics can contain a multitude of emotions (sadness, nostalgia, etc.) and can powerfully portray the emotional state of an
artist. This project compares two feature sets to assess whether content or style is more suitable as a feature set in multi-emotion
classification for song lyrics.

## Methodology
Current literature around the multi-emotion classification problem for song lyrics compares classification algorithms,
different emotion labels, as well as textual vs. audio analysis. (https://aclanthology.org/C18-1029.pdf) 
This project uses the concept of comparing "Topic or Style" (https://aclanthology.org/C18-1029.pdf) feature sets for 
application in the multi-emotion classification problem. Following prior research, Logistic Regression and Naive Bayes classifiers
are used as models for comparing the predictive performance of the two feature sets. 

`data/MultiLabel.csv` contains the original song lyric dataset used for this project. 

`data/cleaned_multiLabel.csv` contains the cleaned dataset used for analysis. `data_cleaning.py` splits the bunched labels
and utilizes`CountVectorizer()` for one-hot encoding columns to indicate the presence of the following emotions: 
- amazement
- calmness
- joyful
- nostalgia
- power
- sadness
- solemnity
- tenderness
- tension

`content_features.py` creates a feature set based off of lyrical content and trains a `LogisticRegression()`
model as well as `MultinomialNB` model. 

`style_features.py` creates a feature set based off of stopwords and trains a `LogisticRegression()`
model as well as `MultinomialNB` model. 
## Results
On average, the feature set based off of lyrical content had stronger predictive performance across most emotions and both models. 
![](https://github.com/akaAnam/detecting-emotion-in-song-lyrics/blob/main/visuals/results_heatmap.PNG?raw=true)


## Conclusion
Applications, limitations, further direction. 