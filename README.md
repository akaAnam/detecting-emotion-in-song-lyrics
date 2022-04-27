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
On average, the content feature set had slightly stronger predictive performance across most emotions. Both feature sets
performed about the same on: 
- amazement
- calmness

While content features performed better in predicting the following emotions: 
- joyful
- nostalgia
- power
- tenderness
- tension

And style features better in predicting the following emotions:
- sadness
- solemnity

![](https://github.com/akaAnam/detecting-emotion-in-song-lyrics/blob/main/visuals/results_heatmap.PNG?raw=true)

## Conclusion
Going into this project, we hypothesized that content-based feature sets would be more important in accurately classifying
multiple emotions given a lyrical dataset. Our findings reveal that is not the case, and that style-based feature sets tend to more
accurately classify melancholic emotions like 'sadness' or 'solemnity'. 

We also found that developing a style-based feature set from song lyric data is difficult to accomplish, 
given the arbitrary nature of transcribing punctuation from artists. Transcriptions can vary from artist to artist
with little consistency, which underscores the potential importance of audio data as opposed to text data in
developing a style-based feature set for multi-emotion classification. 
