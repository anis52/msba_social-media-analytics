#########
'''
# Step 3_Sentiment Analysis
**Note**: This part of codes include 3 parts:
* Part I: Sentiment Analysis for fbcomment
* Part II: Sentiment Analysis for fbpost
* Part II: Sentiment Analysis for fbreview

**Data Required**:
comments.csv, fbcomments.csv, posts.csv, fbposts.csv, reviews.csv, fandango_review.csv
'''
#########

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

sid = SentimentIntensityAnalyzer()


# Part I: Sentiment Analysis for fbcomment

fbcomment = pd.read_csv('comments.csv')
fbcomment['sentiment_score'] = None
fbcomment['sentiment'] = None

for i in range(len(fbcomment)):
    if fbcomment.comment_text[i] is not np.nan:
        fbcomment_text = fbcomment.comment_text[i]
        scores = sid.polarity_scores(fbcomment_text)
        fbcomment['sentiment_score'][i] = scores['compound']
        if scores['compound'] >= 0:
            fbcomment['sentiment'][i] = 'Positive'
        else:
            fbcomment['sentiment'][i] = 'Negative'
    else:
        pass

fbcomment_all = pd.read_csv('fbcomments.csv')
fbcomment_all['sentiment'] = fbcomment['sentiment']
fbcomment_all['sentiment_score'] = fbcomment['sentiment_score']
fbcomment_all.to_csv('fbcomment_sentiment.csv', index=False)
fbcomment_all

# Part II: Sentiment Analysis for fbpost

fbpost = pd.read_csv('posts.csv')
fbpost['sentiment_score'] = None
fbpost['sentiment'] = None

for i in range(len(fbpost)):
    if fbpost.posttext[i] is not np.nan:
        fbpost_text = fbpost.posttext[i]
        scores = sid.polarity_scores(fbpost_text)
        fbpost['sentiment_score'][i] = scores['compound']
        if scores['compound'] >= 0:
            fbpost['sentiment'][i] = 'Positive'
        else:
            fbpost['sentiment'][i] = 'Negative'
    else:
        pass

fbpost_all = pd.read_csv('fbposts.csv')
fbpost_all['sentiment'] = fbpost['sentiment']
fbpost_all['sentiment_score'] = fbpost['sentiment_score']
fbpost_all.to_csv('fbpost_sentiment.csv', index=False)
fbpost_all

# Part III: Sentiment Analysis for fbreview

fbreview_c = pd.read_csv('reviews.csv')
fbreview_c['sentiment_score'] = None
fbreview_c['sentiment'] = None

for i in range(len(fbreview_c)):
    if fbreview_c.review_content[i] is not np.nan:
        fbreview_text = fbreview_c.review_content[i]
        scores = sid.polarity_scores(fbreview_text)
        fbreview_c['sentiment_score'][i] = scores['compound']
        if scores['compound'] >= 0:
            fbreview_c['sentiment'][i] = 'Positive'
        else:
            fbreview_c['sentiment'][i] = 'Negative'
    else:
        pass

fbreviews_all = pd.read_csv('fandango_review.csv')
fbreviews_all['sentiment'] = fbreview_c['sentiment']
fbreviews_all['sentiment_score'] = fbreview_c['sentiment_score']
fbreviews_all.to_csv('fbreviews_sentiment.csv', index=False)
fbreviews_all

# Here, we also tried to use the review header to do the sentiment analysis, but the results are less accurate than the analysis using review content.

fbreview_h = pd.read_csv('reviews.csv')
fbreview_h['sentiment_score'] = None
fbreview_h['sentiment'] = None

for i in range(len(fbreview_h)):
    if fbreview_h.review_header[i] is not np.nan:
        fbreview_text = fbreview_h.review_header[i]
        scores = sid.polarity_scores(fbreview_text)
        fbreview_h['sentiment_score'][i] = scores['compound']
        if scores['compound'] >= 0:
            fbreview_h['sentiment'][i] = 'Positive'
        else:
            fbreview_h['sentiment'][i] = 'Negative'
    else:
        pass

fbreviews_all_2 = pd.read_csv('fandango_review.csv')
fbreviews_all_2['sentiment'] = fbreview_h['sentiment']
fbreviews_all_2['sentiment_score'] = fbreview_h['sentiment_score']
fbreviews_all_2