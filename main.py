import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import gradio as gr
from tweetscrape import fetch_twitter_data, save_csv, del_csv

names = []
pos = []
neg = []

csv_file_path0 = 'cand1.csv'
csv_file_path1 = 'cand2.csv'

def main_fn(query0, query1, num):

    df0 = fetch_twitter_data(query0, num)
    df1 = fetch_twitter_data(query1, num)

    if len(df0) != len(df1):
        return "Not enough tweets", None

    save_csv(df0, csv_file_path0)
    save_csv(df1, csv_file_path1)

    candidate1 = pd.read_csv("cand1.csv")
    candidate2 = pd.read_csv("cand2.csv")

    def polarity(text):
        return TextBlob(text).sentiment.polarity

    candidate1['Polarity'] = candidate1["snippet"].apply(polarity)
    candidate2['Polarity'] = candidate2["snippet"].apply(polarity)

    candidate1['Sentiment'] = np.where(candidate1['Polarity']>0,"Positive", "Negative")
    candidate1['Sentiment'][candidate1['Polarity']==0]="Neutral"

    candidate2['Sentiment'] = np.where(candidate2['Polarity']>0,"Positive", "Negative")
    candidate2['Sentiment'][candidate2['Polarity']==0]="Neutral"

    candidate1_neutral = candidate1[candidate1['Polarity']==0]
    candidate2_neutral = candidate2[candidate2['Polarity']==0]

    neutral_na_tweets = (candidate1_neutral.shape[0] + candidate2_neutral.shape[0])

    candidate1.drop(candidate1[candidate1['Polarity']==0].index, inplace=True)
    candidate2.drop(candidate2[candidate2['Polarity']==0].index, inplace=True)

    count_candidate1 = candidate1.groupby('Sentiment').count()
    count_candidate2 = candidate2.groupby('Sentiment').count()

    names = [query0,query1]
    pos_candidate1 = count_candidate1['Polarity'][1] if len(count_candidate1['Polarity']) > 1 else 0
    pos_candidate2 = count_candidate2['Polarity'][1] if len(count_candidate2['Polarity']) > 1 else 0

    pos = [pos_candidate1, pos_candidate2]

    neg_candidate1 = count_candidate1['Polarity'][0] if len(count_candidate1['Polarity']) > 0 else 0
    neg_candidate2 = count_candidate2['Polarity'][0] if len(count_candidate2['Polarity']) > 0 else 0

    neg = [neg_candidate1, neg_candidate2]
    
    data1 = pd.DataFrame({'Candidates': names, 'Positive Twitter Sentiment':pos})
    data2 = pd.DataFrame({'Candidates': names, 'Negative Twitter Sentiment':neg})
    return data1, data2



iface = gr.Interface(title="TweetPoll",fn=main_fn, inputs=['text','text',gr.Slider(0,100,step=1)], 
                     outputs=[
                         gr.BarPlot(x='Candidates',y='Positive Twitter Sentiment',title="Positive Sentiments",x_title='Polarity of Sentiments', y_title='Number of Tweets', height=500, width=500, interactive=True), 
                         gr.BarPlot(x='Candidates',y='Negative Twitter Sentiment',title="Negative Sentiments",x_title='Polarity of Sentiments', y_title='Number of Tweets', height=500, width=500,interactive=True),
                    ]
                )
iface.launch(share=True)

del_csv(csv_file_path0)
del_csv(csv_file_path1)

