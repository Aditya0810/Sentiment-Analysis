import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import gradio as gr
from tweetscrape import fetch_twitter_data, save_csv, del_csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

positive_words = [
    "Honest", "Trustworthy", "Inspirational", "Dedicated", "Visionary",
    "Capable", "Knowledgeable", "Charismatic", "Compassionate", "Empathetic",
    "Forward-thinking", "Transparent", "Accountable", "Collaborative", "Dynamic",
    "Energetic", "Courageous", "Tenacious", "Resilient", "Positive",
    "Optimistic", "Strategic", "Diplomatic", "Articulate", "Persuasive",
    "Innovative", "Inclusive", "Caring", "Approachable", "Proactive",
    "Experienced", "Thoughtful", "Reputable", "Reliable", "Focused",
    "Open-minded", "Fair", "Adaptable", "Progressive", "Genuine",
    "Responsible", "Hardworking", "Principled", "Insightful", "Resourceful",
    "Tenacious", "Committed", "Efficient", "Results-driven", "Dynamic",
    "Inspirational", "Influential", "Pioneering", "Unifying", "Empowering",
    "Decisive", "Disciplined", "Visionary", "Ambitious", "Respected",
    "Eloquent", "Tenacious", "Creative", "Unbiased", "Persistent",
    "Informed", "Diplomatic", "Charismatic", "Sincere", "Patient",
    "Supportive", "Altruistic", "Dependable", "Authentic", "Ethical",
    "Pioneering", "Results-oriented", "Articulate", "Adaptable", "Strategic",
    "Loyal", "Modest", "Gracious", "Enterprising", "Sympathetic",
    "Inclusive", "Tolerant", "Cooperative", "Dynamic", "Confident",
    "Trusting", "Generous", "Progressive", "Motivated", "Astute",
    "Insightful", "Adept", "Exemplary", "Ingenious", "Inspiring"
]
negative_words = [
    "Corrupt", "Incompetent", "Deceptive", "Untrustworthy", "Scandalous",
    "Manipulative", "Dishonest", "Unreliable", "Scheming", "Controversial",
    "Divisive", "Unethical", "Ineffective", "Inconsistent", "Self-serving",
    "Cunning", "Unprincipled", "Unscrupulous", "Insensitive", "Irresponsible",
    "Unresponsive", "Stubborn", "Close-minded", "Shortsighted", "Reckless",
    "Arrogant", "Egotistical", "Narcissistic", "Indecisive", "Inflexible",
    "Stubborn", "Overbearing", "Unsympathetic", "Uncompassionate", "Stubborn",
    "Inconsiderate", "Obnoxious", "Hostile", "Aggressive", "Disrespectful",
    "Biased", "Bigoted", "Prejudiced", "Intolerant", "Narrow-minded",
    "Divisive", "Polarizing", "Hypocritical", "Disingenuous", "Untruthful",
    "Sarcastic", "Disillusioned", "Pessimistic", "Cynical", "Defensive",
    "Displeased", "Angry", "Resentful", "Disappointed", "Frustrated",
    "Annoyed", "Offended", "Disgusted", "Dismayed", "Hostile", "Bitter",
    "Rebellious", "Defiant", "Furious", "Outraged", "Insulted",
    "Disheartened", "Despondent", "Hopeless", "Pessimistic", "Depressed",
    "Dismal", "Melancholy", "Gloomy", "Regretful", "Remorseful",
    "Guilty", "Shameful", "Embarrassed", "Horrified", "Appalled",
    "Repulsed", "Displeased", "Apprehensive", "Nervous", "Anxious",
    "Worried", "Distressed", "Terrified", "Panicked", "Overwhelmed"
]


positive_words = [word.lower() for word in positive_words]
negative_words = [word.lower() for word in negative_words]

names = []
pos = []
neg = []

csv_file_path0 = 'cand1.csv'
csv_file_path1 = 'cand2.csv'

def calculate_sentiment(text):
    tokens = word_tokenize(text.lower()) #word tokenisation from 'nltk' library

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    positive_count = sum(1 for word in tokens if word in positive_words)
    negative_count = sum(1 for word in tokens if word in negative_words)
    
    if positive_count > negative_count:
        return "Positive"
    elif negative_count > positive_count:
        return "Negative"
    else:
        return "Neutral"
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

