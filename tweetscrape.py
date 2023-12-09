import requests
import pandas as pd
import os

def fetch_twitter_data(query, num):
    twitter_data = []
    payload = {
    'api_key' : '1eebc3a5e907d837040a9bb554970fa1',
    'query': query,
    'num': num
    }

    response = requests.get('https://api.scraperapi.com/structured/twitter/search',params=payload)
    data = response.json()

    all_data = data['organic_results']
    for tweet in all_data:
        twitter_data.append(tweet)

    return pd.DataFrame(twitter_data)

def save_csv(df, csv_file_path):
    df.to_csv(csv_file_path, index=False)

current_directory = os.getcwd()

csv_file_path0 = os.path.join(current_directory, 'cand1.csv')
csv_file_path1 = os.path.join(current_directory, 'cand2.csv')

def del_csv(csv_file_path):
    try:
        os.remove(csv_file_path)
        print(f"CSV file '{csv_file_path}' deleted successfully.")
    except OSError as e:
        print(f"Error deleting CSV file '{csv_file_path}': {e}")