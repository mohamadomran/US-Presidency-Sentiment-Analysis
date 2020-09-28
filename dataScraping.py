# Import the Libraries
from pathlib import Path
from dotenv import load_dotenv

import sys
import csv
import tweepy
import ssl
import os
import json

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Fetching Oauth Keys .env File
consumer_key = os.getenv("API_KEY")
consumer_secret = os.getenv("API_SECRET_KEY")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

# Twitter Authentication
authentication = tweepy.OAuthHandler(consumer_key, consumer_secret)
authentication.set_access_token(access_token, access_token_secret)
ssl._create_default_https_context = ssl._create_unverified_context
api = tweepy.API(authentication)

# enable/disable username accordingly to fetch the replies

# name = 'realDonaldTrump'
# name = 'JoeBiden'

# tweet_id = ['1290967953542909952']

# Getting Replies + filling them in their Respective csv Files
replies=[]
for tweet in tweepy.Cursor(api.search,q='to:'+name, result_type='recent', timeout=999999).items(100):
    if hasattr(tweet, 'in_reply_to_status_id_str'):
        if (tweet.in_reply_to_status_id_str==tweet_id):
            replies.append(tweet)

# with open('Biden_dataset.csv', 'a+') as f:
with open('Trump_dataset.csv', 'a+') as f:
    csv_writer = csv.DictWriter(f, fieldnames=('user', 'text'))
    csv_writer.writeheader()
    for tweet in replies:
        row = {'user': tweet.user.screen_name, 'text': tweet.text.replace('\n', ' ')}
        csv_writer.writerow(row)