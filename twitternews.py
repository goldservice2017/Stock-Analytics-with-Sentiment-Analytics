import sys
import re
import requests
from bs4 import BeautifulSoup
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got


def processTweet(tweet):
    #tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

def tweetNewsSearch(key, since, until, limit):
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(key).setSince(since).setUntil(until).setMaxTweets(limit).setTopTweets(True)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    results = []
    txt = ''
    print(key + "Found %s tweets." % str(len(tweets)))
    for tweet in tweets:
        text = tweet.text.encode('utf-8')
        text = processTweet(text)
        # text = text.encode('ascii', 'ignore')
        # print("Tweet News: %s" % text)
        results.append(text)
        if txt == '':
      	    txt = txt + text
	else:
            txt = txt + ". " + text
    return txt

#news = tweetNewsSearch("3M Company","2018-01-01","2018-01-03",10)
# url = "https://twitter.com/search?l=en&q=EQT%20Corporation%20since%3A2018-02-01%20until%3A2018-02-03&src=typd"
# response = requests.get(url)
# soup = BeautifulSoup(response.text, "html.parser")
# subs = soup.findAll('div', {'class': 'js-tweet-text-container'})
# divs = []
# if len(subs) > 0:
#     divs.extend(subs)
#
# data = ''
# for div in divs:
#
#     content = div.findAll(text=True)
#     if content is not None:
#         # data = data.join(content[0])
#         data = data + content[0]
#
# print("Done")