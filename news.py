from bs4 import BeautifulSoup
import datetime
import pickle
import requests
import csv
import sys
import numpy as np
import twitternews
import re

# FNAME = "snp500_formatted.txt"
FNAME = "snp500.csv"
stocks = []

def getNewsForDate(date):
    file = open('data/news/' + date.strftime('%Y-%m-%d') + '.csv', 'w')
    print('Getting news for ' + date.strftime('%Y-%m-%d'))
    for i in range(len(stocks)):
        #query = 'http://www.reuters.com/finance/stocks/companyNews?symbol=' + stocks[i] + '&date=' + format(date.month, '02d') + format(date.day, '02d') + str(date.year)
        #print('Getting news for ' + stocks[i])
	query = 'https://www.reuters.com/finance/stocks/company-news/' + stocks[i] + '?date=' + format(date.month, '02d') + format(date.day, '02d') + str(date.year)
        print('Getting news for ' + stocks[i])
	
        response = requests.get(query)
        soup = BeautifulSoup(response.text, "html.parser")

        divs = soup.findAll('div', {'class': 'feature'})
        print('Found ' + str(len(divs)) + ' articles.')

        if(len(divs) == 0 or len(divs) == 9):
            continue

        data = ''
        for div in divs:

            content = div.findAll(text=True)
            if content is not None:
                # data = data.join(content[0])
                data = data + content[0]    
	        # data = data.join(div.findAll(text=True))
        file.write(stocks[i] + ',' + data.encode('utf-8').replace('\n', ' '))
        file.write('\n')
    file.close()

def getNews():
    dataHistFile = open('dat.pkl', 'rb+')
    dataHist = pickle.load(dataHistFile)
   # date = dataHist['last_updated'] + datetime.timedelta(days=1)
    dataHist['last_updated'] = datetime.datetime(2018, 02, 02).date()
    dataHistFile.seek(0)
    pickle.dump(dataHist, dataHistFile, protocol=pickle.HIGHEST_PROTOCOL)
    
    date = dataHist['last_updated'] + datetime.timedelta(days=1)
    endDate = datetime.date.today()

    while(date <= endDate):
        getNewsForDate(date)
        date += datetime.timedelta(days=1)

    dataHist['last_updated'] = endDate
    dataHistFile.seek(0)
    pickle.dump(dataHist, dataHistFile, protocol = pickle.HIGHEST_PROTOCOL)
    dataHistFile.close()

def init():
    global stocks
    with open(FNAME) as f:
        stocks = f.readlines()
    for i in range(len(stocks)):
        stocks[i] = stocks[i].rstrip('\n')

    getNews()

# Test Scope
def getStocksymbol():
    with open(FNAME) as csv_file:
        reader = csv.reader(csv_file)
        stockinfo = list(reader)
    data = np.array(stockinfo)
    data = data[1:,:]
    return data

def getEconomicsNews():
    query = 'https://www.reuters.com/finance/markets/us'
    print('Getting economics news')
    response = requests.get(query)
    soup = BeautifulSoup(response.text, "html.parser")
    divs = soup.findAll('div', {'class': 'feature'})
    if len(divs) > 0:
        divs = divs[:1]
    print('Found ' + str(len(divs)) + ' articles.')
    

    data = ''
    for div in divs:

        content = div.findAll(text=True)
        if content is not None:
        # data = data.join(content[0])
            data = data + content[0]

    return data
def getGeneralNews(name):
    query = 'https://news.google.com/news/search/section/q/' + name + '?hl=en&gl=US&ned=us'
    print('General news from:%s' % query)
    response = requests.get(query)
    soup = BeautifulSoup(response.text, "html.parser")
    divs = soup.findAll('a', {'class': 'neEeue hzdq5d ME7ew'})
    print('Found' + str(len(divs)) + ' articles.')

    if len(divs) > 0:
        divs = divs[:1]

    data = ''
    for div in divs:
        content = div.findAll(text=True)
        if content is not None:
            data = data + content[0]

    return data

def getNewsForDate_test(date):
    stocks = getStocksymbol()
    file = open('test_data/news/' + date.strftime('%Y-%m-%d') + '.csv', 'w')
    
    print('Getting news for ' + date.strftime('%Y-%m-%d'))
    # economics_data = getEconomicsNews()
    for i in range(len(stocks)):
        stock = stocks[i]
        symbol = stock[0]
        company_name = stock[1]
        # stock_query = 'https://www.reuters.com/finance/stocks/company-news/' + symbol + '?date=' + format(date.month, '02d') + format(date.day, '02d') + str(date.year)
        # print('Getting stock news for ' + symbol)
        #response = requests.get(stock_query)
        # soup = BeautifulSoup(response.text, "html.parser")
        # divs = soup.findAll('div', {'class': 'feature'})
        # divs = getReuterNews(symbol, date)
        # print('Stock: '+ symbol + ' Found ' + str(len(divs)) + ' articles.')

        #data_stock = ''
        data_google = ''
        #for div in divs:
        #    content = div.findAll(text=True)
        #    if content is not None:
        #        data_stock = data_stock + content[0]
        
        st = date - datetime.timedelta(days=1)
        tweets = twitternews.tweetNewsSearch(company_name,st.strftime('%Y-%m-%d'),date.strftime('%Y-%m-%d'), 10)


        # google_query = 'https://newsapi.org/v2/everything?q=' + company_name + '&from=' + st.strftime('%Y-%m-%d') + '&sortBy=publishAt&language=en&pageSize=5&&to='+ date.strftime('%Y-%m-%d') +'&apiKey=823c1e93d0714adbaefdc9e9fc89be4a'   
        # google_query = 'https://newsapi.org/v2/everything?q=' + company_name + '&from=' + st.strftime('%Y-%m-%d') + '&sortBy=publishedAt&language=en&pageSize=5&to=' + date.strftime('%Y-%m-%d') + '&apiKey=823c1e93d0714adbaefdc9e9fc89be4a'
        # print google_query
        # google_response = requests.get(google_query)
        #google_data = google_response.json()

        # print google_data
        #if google_data['status'] == 'ok':
        #    google_articles = google_data['articles']
        #    if len(google_articles) > 4:
        #        google_articles = google_articles[:4]
        #    for article in google_articles:
        #        article_title = article['title']
        #        if article_title is not None:
        #            data_google = data_google + article_title + " ."
        
        
        # general_data = getGeneralNews(company_name)

        #if (data_stock == '' and data_google == ''):
        #    all_data = general_data + ' ' + economics_data
        #else:
        #    all_data = data_stock + " " + data_google
        
        # e_data = ''
        # e_data = company_name + ',  ' + economics_data
        # if (data_stock == ''):
            # all_data = data_google + economics_data
        #    all_data = e_data
        # else:
        #    all_data = data_stock
        if (tweets== ''):
            all_data = company_name + ' do not have any news.'
        else:
            all_data = tweets
        file.write(symbol.encode('utf-8') + ',' + all_data.replace('\n', ' '))
        file.write('\n')
    file.close()
def getReuterNews(symbol,date):
    stDate = date - datetime.timedelta(days=3)
    divs = []
    while(date > stDate):
        if len(divs) > 0:
            return divs
        stock_query = 'https://www.reuters.com/finance/stocks/company-news/' + symbol + '?date=' + format(date.month, '02d') + format(date.day, '02d') + str(date.year)
        response = requests.get(stock_query)
        soup = BeautifulSoup(response.text, "html.parser")
        subs = soup.findAll('div', {'class': 'feature'})
        if len(subs) > 0:
            divs.extend(subs)
        date -= datetime.timedelta(days=1)

    return divs

def getNews_test():

    date = datetime.datetime(2018,4,12).date()
    #endDate = datetime.datetime(2018,1,26).date()
    endDate = datetime.date.today()

    while(date <= endDate):
        getNewsForDate_test(date)
        date += datetime.timedelta(days=1)

def test():
    # global stocks
    # with open(FNAME) as f:
    #    stocks = f.readlines()
    # for i in range(len(stocks)):
    #    stocks[i] = stocks[i].rstrip('\n')

    # stocks = getStocksymbol()

    getNews_test()

#test()
