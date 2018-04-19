import csv
import datetime
import histdata
import news
import sentiment
import sys
import os
import numpy as np
import csv
import string
import nltk
import enchant

csv.field_size_limit(sys.maxint)
FNAME = 'snp500.csv'

def fetchData():
    print 'Updating historical stock data'
    histdata.getHistData()

    print 'Updating news data'
    # news.init()

def getStockData(symbol, date):

    base_dir = os.getcwd()
    fname = '/data/hsd/' + symbol + '.csv'
    fpath = base_dir + fname

    if not os.path.isfile(fpath):
        print("File is not exist:%s" % fpath)
        return -1

    file = open('data/hsd/' + symbol + '.csv')
    csv_file = csv.reader(file)

    # Get stock data for the next day
    date += datetime.timedelta(days=1)

    data = []

    print 'Getting stock data for %s for date %s' % (symbol, date.strftime('%Y-%m-%d'))

    for row in csv_file:
        t_date = row[2]
        t_day = t_date.split(" ")
        t_d = t_day[0]
        if(t_d == date.strftime('%Y-%m-%d')):
            data.append(float(row[1]))
            data.append(float(row[4]))
            return row

    # No data found for symbol for given date
    return -1

def genData():
    #dataHistFile = open('dat.pkl', 'r+b')
    #dataHist = pickle.load(dataHistFile)
    # dataFileNumber = dataHist['data_file_number'] + 1

    # dataFile = open('data/dat_' + dataFileNumber + '.csv', 'a')
    dataFile = open('data/dat_1.csv', 'a')
    csvWriter = csv.writer(dataFile)
    # date = dateHist['last_updated']
    date = datetime.datetime(2018,01,02).date()
    endDate = datetime.date.today()

    while(date < endDate):
        print 'Checking data for ' + date.strftime('%Y-%m-%d')

        day = date.weekday()
        if(day == 4 or day == 5):
            date += datetime.timedelta(days=1)
            continue

        fname = date.strftime('%Y-%m-%d')
        file = open('data/news/' + fname + '.csv')
        csv_file = csv.reader(file)

        for row in csv_file:
            stockdata = getStockData(row[0], date)
            if(stockdata == -1):
                continue
            sentdata = sentiment.analyzeText(row[1])

            data = []
            data.extend((row[0], date.timetuple().tm_yday))
            data.extend((sentdata.score, sentdata.magnitude))
            data.extend((stockdata[5],stockdata[1],stockdata[0],stockdata[3],stockdata[4],stockdata[6]))
            csvWriter.writerow(data)

        date += datetime.timedelta(days=1)

    # dataHist['data_file_number'] = dataFileNumber
    #dataHist['last_updated'] = endDate
    #dataHistFile.seek(0)
    #pickle.dump(dataHist, dataHistFile, protocol = pickle.HIGHEST_PROTOCOL)
    #dataHistFile.close()

def init():
    # fetchData()
    genData()

# init()




# Test Scope
def filterByEnchant(s):
    d = enchant.Dict("en_US")
    english_words = []
    for word in s.split():
        if d.check(word):
            english_words.append(word)
    b = " ".join(english_words)
    return b

def filterString(s):
    printable = set(string.printable)
    filter(lambda x: x in printable, s)
    return s
def filterByNLTK(s):
    words = set(nltk.corpus.words.words())
    b = " ".join(w for w in nltk.wordpunct_tokenize(s) \
         if w.lower() in words or not w.isalpha())
    return b

def getData():
    with open(FNAME) as csv_file:
        reader = csv.reader(csv_file)
        strockInfo = list(reader)
        data = np.array(strockInfo)
        data = data[1:, :]
    return data

def fetchData_test():
    print 'Updating historical stock data'
    histdata.getHistdata_test()

    print 'Updating news data'
    news.test()

def getStockData_Test(symbol, date):

    base_dir = os.getcwd()
    fname = '/test_data/hsd/' + symbol + '.csv'
    fpath = base_dir + fname

    if not os.path.isfile(fpath):
        print("File is not exist:%s" % fpath)
        return -1

    file = open('test_data/hsd/' + symbol + '.csv')
    csv_file = csv.reader(file)

    # Get stock data for the next day
    #date += datetime.timedelta(days=1)

    data = []

    print 'Getting stock data for %s for date %s' % (symbol, date.strftime('%Y-%m-%d'))

    for row in csv_file:
        t_date = row[2]
        t_day = t_date.split(" ")
        t_d = t_day[0]
        # row 1: close, row 3: high, row 4: low, row 5: open
        if(t_d == date.strftime('%Y-%m-%d')):
            data.append(float(row[1]))
            data.append(float(row[5]))
            data.append(float(row[3]))
            data.append(float(row[4]))
            return row

    # No data found for symbol for given date
    return -1

# Data CSV file
def genData_Test():
    #dataHistFile = open('dat.pkl', 'r+b')
    #dataHist = pickle.load(dataHistFile)
    # dataFileNumber = dataHist['data_file_number'] + 1

    # dataFile = open('data/dat_' + dataFileNumber + '.csv', 'a')
    dataFile = open('test_data/dat_0.csv', 'a')
    csvWriter = csv.writer(dataFile)
    # date = dateHist['last_updated']
    date = datetime.datetime(2018,4,12).date()
    endDate = datetime.date.today()

    while(date <= endDate):
        print 'Checking data for ' + date.strftime('%Y-%m-%d')

        day = date.weekday()
        if(day == 4 or day == 5):
            date += datetime.timedelta(days=1)
            continue

        fname = date.strftime('%Y-%m-%d')
        print("File Name: " + fname)
        file = open('test_data/news/' + fname + '.csv')
        csv_file = csv.reader(file)

        for row in csv_file:
            stockdata = getStockData_Test(row[0], date)
            if(stockdata == -1):
                continue
            print("Origin String: %s" % row[1])
            filtered_string = filterString(row[1])
            filtered_string = filtered_string.decode("ascii", errors="ignore").encode()
            print("Filtered String: %s" % filtered_string)
            try:
                sentdata = sentiment.analyzeText(filtered_string)
            except:
                enchant_string = filterByEnchant(filtered_string)
                print("Enchanted String: %s" % enchant_string)
                try:
                    sentdata = sentiment.analyzeText(enchant_string)
                except:
                    sentdata = sentiment.analyzeText(row[0])

            data = []
            data.extend((row[0], date.timetuple().tm_yday))
            data.extend((sentdata.score, sentdata.magnitude))
            # row 0: adj close, row 1: close, row 3: high, row 4: low, row 5: open, row 6: volume
            data.extend((stockdata[5],stockdata[1],stockdata[0],stockdata[3],stockdata[4],stockdata[6]))
            # row 0: symbol, row 1: date, row 2: sentiment score, row 3: sentiment magnitude, row 4: open, row 5: close, row 6: adj close, row 7: high, row 8: low, row 9: volume
            csvWriter.writerow(data)

        date += datetime.timedelta(days=1)

def test():

    fetchData_test()
    genData_Test()

# Test 
#test()
