import requests
import datetime
import time
import json
from urllib2 import Request, urlopen
import pandas as pd
import numpy as np
import csv
import sys

csv.field_size_limit(sys.maxint)
FNAME = "snp500_formatted.txt"
# FNAME = 'snp500.csv'

def getData():
    with open(FNAME) as csv_file:
        reader = csv.reader(csv_file)
        strockInfo = list(reader)
        data = np.array(strockInfo)
        data = data[1:, :]
    return data

def getHistData():
    DAY = '01'
    MONTH = '00' # month must be m - 1
    YEAR = '2013'

    with open(FNAME) as f:
        stocks = f.readlines()

    for i in range(len(stocks)):
        stock = stocks[i].rstrip('\n')
	k_stock = stock.split('.')[0]
        date = datetime.datetime(2013, 1, 1)
        dt = time.mktime(date.timetuple()) + date.microsecond / 1E6
        dt = int(dt)
        query = 'https://query1.finance.yahoo.com/v7/finance/chart/' + k_stock + '?period1=' + str(dt) + '&period2=9999999999&interval=1d&indicators=quote&includeTimestamps=true'
        # query = 'http://ichart.finance.yahoo.com/table.csv?s=' + stock + '&a=' + MONTH + '&b=' + DAY + '&c=' + YEAR
        print 'Getting historical data for ' + stock
        print query
        # response = requests.get(query)

        try:
            request = Request(query)
            response = urlopen(request)

            elevations = response.read()
            data = json.loads(elevations)
            error = data['chart']['error']
            if error is not None:
                continue
            else:
                val_data = data['chart']['result']
                main_data = val_data[0]
                indicators = main_data['indicators']
                adjclose = indicators['adjclose']
                adjclose = adjclose[0]
                adjclose = adjclose['adjclose']
                quote = indicators['quote']
                quote = quote[0]
                close = quote['close']
                high = quote['high']
                low = quote['low']
                open_data = quote['open']
                volume = quote['volume']
                # unadjclose = indicators['unadjclose']
                # unadjclose = unadjclose[0]
                # meta = main_data['meta']
                timestamp = main_data['timestamp']

                # dt = pd.DataFrame()
                # dt.columns = ["Date","Open","High","Low","Close","Volume","Adj Close"]
                dt = pd.DataFrame(
                    {'Date': timestamp,
                     'Open': open_data,
                     'High': high,
                     'Low': low,
                     'Close': close,
                     'Volume': volume,
                     'Adj Close': adjclose
                     })
                dt['Date'] = pd.to_datetime(dt['Date'], unit='s')
                file_name = 'data/hsd/' + stock + '.csv'
                dt.to_csv(file_name, index=None)

                print("Parsing")
        except :
            print("Error on %s" % stock)
            continue

def getHistdata_test():
    with open(FNAME) as f:
        stocks = f.readlines()

    for i in range(len(stocks)):
        stock = stocks[i].rstrip('\r\n')
        k_stock = stock.split('.')[0]
        date = datetime.datetime(2018, 1, 1)
        dt = time.mktime(date.timetuple()) + date.microsecond / 1E6
        dt = int(dt)
        query = 'https://query1.finance.yahoo.com/v7/finance/chart/' + k_stock + '?period1=' + str(dt) + '&period2=9999999999&interval=1d&indicators=quote&includeTimestamps=true'
        # query = 'http://ichart.finance.yahoo.com/table.csv?s=' + stock + '&a=' + MONTH + '&b=' + DAY + '&c=' + YEAR
        print 'Getting historical data for ' + stock
        print query

        try:
            request = Request(query)
            response = urlopen(request)

            elevations = response.read()
            data = json.loads(elevations)
            error = data['chart']['error']
            if error is not None:
                continue
            else:
                val_data = data['chart']['result']
                main_data = val_data[0]
                indicators = main_data['indicators']
                adjclose = indicators['adjclose']
                adjclose = adjclose[0]
                adjclose = adjclose['adjclose']
                quote = indicators['quote']
                quote = quote[0]
                close = quote['close']
                high = quote['high']
                low = quote['low']
                open_data = quote['open']
                volume = quote['volume']
                # unadjclose = indicators['unadjclose']
                # unadjclose = unadjclose[0]
                # meta = main_data['meta']
                timestamp = main_data['timestamp']

                # dt = pd.DataFrame()
                # dt.columns = ["Date","Open","High","Low","Close","Volume","Adj Close"]
                dt = pd.DataFrame(
                    {'Date': timestamp,
                     'Open': open_data,
                     'High': high,
                     'Low': low,
                     'Close': close,
                     'Volume': volume,
                     'Adj Close': adjclose
                     })
                dt['Date'] = pd.to_datetime(dt['Date'], unit='s')
                file_name = 'test_data/hsd/' + stock + '.csv'
                dt.to_csv(file_name, index=None)

                print("Parsing")
        except :
            print("Error on %s" % stock)
            continue
#getHistdata_test()

