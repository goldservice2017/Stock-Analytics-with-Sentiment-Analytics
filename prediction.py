import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import datetime
import csv
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
import gendata
import sentiment
import pandas as pd
import sys
import nn
import lin_reg

predict_encoder = preprocessing.LabelEncoder()

FNAME = "snp500_formatted.txt"

Model_Path = os.getcwd() + '/model'

saver = None

arguments = len(sys.argv)
if arguments == 1:
    print("Input Date as Year-Month-Day")
    #inputdate = datetime.date.today()
    predict_date = "2018-03-28"
    inputdate = datetime.datetime.strptime(predict_date, '%Y-%m-%d')
else:
    predict_date = sys.argv[1]
    inputdate = datetime.datetime.strptime(predict_date, '%Y-%m-%d')


def predictForDate(date):
    print("Date: %s" % date.strftime('%Y-%m-%d'))
    # gendata.test()
    nn.buildModel()
    #lin_reg.main()
    nn.prediction(date)
    #lin_reg.prediction(date)

if __name__ == "__main__":
    predictForDate(inputdate)

    

