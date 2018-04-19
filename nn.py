import datetime
import csv
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gendata
import sentiment
import pandas as pd
import os

predict_encoder = preprocessing.LabelEncoder()

FNAME = "snp500_formatted.txt"

Model_Path = os.getcwd() + '/model'

saver = None

def getData():
    with open('data/dat_1.csv') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)

    return data

def preprocessData(data):
    label_encoder = preprocessing.LabelEncoder()
    one_hot_encoder = preprocessing.OneHotEncoder()

    data[:,0] = label_encoder.fit_transform(data[:,0])
    data = data.astype(float)

    # Uncomment lines below to use stock symbol and day parameters
    # WARNING: Epochs may be extremely slow
    # processed_data = one_hot_encoder.fit_transform(data[:,0:2]).toarray()
    # processed_data = np.append(processed_data, data[:,2:6], 1)

    # Do not use stock symbol and day parameters for training
    processed_data = data[:,2:6]

    processed_data = preprocessing.normalize(processed_data)
    np.random.shuffle(processed_data)

    return processed_data

def learn(data):
    data = preprocessData(data)
    num_params = data.shape[1] - 1

    X = data[:,0:num_params]
    Y = data[:,num_params].reshape(-1, 1)

    # Split the data into training and testing sets (70/30)
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X, Y, test_size=0.30)
    train_opening_price = train_X[:, num_params - 1].reshape(-1, 1)
    test_opening_price = test_X[:, num_params - 1].reshape(-1, 1)

    # Get the initial stock prices for computing the relative cost
    stock_data = tf.placeholder(tf.float32, [None, num_params])
    opening_price = tf.placeholder(tf.float32, [None, 1])
    stock_price = tf.placeholder(tf.float32, [None, 1])

    # Number of neurons in the hidden layer
    n_hidden_1 = 3
    n_hidden_2 = 3

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([1]))
    }

    # Implement dropout to reduce overfitting
    keep_prob_input = tf.placeholder(tf.float32)
    keep_prob_hidden = tf.placeholder(tf.float32)

    # Hidden layers
    input_dropout = tf.nn.dropout(stock_data, keep_prob_input)
    layer_1 = slim.fully_connected(input_dropout, n_hidden_1, biases_initializer = None, activation_fn = tf.nn.relu)
    layer_1_dropout = tf.nn.dropout(layer_1, keep_prob_hidden)
    layer_2 = slim.fully_connected(input_dropout, n_hidden_1, biases_initializer = None, activation_fn = tf.nn.relu)
    layer_2_dropout = tf.nn.dropout(layer_2, keep_prob_hidden)
    output_layer = tf.add(tf.matmul(layer_2_dropout, weights['out']), biases['out'])

    learning_rate = 1e-4
    cost_function = tf.reduce_mean(tf.pow(tf.div(tf.subtract(stock_price, output_layer), opening_price), 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    last_cost = 0
    tolerance = 1e-6
    epochs = 1
    max_epochs = 1e6

    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        while True:
            sess.run(optimizer, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y, keep_prob_input: 0.8, keep_prob_hidden: 0.5})

            if epochs % 100 == 0:
                cost = sess.run(cost_function, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y, keep_prob_input: 0.8, keep_prob_hidden: 0.5})
                print "Epoch: %d: Error: %f" %(epochs, cost)

                if abs(cost - last_cost) <= tolerance or epochs > max_epochs:
                    print "Converged."
                    break
                last_cost = cost

            epochs += 1

        print "Test error: ", sess.run(cost_function, feed_dict={stock_data: test_X, opening_price: test_opening_price, stock_price: test_Y, keep_prob_input: 1.0, keep_prob_hidden: 1.0})
        test_results = sess.run(output_layer, feed_dict={stock_data: test_X, stock_price: test_Y, keep_prob_input: 1.0, keep_prob_hidden: 1.0})

    avg_perc_error = 0
    max_perc_error = 0
    mei = 0
    for i in range(len(test_Y)):
        actual_change = abs(test_Y[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        predicted_change = abs(test_results[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        delta = abs(actual_change - predicted_change)
        avg_perc_error = avg_perc_error + delta
        if delta > max_perc_error:
            max_perc_error = delta
            mei = i

    avg_perc_error = (avg_perc_error * 100) / len(test_Y)
    max_perc_error *= 100
    print "Maximum percentage error: %f\nAverage percentage error: %f\n" % (max_perc_error, avg_perc_error)

    test_dates = []    
    #p_f = open(FNAME, 'r')
       # stocks = p_f.readLines()
    t_date = datetime.datetime(2018, 03, 03)
    for i in range(4):
        test_dates.append(t_date)
        t_date += datetime.timedelta(days=1)
    t_date = datetime.datetime(2018, 03, 12)
    for i in range(3):
        test_dates.append(t_date)
        t_date += datetime.timedelta(days=1)

    print('Prediction Test Dates:' + ', '.join(d.strftime('%Y-%m-%d') for d in test_dates))
    precit_data = []
    for dt in test_dates:
        t_news_file = open('data/news/' + dt.strftime('%Y-%m-%d') + '.csv')
        print('Collecting Test Data for %s' % (dt + datetime.timedelta(days = 1)).strftime('%Y-%m-%d'))
        p_single_data = []
        p_csv_file = csv.reader(t_news_file)

        for row in p_csv_file:
            p_stockdata = gendata.getStockData(row[0], dt)
            if (p_stockdata == -1):
                continue
            sentdata = sentiment.analyzeText(row[1])

            data = []
            data.extend((row[0], dt.timetuple().tm_yday))
            data.extend((sentdata.score, sentdata.magnitude))
            data.extend((p_stockdata[5],p_stockdata[1],p_stockdata[0],p_stockdata[3],p_stockdata[4],p_stockdata[6]))

            p_single_data.append(data)

        if len(p_single_data) > 0:
            p_data = np.array(p_single_data)
            print('Test Data collected for %s data:' % (dt + datetime.timedelta(days = 1)).strftime('%Y-%m-%d') + ', '.join(str(c) for c in p_data))
            p_origin = p_data
            p_data = normalizeInput(p_data)
            params = p_data.shape[1] - 1
            p_x = p_data[:,0:params]
            p_y = p_data[:,params].reshape(-1, 1)
            p_test = sess.run(output_layer, feed_dict={stock_data: p_x, stock_price: p_y, keep_prob_input: 1.0,
                                                                 keep_prob_hidden: 1.0})

            orign_test = p_origin[:, 2:6]
            for i in range(orign_test.shape[0]):
                norm = np.linalg.norm(orign_test[i, :],axis=0)
                p_test[i,0] = p_test[i,0] * norm

            result_date = dt + datetime.timedelta(days=1)
            print('Predict Data for %s :' % result_date.strftime('%Y-%m-%d')+ ', '.join('Stock %s :%s'%(d[0], str(c)) for d, c in zip(p_single_data, p_test)))

            s_name = []
            s_close = []
            s_adjclose = []
            s_low = []
            s_high = []
            s_volume = []
            s_open = []
            for i in range(len(p_single_data)):
                s_name.append(p_single_data[i][0])
                s_close.append(p_single_data[i][5])
                s_adjclose.append(p_single_data[i][6])
                s_low.append(p_single_data[i][8])
                s_high.append(p_single_data[i][7])
                s_volume.append(p_single_data[i][9])
                s_open.append(p_single_data[i][4])
            p_result = pd.DataFrame({
                'symbol': s_name,
                'prediction': p_test[:, 0],
                'open': s_open,
                'close': s_close,
                'prev close': s_adjclose,
                'high': s_high,
                'low': s_low,
                'volume': s_volume})

            csv_name = 'data/result/result-%s.csv' % result_date.date()
            #file_result = open('data/result/' + csv_name, 'rb+')
            p_result.to_csv(csv_name)
         

def normalizeInput(data, type):
    data[:, 0] = predict_encoder.fit_transform(data[:, 0])
    data = data.astype(float)

    # Uncomment lines below to use stock symbol and day parameters
    # WARNING: Epochs may be extremely slow
    # processed_data = one_hot_encoder.fit_transform(data[:,0:2]).toarray()
    # processed_data = np.append(processed_data, data[:,2:6], 1)

    # Do not use stock symbol and day parameters for training
    if type == 0:
        processed_data = data[:, 2:6]
    else:
        # processed_data = np.concatenate([data[:,2:5], data[:,7:9]])
        processed_data = data[:, 2:9]
        processed_data = np.delete(processed_data, 3, 1)
        processed_data = np.delete(processed_data, 3, 1)

    processed_data = preprocessing.normalize(processed_data)
    np.random.shuffle(processed_data)
    return processed_data
def reverseProcessing(data):
    # label_encoder = preprocessing.LabelEncoder()
    label = predict_encoder.inverse_transform(data)
    return label

def main():
    data = np.array(getData())
    learn(data)

# main()
# test scope
def getData_test():
    # with open('test_data/dat_0.csv') as csv_file:
    #     reader = csv.reader(csv_file)
    #     data = list(reader)
    #
    # return data

    my_data = np.genfromtxt('test_data/dat_0.csv', delimiter=',')
    print("done")

    return my_data

    # result = np.array(list(csv.reader(open("test_data/dat_0.csv", "rb"), delimiter=","))).astype("float")
    # return result

def preprocessData_test(data, type=0):
    label_encoder = preprocessing.LabelEncoder()

    # row 0: symbol, row 1: date, row 2: sentiment score, row 3: sentiment magnitude, row 4: open, row 5: close, row 6: adj close, row 7: high, row 8: low, row 9: volume

    data[:,0] = label_encoder.fit_transform(data[:,0])
    data = data.astype(float)

    # Uncomment lines below to use stock symbol and day parameters
    # WARNING: Epochs may be extremely slow
    # processed_data = one_hot_encoder.fit_transform(data[:,0:2]).toarray()
    # processed_data = np.append(processed_data, data[:,2:6], 1)
    
    # Do not use stock symbol and day parameters for training
    processed_data = None
    if type == 0:
        processed_data = data[:, 2:6]
    else:
        # processed_data = np.concatenate([data[:,2:5], data[:,7:9]])
        processed_data = data[:, 2:9]
        processed_data = np.delete(processed_data, 3, 1)
        processed_data = np.delete(processed_data, 3, 1)
    processed_data = preprocessing.normalize(processed_data)
    np.random.shuffle(processed_data)
    return processed_data
def learn_test_model(data):
    data = preprocessData_test(data)
    num_params = data.shape[1] - 1

    X = data[:, 0:num_params]
    Y = data[:, num_params].reshape(-1, 1)

    # Split the data into training and testing sets (70/30)
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X, Y, test_size=0.30)

    #test
    np.delete(train_Y, 0)
    np.append(train_Y, train_Y[-1])
    np.delete(test_Y, 0)
    np.append(test_Y, test_Y[-1])

    train_opening_price = train_X[:, num_params - 1].reshape(-1, 1)
    test_opening_price = test_X[:, num_params - 1].reshape(-1, 1)

    # Get the initial stock prices for computing the relative cost
    stock_data = tf.placeholder(tf.float32, shape=[None, num_params], name='stock_data')
    opening_price = tf.placeholder(tf.float32, shape=[None, 1], name='opening_price')
    stock_price = tf.placeholder(tf.float32, shape=[None, 1], name='stock_price')

    # Number of neurons in the hidden layer
    n_hidden_1 = 3
    n_hidden_2 = 3

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([1]))
    }

    # Implement dropout to reduce overfitting
    keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
    keep_prob_hidden = tf.placeholder(tf.float32, name='keep_prob_hidden')

    # Hidden layers
    input_dropout = tf.nn.dropout(stock_data, keep_prob_input)
    #layer_1 = slim.fully_connected(input_dropout, n_hidden_1, biases_initializer=None, activation_fn=tf.nn.relu)
    #layer_1_dropout = tf.nn.dropout(layer_1, keep_prob_hidden)
    layer_2 = slim.fully_connected(input_dropout, n_hidden_1, biases_initializer=None, activation_fn=tf.nn.relu)
    layer_2_dropout = tf.nn.dropout(layer_2, keep_prob_hidden)
    output_layer = tf.add(tf.matmul(layer_2_dropout, weights['out']), biases['out'], name='output_layer')

    learning_rate = 1e-4
    cost_function = tf.reduce_mean(tf.pow(tf.div(tf.subtract(stock_price, output_layer), opening_price), 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    last_cost = 0
    tolerance = 1e-6
    epochs = 1
    max_epochs = 1e6

    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        while True:
            sess.run(optimizer,
                     feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y,
                                keep_prob_input: 0.8, keep_prob_hidden: 0.5})

            if epochs % 100 == 0:
                cost = sess.run(cost_function, feed_dict={stock_data: train_X, opening_price: train_opening_price,
                                                          stock_price: train_Y, keep_prob_input: 0.8,
                                                          keep_prob_hidden: 0.5})
                print "Epoch: %d: Error: %f" % (epochs, cost)

                if abs(cost - last_cost) <= tolerance or epochs > max_epochs:
                    print "Converged."
                    break
                last_cost = cost

            epochs += 1

        saver = tf.train.Saver()
        model_path = Model_Path + '/model.ckpt'
        save_path = saver.save(sess, model_path)
        print("Model saved in path: %s" % save_path)


        print "Test error: ", sess.run(cost_function, feed_dict={stock_data: test_X, opening_price: test_opening_price,
                                                                 stock_price: test_Y, keep_prob_input: 1.0,
                                                                 keep_prob_hidden: 1.0})
        test_results = sess.run(output_layer, feed_dict={stock_data: test_X, stock_price: test_Y, keep_prob_input: 1.0,
                                                         keep_prob_hidden: 1.0})

    avg_perc_error = 0
    max_perc_error = 0
    mei = 0
    for i in range(len(test_Y)):
        actual_change = abs(test_Y[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        predicted_change = abs(test_results[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        delta = abs(actual_change - predicted_change)
        avg_perc_error = avg_perc_error + delta
        if delta > max_perc_error:
            max_perc_error = delta
            mei = i

    avg_perc_error = (avg_perc_error * 100) / len(test_Y)
    max_perc_error *= 100
    print "Maximum percentage error: %f\nAverage percentage error: %f\n" % (max_perc_error, avg_perc_error)

def prediction(dt):
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    model_path = os.getcwd() + '/model/model.ckpt'
    meta_path = os.getcwd() + '/model/model.ckpt.meta'
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(session, model_path)
    # p_test = sess.run(output_layer, feed_dict={stock_data: p_x, stock_price: p_y, keep_prob_input: 1.0,
    #                                            keep_prob_hidden: 1.0})
    graph = tf.get_default_graph()
    # output_layer = tf.get_collection('output_layer')[0]
    # stock_data = tf.get_collection('placeholder')[0]
    # stock_price = tf.get_collection('placeholder')[1]
    stock_data = graph.get_tensor_by_name('stock_data:0')
    opening_price = graph.get_tensor_by_name('opening_price:0')
    stock_price = graph.get_tensor_by_name('stock_price:0')
    keep_prob_input = graph.get_tensor_by_name('keep_prob_input:0')
    keep_prob_hidden = graph.get_tensor_by_name('keep_prob_hidden:0')
    output_layer = graph.get_tensor_by_name('output_layer:0')
    today = datetime.datetime.today()
    t_news_file = open('test_data/news/' + dt.strftime('%Y-%m-%d') + '.csv')
    print('Collecting Test Data for %s' % (dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    p_single_data = []
    p_next_data = []
    p_csv_file = csv.reader(t_news_file)

    for row in p_csv_file:
        p_stockdata = gendata.getStockData_Test(row[0], dt)
        if (p_stockdata == -1):
            continue
    
        # sentdata = sentiment.analyzeText(row[1])
        print("Origin String: %s" % row[1])
        filtered_string = gendata.filterString(row[1])
        filtered_string = filtered_string.decode("ascii", errors="ignore").encode()
        print("Filtered String: %s" % filtered_string)

        try:
            sentdata = sentiment.analyzeText(filtered_string)
        except:
            enchant_string = gendata.filterByEnchant(filtered_string)
            print("Enchanted String: %s" % enchant_string)
            try:
                sentdata = sentiment.analyzeText(enchant_string)
            except:
                sentdata = sentiment.analyzeText(row[0])

        data = []
        data.extend((row[0], dt.timetuple().tm_yday))
        data.extend((sentdata.score, sentdata.magnitude))
        data.extend((p_stockdata[5], p_stockdata[1], p_stockdata[0], p_stockdata[3], p_stockdata[4], p_stockdata[6]))

        p_single_data.append(data)

        if (dt < today):
            dtnext = dt + datetime.timedelta(days=1)
            p_stocknextday = gendata.getStockData_Test(row[0], dtnext)
            if (p_stocknextday is not -1):
                datanext = []
                datanext.extend((row[0], dtnext.timetuple().tm_yday))
                datanext.extend((sentdata.score, sentdata.magnitude))
                datanext.extend((p_stocknextday[5], p_stocknextday[1], p_stocknextday[0], p_stocknextday[3], p_stocknextday[4], p_stocknextday[6]))

                p_next_data.append(datanext)

    if len(p_single_data) > 0:
        p_data = np.array(p_single_data)
        print('Test Data collected for %s data:' % (dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d') + ', '.join(str(c) for c in p_data))
        p_origin = p_data
        p_data = normalizeInput(p_data, 0)
        params = p_data.shape[1] - 1
        p_x = p_data[:, 0:params]
        p_y = p_data[:, params].reshape(-1, 1)
        p_test = session.run(output_layer, feed_dict={stock_data: p_x, stock_price: p_y, keep_prob_input: 1.0, keep_prob_hidden: 1.0})

        origin_test = p_origin[:, 2:6]

        for i in range(origin_test.shape[0]):
            norm = np.linalg.norm(origin_test[i, :], axis=0)
            p_test[i, 0] = p_test[i, 0] * norm

        result_date = dt + datetime.timedelta(days=1)
        print('Predict Data for %s :' % result_date.strftime('%Y-%m-%d') + ', '.join('Stock %s :%s' % (d[0], str(c)) for d, c in zip(p_single_data, p_test)))

        p_compare_data = []
        if (dt < today and len(p_next_data) > 0):
            p_compare_data = p_single_data
        else:
            p_compare_data = p_single_data

        s_name = []
        s_close = []
        s_adjclose = []
        s_low = []
        s_high = []
        s_volume = []
        s_open = []
        s_next_close = []
        s_sentiment_score = []
        s_sentiment_magnitude = []
        s_average = []
        for i in range(len(p_compare_data)):
            s_name.append(p_compare_data[i][0])
            s_close.append(p_compare_data[i][5])
            s_adjclose.append(p_compare_data[i][6])
            s_low.append(p_compare_data[i][8])
            s_high.append(p_compare_data[i][7])
            s_volume.append(p_compare_data[i][9])
            s_open.append(p_compare_data[i][4])
            s_sentiment_score.append(p_origin[i][2])
            s_sentiment_magnitude.append(p_origin[i][3])
            average = (float(p_compare_data[i][8]) + float(p_compare_data[i][7]) + float(p_compare_data[i][4]) + float(p_compare_data[i][5]))/ 4
            s_average.append(average)
            if dt < today:
                s_next_close.append(p_next_data[i][5])
        
        if (dt == today):
            p_result = pd.DataFrame({
                'symbol':s_name,
                'prediction':p_test[:, 0],
                'previous open': s_open,
                'previous close': s_close,
                'previous high': s_high,
                'previous low': s_low,
                'sentiment score': s_sentiment_score,
                'sentiment magnitude': s_sentiment_magnitude,
                'stock average': s_average,
                'previous volume': s_volume})
        else:
            p_result = pd.DataFrame({
                    'symbol': s_name,
                    'prediction': p_test[:, 0],
                    'open': s_open,
                    'close': s_close,
                    'sentiment score': s_sentiment_score,
                    'sentiment magnitude': s_sentiment_magnitude,
                    'next close': s_next_close,
                    # 'prev close': s_adjclose,
                    'high': s_high,
                    'low': s_low,
                    'stock average': s_average,
                    'volume': s_volume})

        csv_name = 'test_data/result/result-%s.csv' % (result_date.date())
        # file_result = open('data/result/' + csv_name, 'rb+')
        p_result.to_csv(csv_name)


def test_prediction(type):
    test_dates = []
    # t_date = datetime.datetime(2018, 03, 01)
    # for i in range(4):
    #     test_dates.append(t_date)
    #     t_date += datetime.timedelta(days=1)
    t_date = datetime.datetime(2018, 03, 16)
    for i in range(3):
        test_dates.append(t_date)
        t_date += datetime.timedelta(days=1)

    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        model_path = os.getcwd() + '/model/model_' + str(type) + '.ckpt'
        meta_path = os.getcwd() + '/model/model_' + str(type) + '.ckpt.meta'
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(session, model_path)
        # p_test = sess.run(output_layer, feed_dict={stock_data: p_x, stock_price: p_y, keep_prob_input: 1.0,
        #                                            keep_prob_hidden: 1.0})
        graph = tf.get_default_graph()
        # output_layer = tf.get_collection('output_layer')[0]
        # stock_data = tf.get_collection('placeholder')[0]
        # stock_price = tf.get_collection('placeholder')[1]
        stock_data = graph.get_tensor_by_name('stock_data:0')
        opening_price = graph.get_tensor_by_name('opening_price:0')
        stock_price = graph.get_tensor_by_name('stock_price:0')
        keep_prob_input = graph.get_tensor_by_name('keep_prob_input:0')
        keep_prob_hidden = graph.get_tensor_by_name('keep_prob_hidden:0')
        output_layer = graph.get_tensor_by_name('output_layer:0')

        print('Prediction Test Dates:' + ', '.join(d.strftime('%Y-%m-%d') for d in test_dates))
        precit_data = []
        for dt in test_dates:
            t_news_file = open('test_data/news/' + dt.strftime('%Y-%m-%d') + '.csv')
            print('Collecting Test Data for %s' % (dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
            p_single_data = []
            p_csv_file = csv.reader(t_news_file)

            for row in p_csv_file:
                p_stockdata = gendata.getStockData_Test(row[0], dt)
                if (p_stockdata == -1):
                    continue
                sentdata = sentiment.analyzeText(row[1])

                data = []
                data.extend((row[0], dt.timetuple().tm_yday))
                data.extend((sentdata.score, sentdata.magnitude))
                data.extend(
                    (p_stockdata[5], p_stockdata[1], p_stockdata[0], p_stockdata[3], p_stockdata[4], p_stockdata[6]))

                p_single_data.append(data)

            if len(p_single_data) > 0:
                p_data = np.array(p_single_data)
                print('Test Data collected for %s data:' % (dt + datetime.timedelta(days=1)).strftime(
                    '%Y-%m-%d') + ', '.join(str(c) for c in p_data))
                p_origin = p_data
                p_data = normalizeInput(p_data, type)
                if type == 0:
                    params = p_data.shape[1] - 1
                else:
                    params = p_data.shape[1] - 2
                
                p_x = p_data[:, 0:params]
                p_y = p_data[:, params].reshape(-1, 1)
                p_test = session.run(output_layer, feed_dict={stock_data: p_x, stock_price: p_y, keep_prob_input: 1.0,
                                                           keep_prob_hidden: 1.0})

                
                origin_test =None
                if type == 0:
                    origin_test = p_origin[:, 2:6]
                else:
                    origin_test = p_origin[:, 2:7]

                for i in range(origin_test.shape[0]):
                    norm = np.linalg.norm(origin_test[i, :], axis=0)
                    p_test[i, 0] = p_test[i, 0] * norm

                result_date = dt + datetime.timedelta(days=1)
                print('Predict Data for %s :' % result_date.strftime('%Y-%m-%d') + ', '.join(
                    'Stock %s :%s' % (d[0], str(c)) for d, c in zip(p_single_data, p_test)))

                s_name = []
                s_close = []
                s_adjclose = []
                s_low = []
                s_high = []
                s_volume = []
                s_open = []
                for i in range(len(p_single_data)):
                    s_name.append(p_single_data[i][0])
                    s_close.append(p_single_data[i][5])
                    s_adjclose.append(p_single_data[i][6])
                    s_low.append(p_single_data[i][8])
                    s_high.append(p_single_data[i][7])
                    s_volume.append(p_single_data[i][9])
                    s_open.append(p_single_data[i][4])
                p_result = pd.DataFrame({
                    'symbol': s_name,
                    'prediction': p_test[:, 0],
                    'open': s_open,
                    'close': s_close,
                    # 'prev close': s_adjclose,
                    'high': s_high,
                    'low': s_low,
                    'volume': s_volume})

                csv_name = 'test_data/result/result-%s_model_%d.csv' % (result_date.date(), type)
                # file_result = open('data/result/' + csv_name, 'rb+')
                p_result.to_csv(csv_name)

def learn_test(data):
    data = preprocessData(data)
    num_params = data.shape[1] - 1

    X = data[:,0:num_params]
    Y = data[:,num_params].reshape(-1, 1)

    # Split the data into training and testing sets (70/30)
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X, Y, test_size=0.30)
    train_opening_price = train_X[:, num_params - 1].reshape(-1, 1)
    test_opening_price = test_X[:, num_params - 1].reshape(-1, 1)

    # Get the initial stock prices for computing the relative cost
    stock_data = tf.placeholder(tf.float32, [None, num_params])
    opening_price = tf.placeholder(tf.float32, [None, 1])
    stock_price = tf.placeholder(tf.float32, [None, 1])

    # Number of neurons in the hidden layer
    n_hidden_1 = 3
    n_hidden_2 = 3

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([1]))
    }

    # Implement dropout to reduce overfitting
    keep_prob_input = tf.placeholder(tf.float32)
    keep_prob_hidden = tf.placeholder(tf.float32)

    # Hidden layers
    input_dropout = tf.nn.dropout(stock_data, keep_prob_input)
    layer_1 = slim.fully_connected(input_dropout, n_hidden_1, biases_initializer = None, activation_fn = tf.nn.relu)
    layer_1_dropout = tf.nn.dropout(layer_1, keep_prob_hidden)
    layer_2 = slim.fully_connected(input_dropout, n_hidden_1, biases_initializer = None, activation_fn = tf.nn.relu)
    layer_2_dropout = tf.nn.dropout(layer_2, keep_prob_hidden)
    output_layer = tf.add(tf.matmul(layer_2_dropout, weights['out']), biases['out'])

    learning_rate = 1e-4
    cost_function = tf.reduce_mean(tf.pow(tf.div(tf.subtract(stock_price, output_layer), opening_price), 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    last_cost = 0
    tolerance = 1e-6
    epochs = 1
    max_epochs = 1e6

    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        while True:
            sess.run(optimizer, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y, keep_prob_input: 0.8, keep_prob_hidden: 0.5})

            if epochs % 100 == 0:
                cost = sess.run(cost_function, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y, keep_prob_input: 0.8, keep_prob_hidden: 0.5})
                print "Epoch: %d: Error: %f" %(epochs, cost)

                if abs(cost - last_cost) <= tolerance or epochs > max_epochs:
                    print "Converged."
                    break
                last_cost = cost

            epochs += 1

        print "Test error: ", sess.run(cost_function, feed_dict={stock_data: test_X, opening_price: test_opening_price, stock_price: test_Y, keep_prob_input: 1.0, keep_prob_hidden: 1.0})
        test_results = sess.run(output_layer, feed_dict={stock_data: test_X, stock_price: test_Y, keep_prob_input: 1.0, keep_prob_hidden: 1.0})

    avg_perc_error = 0
    max_perc_error = 0
    mei = 0
    for i in range(len(test_Y)):
        actual_change = abs(test_Y[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        predicted_change = abs(test_results[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        delta = abs(actual_change - predicted_change)
        avg_perc_error = avg_perc_error + delta
        if delta > max_perc_error:
            max_perc_error = delta
            mei = i

    avg_perc_error = (avg_perc_error * 100) / len(test_Y)
    max_perc_error *= 100
    print "Maximum percentage error: %f\nAverage percentage error: %f\n" % (max_perc_error, avg_perc_error)

    test_dates = []    
    #p_f = open(FNAME, 'r')
       # stocks = p_f.readLines()
    t_date = datetime.datetime(2018, 03, 01)
    for i in range(4):
        test_dates.append(t_date)
        t_date += datetime.timedelta(days=1)
    t_date = datetime.datetime(2018, 03, 16)
    for i in range(3):
        test_dates.append(t_date)
        t_date += datetime.timedelta(days=1)

    print('Prediction Test Dates:' + ', '.join(d.strftime('%Y-%m-%d') for d in test_dates))
    precit_data = []
    for dt in test_dates:
        t_news_file = open('test_data/news/' + dt.strftime('%Y-%m-%d') + '.csv')
        print('Collecting Test Data for %s' % (dt + datetime.timedelta(days = 1)).strftime('%Y-%m-%d'))
        p_single_data = []
        p_csv_file = csv.reader(t_news_file)

        for row in p_csv_file:
            p_stockdata = gendata.getStockData_Test(row[0], dt)
            if (p_stockdata == -1):
                continue
            sentdata = sentiment.analyzeText(row[1])

            data = []
            data.extend((row[0], dt.timetuple().tm_yday))
            data.extend((sentdata.score, sentdata.magnitude))
            data.extend((p_stockdata[5],p_stockdata[1],p_stockdata[0],p_stockdata[3],p_stockdata[4],p_stockdata[6]))

            p_single_data.append(data)

        if len(p_single_data) > 0:
            p_data = np.array(p_single_data)
            print('Test Data collected for %s data:' % (dt + datetime.timedelta(days = 1)).strftime('%Y-%m-%d') + ', '.join(str(c) for c in p_data))
            p_origin = p_data
            p_data = normalizeInput(p_data)
            params = p_data.shape[1] - 1
            p_x = p_data[:,0:params]
            p_y = p_data[:,params].reshape(-1, 1)
            p_test = sess.run(output_layer, feed_dict={stock_data: p_x, stock_price: p_y, keep_prob_input: 1.0,
                                                                 keep_prob_hidden: 1.0})

            orign_test = p_origin[:, 2:6]
            for i in range(orign_test.shape[0]):
                norm = np.linalg.norm(orign_test[i, :],axis=0)
                p_test[i,0] = p_test[i,0] * norm

            result_date = dt + datetime.timedelta(days=1)
            print('Predict Data for %s :' % result_date.strftime('%Y-%m-%d')+ ', '.join('Stock %s :%s'%(d[0], str(c)) for d, c in zip(p_single_data, p_test)))

            s_name = []
            s_close = []
            s_adjclose = []
            s_low = []
            s_high = []
            s_volume = []
            s_open = []
            for i in range(len(p_single_data)):
                s_name.append(p_single_data[i][0])
                s_close.append(p_single_data[i][5])
                s_adjclose.append(p_single_data[i][6])
                s_low.append(p_single_data[i][8])
                s_high.append(p_single_data[i][7])
                s_volume.append(p_single_data[i][9])
                s_open.append(p_single_data[i][4])
            p_result = pd.DataFrame({
                'symbol': s_name,
                'prediction': p_test[:, 0],
                'open': s_open,
                'close': s_close,
                'prev close': s_adjclose,
                'high': s_high,
                'low': s_low,
                'volume': s_volume})

            csv_name = 'test_data/result/result-%s.csv' % result_date.date()
            #file_result = open('data/result/' + csv_name, 'rb+')
            p_result.to_csv(csv_name)

def main_test():
    data = np.array(getData_test())
    #learn_test(data)
    learn_test_model(data, 0)
    learn_test_model(data, 1)
    test_prediction(0)
    test_prediction(1)    


def buildModel():
    #array(array(['looooooool'], dtype=object), dtype='|S10')
    data = np.array(getData_test())

    t = data[:, None]

    learn_test_model(data)

# main_test()
#buildModel()