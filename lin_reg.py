import csv
import numpy as np
from sklearn import cross_validation
from sklearn import preprocessing
import tensorflow as tf
import os
import pandas as pd
import datetime
import gendata
import sentiment

def getData():
    with open('test_data/dat_0.csv') as csv_file:
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

    # Split the data into training, validation, and testing sets (60/20/20)
    train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(X, Y, test_size = 0.40)
    test_X, valid_X, test_Y, valid_Y = cross_validation.train_test_split(test_X, test_Y, test_size = 0.50)

    # Get the initial stock prices for computing the relative cost
    train_opening_price = train_X[:, num_params - 1].reshape(-1, 1)
    valid_opening_price = valid_X[:, num_params - 1].reshape(-1, 1)
    test_opening_price = test_X[:, num_params - 1].reshape(-1, 1)

    stock_data = tf.placeholder(tf.float32, [None, num_params], name='stock_data')
    opening_price = tf.placeholder(tf.float32, [None, 1], name='opening_price')
    stock_price = tf.placeholder(tf.float32, [None, 1], name='stock_price')

    W = tf.Variable(tf.random_uniform([num_params, 1], dtype=tf.float32), name="Weight")
    y = tf.matmul(stock_data, W, name="operator")

    learning_rate = 1e-3
    cost_function = tf.reduce_mean(tf.pow(tf.div(tf.subtract(stock_price, y), opening_price), 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

    last_train_cost = 0
    best_valid_cost = 1e6
    best_valid_epoch = 0
    valid_epoch_threshold = 1
    tolerance = 1e-12
    epochs = 1
    max_epochs = 1e6
    # saver = tf.train.Saver([W])

    sess = tf.Session()
    with sess.as_default():
        init = tf.global_variables_initializer()
        sess.run(init)

        while True:
            sess.run(optimizer, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y})

            if epochs % 100 == 0:
                train_cost = sess.run(cost_function, feed_dict={stock_data: train_X, opening_price: train_opening_price, stock_price: train_Y})
                valid_cost = sess.run(cost_function, feed_dict={stock_data: valid_X, opening_price: valid_opening_price, stock_price: valid_Y})
                print "Epoch: %d: Training error: %f Validation error: %f" %(epochs, train_cost, valid_cost)

                test_results = sess.run(y, feed_dict={stock_data: test_X, opening_price: test_opening_price,
                                                      stock_price: test_Y})

                avg_perc_error = 0
                max_perc_error = 0
                for i in range(len(test_Y)):
                    actual_change = abs(test_Y[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
                    predicted_change = abs(test_results[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
                    delta = abs(actual_change - predicted_change)
                    max_perc_error = max(max_perc_error, delta)
                    avg_perc_error = avg_perc_error + delta

                avg_perc_error = (avg_perc_error * 100) / len(test_Y)
                max_perc_error *= 100
                print "Maximum percentage error: %f\nAverage percentage error: %f\n" % (max_perc_error, avg_perc_error)

                if(valid_cost < best_valid_cost):
                    best_valid_cost = valid_cost
                    best_valid_epoch = epochs
                    # save_path = saver.save(sess, 'lr-model-valid')

                if(valid_epoch_threshold <= epochs - best_valid_epoch):
                    # saver.restore(sess, save_path)
                    print "Early stopping."
                    break

                if abs(train_cost - last_train_cost) <= tolerance or epochs > max_epochs:
                    print "Converged."
                    break

                last_train_cost = train_cost

            epochs += 1

        saver = tf.train.Saver()
        model_path = 'model/lin-model.ckpt'
        save_path = saver.save(sess, model_path)
        print("Model saved in path: %s" % save_path)

        print "Test error: ", sess.run(cost_function, feed_dict={stock_data: test_X, opening_price: test_opening_price, stock_price: test_Y})
        test_results = sess.run(y, feed_dict={stock_data: test_X, opening_price: test_opening_price, stock_price: test_Y})

    avg_perc_error = 0
    max_perc_error = 0
    for i in range(len(test_Y)):
        actual_change = abs(test_Y[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        predicted_change = abs(test_results[i][0] - test_X[i][num_params - 1]) / test_X[i][num_params - 1]
        delta = abs(actual_change - predicted_change)
        max_perc_error = max(max_perc_error, delta)
        avg_perc_error = avg_perc_error + delta

    avg_perc_error = (avg_perc_error * 100) / len(test_Y)
    max_perc_error *= 100
    print "Maximum percentage error: %f\nAverage percentage error: %f\n" % (max_perc_error, avg_perc_error)


def prediction(dt):
    tf.reset_default_graph()
    session = tf.Session()
    #init = tf.global_variables_initializer()
    #session.run(init)

    model_path = os.getcwd() + '/model/lin-model.ckpt'
    meta_path = os.getcwd() + '/model/lin-model.ckpt.meta'
    #saver = tf.train.Saver()
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
    W = tf.get_variable("Weight",[3,1],dtype=np.float32)
    y = graph.get_tensor_by_name('operator:0')
    #keep_prob_input = graph.get_tensor_by_name('keep_prob_input:0')
    #keep_prob_hidden = graph.get_tensor_by_name('keep_prob_hidden:0')
    #output_layer = graph.get_tensor_by_name('output_layer:0')
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
                datanext.extend((p_stocknextday[5], p_stocknextday[1], p_stocknextday[0], p_stocknextday[3],
                                 p_stocknextday[4], p_stocknextday[6]))

                p_next_data.append(datanext)

    if len(p_single_data) > 0:
        p_data = np.array(p_single_data)
        # print('Test Data collected for %s data:' % (dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d') + ', '.join(
#             str(c) for c in p_data))
        
        print('Test Data collected for %s data:' % (dt + datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
        
        p_origin = p_data
        p_data = preprocessData(p_data)
        params = p_data.shape[1] - 1
        p_x = p_data[:, 0:params]
        p_y = p_data[:, params].reshape(-1, 1)
        p_opening_price = p_x[:, params - 1].reshape(-1, 1)
        shape = p_x.shape
        print(shape)
        p_test = session.run(y, feed_dict={stock_data: p_x, opening_price: p_opening_price, stock_price: p_y})

        origin_test = p_origin[:, 2:6]

        for i in range(origin_test.shape[0]):
            norm = np.linalg.norm(origin_test[i, :], axis=0)
            p_test[i, 0] = p_test[i, 0] * norm

        result_date = dt + datetime.timedelta(days=1)
        print('Predict Data for %s :' % result_date.strftime('%Y-%m-%d') + ', '.join(
            'Stock %s :%s' % (d[0], str(c)) for d, c in zip(p_single_data, p_test)))

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
        for i in range(len(p_compare_data)):
            s_name.append(p_compare_data[i][0])
            s_close.append(p_compare_data[i][5])
            s_adjclose.append(p_compare_data[i][6])
            s_low.append(p_compare_data[i][8])
            s_high.append(p_compare_data[i][7])
            s_volume.append(p_compare_data[i][9])
            s_open.append(p_compare_data[i][4])
            if dt < today:
                s_next_close.append(p_next_data[i][5])

        if (dt == today):
            p_result = pd.DataFrame({
                'symbol': s_name,
                'prediction': p_test[:, 0],
                'previous open': s_open,
                'previous close': s_close,
                'previous high': s_high,
                'previous low': s_low,
                'previous volume': s_volume})
        else:
            p_result = pd.DataFrame({
                'symbol': s_name,
                'prediction': p_test[:, 0],
                'open': s_open,
                'close': s_close,
                'next close': s_next_close,
                # 'prev close': s_adjclose,
                'high': s_high,
                'low': s_low,
                'volume': s_volume})

        csv_name = 'test_data/line-result/result-%s.csv' % (result_date.date())
        # file_result = open('data/result/' + csv_name, 'rb+')
        p_result.to_csv(csv_name)

def main():
    data = np.array(getData())
    learn(data)

#main()
