import tensorflow as tf
import numpy as np
import TomTheDriver
import random
import os


input = 7
n_nodes_hl = 5

n_classes = 3 # -1..0..1

input_data = 0
labels = 0
test_data = 0
test_labels = 0

train_set_size = 0

buy_threshold = -1
sell_threshold = 1

x=0
y=0

def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([input, n_nodes_hl])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}
    # (input_Data*weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(xdata):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    # default lr = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    hm_epochs = 100
    batch_size = 300

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for i in range(int(train_set_size/batch_size)):
                epoch_x = xdata[i*batch_size:(i+1)*batch_size]
                epoch_y = labels[i*batch_size:(i+1)*batch_size]

                i, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y: epoch_y})
                epoch_loss += c
            #print('Epoch', epoch, ' Completed out of ', hm_epochs, 'loss: ',epoch_loss)
            if epoch_loss != epoch_loss:
                break;
            #correct = tf.equal(tf.arg_max(prediction, 1), tf.argmax(y, 1))
            #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            #print('Accuracy: ', accuracy.eval({x: xdata, y: labels}))
            #print('Accuracy over unseen data: ', accuracy.eval({x:test_data, y:test_data}))
        
        correct = tf.equal(tf.arg_max(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy over unseen data: ', accuracy.eval({x:test_data, y:test_data}))
        return prediction.eval({x:test_data, y:test_labels})

def calculate_fitness(predictions, threshold):
    budget = 1000000
    total_shares = 0
    closing_prices = TomTheDriver.input_data[-len(predictions):,3]

    for i in range(len(predictions)):
        pred_decision = np.argmax(predictions[i])
        if pred_decision == 1:
            continue
        # each trade consists of 1% of the overall budget...
        amount_to_trade = np.round(0.01 * budget/closing_prices[i])

        if pred_decision == 0: #sell decision
            if abs(predictions[i][0]/predictions[i][2]) > threshold:
                #place sell trade, check whether there are opened buy positions

                if total_shares > 0:
                    if total_shares >= amount_to_trade:
                        total_shares -= amount_to_trade
                        budget += amount_to_trade * closing_prices[i]
                        amount_to_trade = 0
                    else:
                        budget += total_shares * closing_prices[i]
                        amount_to_trade -= total_shares
                        total_shares = 0
                total_shares -= amount_to_trade
                budget -= amount_to_trade * closing_prices[i]

        elif pred_decision == 2: #buy decision
            if abs(predictions[i][2]/predictions[i][0]) > threshold:
                #place buy trade, check whether there are opened sell posisions
                if total_shares < 0:
                    if abs(total_shares) >= amount_to_trade:
                        total_shares += amount_to_trade
                        budget += amount_to_trade * closing_prices[i]
                        amount_to_trade = 0
                    else:
                        budget += abs(total_shares) * closing_prices[i]
                        amount_to_trade -= total_shares
                        total_shares = 0
                total_shares += amount_to_trade
                budget -= amount_to_trade * closing_prices[i]

    # At the end of all periods, sell/buy every stock owned
    budget += abs(total_shares) * closing_prices[-1]
    return  budget


def build_nn_get_fitness(dna=[1, 0, 1, 1, 1, 0, 0, 20, 0.8]):
    global x, y, input_data, labels, input, n_nodes_hl, buy_threshold, sell_threshold, train_set_size
    global test_labels, test_data
    input_data = 0
    labels = 0
    predictions = 0
    #Check whether the current configuration has been build earlier...
    if os.path.exists(''.join(str(i) for i in dna[:-1])+".npy"):
        predictions = np.load(''.join(str(i) for i in dna[:-1])+".npy")
    else:
        n_input = 0
        #initialize number of nodes in hidden layer
        n_nodes_hl = dna[7]
        #select inputs
        for i in range(7):
            if dna[i] == 1:
                n_input += 1
                if(np.isscalar(input_data)):
                    input_data = TomTheDriver.indicators[:, i]
                else:
                    input_data = np.hstack((input_data, TomTheDriver.indicators[:, i]))
        input_data = np.reshape(input_data, [len(TomTheDriver.indicators), n_input])
        input = n_input

        train_set_size = round(len(input_data) * 0.8)
        test_data = input_data[train_set_size + 1:]
        test_labels = TomTheDriver.labels[train_set_size+1:]

        input_data = input_data[:train_set_size]
        labels = TomTheDriver.labels[:train_set_size]


        input_data = np.array(input_data)
        input_data = input_data.astype(np.float32)
        labels = np.array(labels)

        #initialize x and y for nn
        x = tf.placeholder('float', [None, n_input])
        y = tf.placeholder('float')

        predictions = (train_neural_network(input_data))
        np.save(''.join(str(i) for i in dna[:-1]), predictions)



    #get the fitness of the configuration...
    return calculate_fitness(predictions, dna[-1])

def run_genetic_algorithm():
    generations = 50
    n_traders = 50
    list = []

    # randomly generate first generation
    while len(list) < n_traders:
        temp_trader = np.random.choice([0,1], size=(7,))
        if np.sum(temp_trader) == 0:
            continue
        temp_trader = temp_trader.tolist()
        temp_trader.append(random.randrange(1, 70))
        temp_trader.append(random.uniform(0.01,1.0))
        if list.__contains__(temp_trader) == False:
            list.append(temp_trader)

    for g in range(generations):

        for trader in list:
            while len(trader) > 9:
                trader.pop()
            print('Training & testing trader: ',trader)
            trader.append(build_nn_get_fitness(trader))
            print(trader[-1])
            if trader[-1] == 1000000.0:
                trader[-1] *=-1
            #print("Generation: ",g, "Fitness: ", trader[-1])


        list = sorted(list, key=lambda x: x[-1], reverse=True)
        print("Generation: ", g+1,"Top Traders: ", list[:5])

        for i in range(int(n_traders/2)-1, n_traders-1):
            parent_id = i - int(n_traders/2)
            list[i] = list[parent_id]

            for f in range(0,7):
                if random.random() > (generations-g)/(generations):
                    list[i][f] = random.choice([0,1])
            if sum(list[i][:7]) == 0:
                i -= 1

            #list[i][7] = abs(int(random.gauss(mu=list[parent_id][7], sigma=(generations - g))))
            #threshold
            #list[i][8] = abs(random.gauss(mu=list[parent_id][8], sigma=(generations - g)/100))

            list[i][7] = random.randrange(1, 100)
            # threshold
            list[i][8] = random.uniform(0.01,2.0)

run_genetic_algorithm()
#print(build_nn_get_fitness())
