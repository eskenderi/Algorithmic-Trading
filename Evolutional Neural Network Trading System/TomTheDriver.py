# best tools to trade obtained from: http://www.investopedia.com/slide-show/tools-of-the-trade/
import numpy as np
import os
#import NeuralNetwork as nn

data_file = open('table_spy.csv', mode='r').readlines()
data_file = np.array(data_file)
timeframe = 15 #combine 15 min together
SMA_PERIOD = 12
EMA_PERIOD = 12
AROON_PERIOD = 12
RSI_PERIOD = 14
STOCHASTIC_OSCH_PERIOD = 14
input_data = 0
returns = 0
labels = 0
indicators = 0


def load_data():
    open = 0
    high = 0
    low = 9999
    volume = 0
    temp = 0

    for i in range(len(data_file)):

        row = data_file[i].replace("\n","").split(",")
        row = np.array(row[:][2:])

        row = row.astype('float')
        if (i+1)%timeframe == 1:#first min out of 5
            open = row[0]
        if high < row[1]:
            high = row[1]
        if low > row[2]:
            low = row[2]
        volume += row[4]
        if (i + 1) % timeframe == 0.0:  # last min out of 5
            row[0] = open
            row[1] = high
            row[2] = low
            row[4] = volume
            if np.isscalar(temp):
                temp = row
            else:
                temp = np.vstack((temp, row))
            open = 0
            high = 0
            low = 99999
            volume = 0
    return temp

def load_returns():
    returns = np.zeros(len(input_data))

    for i in range(1,len(returns)):
        returns[i] = input_data[i][3] - input_data[i - 1][3]

    print("Returns loaded...", len(returns))
    return returns

def load_indicators():
    global indicators
    indicators = np.zeros((len(input_data), 7))#7 indicators
    for i in range(len(indicators)):
        indicators[i][0] = on_balance_volume(i)
        indicators[i][1] = accumulation_distribution_line(i)
        indicators[i][2] = simple_moving_average(i)
        indicators[i][3] = aroon_indicator(i)
        indicators[i][4] = exponential_moving_average(i)
        indicators[i][5] = relative_strength_index(i)
        indicators[i][6] = stochastic_oscillator(i)
    #reduce size of obv and accdl
    #normalize indicators...
    for i in range(7):
        if i == 2 or i == 4:
            continue
        indicators[:,i] /= np.max(indicators[:,i])
    indicators[:,2] = np.multiply(input_data[:,3],1/indicators[:,2])
    indicators[:,4] = np.multiply(input_data[:,3],1/indicators[:,4])

# index = 0
def on_balance_volume(i):
    #http://www.investopedia.com/terms/o/onbalancevolume.asp
    if i == 0:
        return input_data[i][4]
    if returns[i] == 0:
        return indicators[i-1][0]
    if returns[i] < 0:
        return indicators[i-1][0] - input_data[i][4]
    return indicators[i-1][0] + input_data[i][4]

# index = 1
def accumulation_distribution_line(i):
    #http://www.investopedia.com/terms/a/accumulationdistribution.asp
    money_flow_multiplier = ((input_data[i][3] - input_data[i][2]) - (input_data[i][1] - input_data[i][3])) / (input_data[i][1] - input_data[i][2])
    if  money_flow_multiplier != money_flow_multiplier:
        return 100
    if i == 0:
        return money_flow_multiplier

    money_flow_volume = money_flow_multiplier * input_data[i][4]
    return indicators[i-1][1] + money_flow_volume

# index = 2
def simple_moving_average(i):
    if i <= SMA_PERIOD:
        return input_data[i][3]
    return np.average(input_data[i - SMA_PERIOD + 1:i + 1, 3])

    #http://www.investopedia.com/terms/a/adx.asp

# index = 3
def aroon_indicator(i):
    cnt = 0
    j = i
    if returns[i] < 0:
        while returns[j]<0 and j>=0:
            j -= 1
            cnt += 1
    else:
        while returns[j]>0 and j>=0:
            j -= 1
            cnt += 1
    return ((AROON_PERIOD-cnt)/AROON_PERIOD)*100.0
    #http://www.investopedia.com/terms/i/indicator.asp

# index = 4
def exponential_moving_average(i):
    #http://www.investopedia.com/terms/e/ema.asp
    if i <= EMA_PERIOD:
        return input_data[i][3]
    return np.average(input_data[i - EMA_PERIOD + 1:i + 1, 3], weights=range(EMA_PERIOD, 0, -1), axis=0)


# index = 5
def relative_strength_index(i):
    #http://www.investopedia.com/terms/r/rsi.asp
    if i < RSI_PERIOD:
        return 50
    up = 0.0
    down = 0.0
    for j in range(i-RSI_PERIOD-1,i+1) :
        if returns[j] < 0:
            down += returns[j]
        else:
            up += returns[j]
    RS = (up/RSI_PERIOD)/(down/RSI_PERIOD)
    return 100.0 - 100/(1+RS)
# index = 6
def stochastic_oscillator(i):
    #http://www.investopedia.com/terms/s/stochasticoscillator.asp
    if i < STOCHASTIC_OSCH_PERIOD:
        return 50
    lowest = np.min(input_data[i - STOCHASTIC_OSCH_PERIOD + 1:i + 1, 3])
    highest = np.max(input_data[i - STOCHASTIC_OSCH_PERIOD + 1:i + 1, 3])

    return 100.0*(input_data[i][3] - lowest) / (highest - lowest)

def init_labels():
    global labels
    labels = np.zeros((len(returns),3))
    threshold = np.average(np.abs(returns))/2
    for i in range(len(returns)):
        index = 1
        if abs(returns[i]) >= threshold:
            if returns[i]<0:
                index = 0
            else:
                index = 2
        labels[i,index] = 1


    #Initializing data...
def init_data():
    global input_data, returns, indicators

    if os.path.exists('binary_data.npy'):
        input_data = np.load('binary_data.npy')
    else:
        input_data = load_data()
        np.save('binary_data', input_data)

    #Initializing the return rates...
    if os.path.exists('binary_returns.npy'):
        returns = np.load('binary_returns.npy')
    else:
        returns = load_returns()
        np.save('binary_returns', returns)

    #Initializing the indicators...
    if os.path.exists('binary_indicators.npy'):
        indicators = np.load('binary_indicators.npy')
    else:
        load_indicators()
        np.save('binary_indicators', indicators)

    #initializing the labels
    init_labels()

init_data()
