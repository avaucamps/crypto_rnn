import gc
import datetime
import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from collections import deque
import random


CLOSE_COLUMN = "Close**"
VOLUME_COLUMN = "Volume"
FUTURE_COLUMN = "Future"
FUTURE_PERIOD_PREDICT = 1
TARGET_COLUMN = "Target"
SEQUENCE_LENGTH = 3
DATE_COLUMN = "Date"


def get_market_data(market):
    """
    Fetches information of the coin name passed in parameter.

    Arguments:
        market: the full name of the cryptocurrency as spelled on coinmarketcap.com. eg.: 'bitcoin'
    
    Returns:
        a dataframe containing the information of the cpin.
    """
    coin_url = str("https://coinmarketcap.com/currencies/{}/historical-data/?start=20130428&end={}") \
        .format(market, time.strftime("%Y%m%d"))
    coin_data = pd.read_html(coin_url, flavor='html5lib')[0]
    coin_data = coin_data.assign(Date=pd.to_datetime(coin_data['Date']))
    coin_data["Volume"] = (pd.to_numeric(coin_data[VOLUME_COLUMN], errors='coerce'))
    coin_data = coin_data[np.isfinite(coin_data[VOLUME_COLUMN])]

    return coin_data


def drop_unnecessary_columns(dataframe):
    dataframe = dataframe[['Date']+[metric for metric in [CLOSE_COLUMN,VOLUME_COLUMN]]]
    return dataframe


def classify(current, future):
    if float(future) < float(current):
        return 1
    else:
        return 0


def add_new_columns(data):
    data[FUTURE_COLUMN] = data[CLOSE_COLUMN].shift(-FUTURE_PERIOD_PREDICT)
    data[TARGET_COLUMN] = list(map(classify, data[CLOSE_COLUMN], data[FUTURE_COLUMN]))
    return data


def split_train_validation(data):
    last_5pct = int(len(data) * 0.05)
    validation_data = data[(data.index < last_5pct)]
    train_data = data[(data.index >= last_5pct)]
    return train_data, validation_data


def normalize_data(data):
    for column in data.columns:
        if column != TARGET_COLUMN and column != DATE_COLUMN:
            data[column] = data[column].pct_change()
            data.dropna(inplace=True)
            data[column] = preprocessing.scale(data[column].values)

    data.dropna(inplace=True)
    return data


def create_sequential_data(data):
    sequential_data = []
    values = deque(maxlen=SEQUENCE_LENGTH)
    for i in data.values:
        values.append([e for e in i[1:-1]])
        if len(values) == SEQUENCE_LENGTH:
            sequential_data.append([np.array(values), i[-1]])

    return sequential_data


def balance_samples(data):
    buys = []
    sells = []
    for sequence, target in data:
        if target == 0:
            sells.append([sequence, target])
        elif target == 1:
            buys.append([sequence, target])

    smallest_list_len = min(len(sells), len(buys))
    buys = buys[:smallest_list_len]
    sells = sells[:smallest_list_len]

    data = buys + sells
    random.shuffle(data)
    return data


def get_inputs_outputs(data):
    X = []
    y = []
    for sequence, target in data:
        X.append(sequence)
        y.append(target)

    return np.array(X), y


def get_train_validation(data):
    """
    Splits data between training data and validation data.

    Arguments:
        data: dataframe containing the data.

    Returns:
        two lists, train_data and validation_data.
    """
    data = drop_unnecessary_columns(data)
    data = add_new_columns(data)
    train_data, validation_data = split_train_validation(data)
    return train_data, validation_data


def preprocess_data(data):
    """
    Preprocesses the data to be fed in the neural network.

    Arguments:
        data: data to be used to train for the neural network.

    Returns:
        inputs as a numpy array.
        labels as a list.
    """
    data = data.drop(FUTURE_COLUMN, 1)
    data = normalize_data(data)
    sequential_data = create_sequential_data(data)
    sequential_data = balance_samples(sequential_data)
    
    return get_inputs_outputs(sequential_data)