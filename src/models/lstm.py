#### Univariate LSTM with Hyperopt 
import numpy as np
import pandas as pd

# For preprocessing
from sklearn.metrics import mean_squared_error
# For LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from tensorflow import keras

# For hyperopt (parameter optimization)
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll.base import scope #quniform returns float, some parameters require int; use this to force int

import src.features.utils as utils
# for files organization
import logging
import os
import sys
import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--window_size", type=int, default=5)
parser.add_argument("-t", "--target", type=str, default='BTC')
parser.add_argument('-d', "--intra", action='store_false')
parser.add_argument('-r', "--tune", action='store_false')
arg = parser.parse_args()

# Experiment Setup
target = [arg.target]
window_size = arg.window_size
use_intra = arg.intra # by default = True if -d flag is not used

def get_dataframe(target, use_intra):
    if use_intra:
        btc_intraday = 'https://raw.githubusercontent.com/Giant316/crypto_scrapy/main/BTC_intraday.csv'
        eth_intraday = 'https://raw.githubusercontent.com/Giant316/crypto_scrapy/main/ETH_intraday.csv'
        xrp_intraday = 'https://raw.githubusercontent.com/Giant316/crypto_scrapy/main/XRP_intraday.csv'

        def load_intradata(fpath, ts_name):
            df = pd.read_csv(fpath, index_col=0).dropna().set_index(['time']).rename(columns={'close':ts_name})
            return df

        df_btc_intra = load_intradata(btc_intraday, "BTC")
        df_eth_intra = load_intradata(eth_intraday, "ETH")
        df_xrp_intra = load_intradata(xrp_intraday, "XRP")

        def rm_duplicate(df):
            return df.reset_index().loc[df.reset_index()[['time']].drop_duplicates().index].set_index(['time'])

        btc_intra = rm_duplicate(df_btc_intra[['BTC']])
        eth_intra = rm_duplicate(df_eth_intra[['ETH']])
        xrp_intra = rm_duplicate(df_xrp_intra[['XRP']])

        df = btc_intra.join(eth_intra, on='time').dropna().join(xrp_intra, on='time').fillna(method="ffill")
        #df_btc_intra[['BTC']].join(df_xrp_intra[['XRP']], on='time').loc[list(df_btc_intra[df_btc_intra['timestamp'].isin(list(set(df_btc_intra.timestamp.values) - set(df_xrp_intra.timestamp.values)))].index)]
        #df_btc_intra[['BTC']].join(df_xrp_intra[['XRP']], on='time')[df_btc_intra[['BTC']].join(df_xrp_intra[['XRP']], on='time').XRP.isnull()]

    else:
        btc_path = 'https://raw.githubusercontent.com/Giant316/crypto_scrapy/main/BTC_blockchain_info.csv'
        eth_path = 'https://raw.githubusercontent.com/Giant316/crypto_scrapy/main/ETH_blockchain_info.csv'
        xrp_path = 'https://raw.githubusercontent.com/Giant316/crypto_scrapy/main/XRP_blockchain_info.csv'

        def load_data(fpath, ts_name):
            df = pd.read_csv(fpath, encoding = 'utf-16', delimiter="\t", index_col=['Time'])
            if(ts_name == "XRP"):
                df = pd.concat([df.iloc[:,1:], df.iloc[:,[0]]], axis=1)
            df_features = ['x'+ str(x) for x in range(1, len(df.columns))]
            df.columns = [ts_name] + df_features
            df.index = pd.DatetimeIndex(df.index)
            return df

        df_btc = load_data(btc_path, "BTC")
        df_eth = load_data(eth_path, "ETH")
        df_xrp = load_data(xrp_path, "XRP")
        df = pd.concat([df_btc[['BTC']].dropna().loc['2016':'2020'], df_eth[['ETH']].dropna().loc['2016':'2020'], df_xrp[['XRP']].dropna().loc['2016':'2020']], axis=1)

    train_weight = 0.8
    split = int(len(df)*train_weight)
    df_train = df.iloc[:split]

    # Scale data 
    mu = np.float(df_train[target].mean())
    sigma = np.float(df_train[target].std())
    standardize = lambda x: (x - mu) / sigma
    #reverse_standardize = lambda x: x*sigma + mu

    df_train = df_train[target].apply(standardize)
    df_test = df[target].apply(standardize).iloc[split:]

    return pd.DataFrame(df_train), pd.DataFrame(df_test)

def read_log():
    tune_result_dir = os.path.join(proj_root, "reports", "lstm_tuning")
    if not Path(tune_result_dir).exists():
        sys.exit("Tuning Result folder does not exist. Exit Modelling!")

    log_files = os.listdir(tune_result_dir)
    log_files.sort()
    keys = ["target", "window_size", "batch_size", "units", "layers", "rate", "l1_reg"]
    hyper_dict = {}
    for idx, f in enumerate(log_files):
        fpath = os.path.join(tune_result_dir, f)
        log = open(fpath, 'r')
        lines = log.read().splitlines()
        hyperparam = [float(line.split(":")[-1]) for idx, line in enumerate(lines) if(idx != 0)]
        # convert the first 3 hyperparams to int (e.g: batch size, units, layers)
        hyperparam = list(map(int, hyperparam)) + hyperparam[-2:]
        exp_info = f.split(".")[0].split("_") # retrieve target & window_size info from filename
        exp_info = [exp_info[0]] + [int(exp_info[1])] # convert window_size to int
        result = exp_info + hyperparam
        hyper_dict[idx] = dict(zip(keys, result))
    return hyper_dict

def create_univariate_data(data, window_size):
    n = len(data)
    y = data[window_size:]
    data = data.values.reshape(-1, 1) # make 2D
    X = np.hstack(tuple([data[i: n-j, :] for i, j in enumerate(range(window_size, 0, -1))]))
    return pd.DataFrame(X, index=y.index), y

def format_data(df_train, df_test, window=30):
    Xtrain, ytrain = create_univariate_data(df_train, window_size=window)
    Xtrain = Xtrain.values.reshape(-1, window, 1)
    ytrain = ytrain.values.reshape(-1)
    
    Xtest, ytest = create_univariate_data(df_test, window_size=window)
    Xtest = Xtest.values.reshape(-1, window, 1)
    ytest = ytest.values.reshape(-1)

    return Xtrain, ytrain, Xtest, ytest

def lstm(Xtrain, ytrain, params):
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=15)
    model = Sequential()
    if params['layers'] == 1:
        model.add(LSTM(units=params['units'], activation='tanh', input_shape=(Xtrain.shape[1],Xtrain.shape[2]), kernel_regularizer=l1(params['l1_reg'])))
        model.add(Dropout(rate=params['rate']))
    else:
        # First layer specifies input_shape and returns sequences
        model.add(LSTM(units=params['units'], activation='tanh', return_sequences=True, input_shape=(Xtrain.shape[1],Xtrain.shape[2]), kernel_regularizer=l1(params['l1_reg'])))
        model.add(Dropout(rate=params['rate']))
        # Middle layers return sequences
        for i in range(params['layers']-2):
            model.add(LSTM(units=params['units'], activation='tanh', return_sequences=True, kernel_regularizer=l1(params['l1_reg'])))
            model.add(Dropout(rate=params['rate']))
        # Last layer doesn't return anything
        model.add(LSTM(units=params['units'], activation='tanh', kernel_regularizer=l1(params['l1_reg'])))
        model.add(Dropout(rate=params['rate']))

    model.add(Dense(1))
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')
    result = model.fit(Xtrain, ytrain, verbose=0, validation_split=0.1,
                    batch_size=params['batch_size'],
                    epochs=200,
                    callbacks = [es,TqdmCallback(verbose=1)]
                    )
    return model, result

def tune(params):
    #tscv = TimeSeriesSplit(n_splits = 3)
    btscv = utils.BlockingTimeSeriesSplit(n_splits=10, test_size=90)
    cv_train = []
    cv_test = []
    for train_index, test_index in btscv.split(df_train):
        cv_train.append(df_train.iloc[train_index])
        cv_test.append(df_train.iloc[test_index])
    
    rmse = []
    for train, test in zip(cv_train, cv_test):
        # Generate data for cross validation with given window    
        cv_Xtrain, cv_ytrain, cv_Xtest, cv_ytest = format_data(train, test, window=window_size)
        
        # LSTM model
        model, _ = lstm(cv_Xtrain, cv_ytrain)
        cv_predictions = model.predict(cv_Xtest).reshape(-1,)
        cv_trueValues = cv_ytest
        rmse.append(np.sqrt(mean_squared_error(cv_trueValues, cv_predictions)))
        
    return {'loss': np.mean(rmse), 'status': STATUS_OK, 'model': model, 'params': params}


df_train, df_test = get_dataframe(target, use_intra)
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

if(arg.tune): # default without -r flag 
    '''
        Tuning Model using Hyperopt with Cross Validation
    '''
    save_result_path =  Path(os.path.join(proj_root, "reports", "lstm_tuning"))
    if not save_result_path.exists():
        save_result_path.mkdir(parents=True)

    space = {'rate'       : hp.uniform('rate',0.01,0.5),
            'units'      : scope.int(hp.quniform('units',10,100,5)),
            'batch_size' : scope.int(hp.quniform('batch_size',100,250,25)),
            'layers'     : scope.int(hp.quniform('layers',1,3,1)),
            'l1_reg'     : hp.uniform('l1_reg',0.01,0.5)
            }

    trials = Trials()
    best = fmin(tune, space, algo=tpe.suggest, max_evals=50, trials=trials)

    best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
    best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']

    print("Hyperparameter Tuning Completed.")
    print(best)

    # log tuning results 
    # find the root directory of the project
    result_dir = os.path.join(str(save_result_path), "")
    logging.basicConfig(filename= f"{result_dir}{target[0]}_{window_size}.log", format='%(asctime)s %(message)s')
    logger = logging.getLogger() # create a logger object
    logger.setLevel(logging.INFO)
    logger.info("Values of Best parameters of %s with window size:%s", target[0], window_size)
    logger.info("Batch Size:%s", best['batch_size'])
    logger.info("Hidden Units:%s", best['units'])
    logger.info("Hidden Layers:%s", best['layers'])
    logger.info("Dropout Rate:%s", best['rate'])
    logger.info("L1 Regulizer:%s", best['l1_reg'])

else:
    save_model_path = Path(os.path.join(proj_root, "reports", "trained_models", "lstm"))
    if not save_model_path.exists():
        save_model_path.mkdir(parents=True) # set parents = True to create nested folder as well

    result_dict = read_log()
    for idx in result_dict:
        params = result_dict[idx]
        # retrieve saved hyperparameter 
        df_train, df_test = get_dataframe(params['target'], use_intra=True)
        Xtrain, ytrain, Xtest, ytest = format_data(df_train, df_test, window=params['window_size'])

        # train the model with the saved hyperparameter 
        model, result = lstm(Xtrain, ytrain, params)
        
        # save model
        model_name = str(params['target']) + "_" + str(params['window_size'])
        save_model_dir = os.path.join(str(save_model_path), "", model_name, "")  
        model.save(save_model_dir + "model")    
        
        # save dataset as numpy binary file 
        with open(f"{save_model_dir}dataset.npy", "wb") as file:
            np.save(file, Xtrain)
            np.save(file, ytrain)
            np.save(file, Xtest)
            np.save(file, ytest)

        # save model training history
        with open(f"{save_model_dir}history.txt", "w") as file:
            json.dump(result.history, file)
