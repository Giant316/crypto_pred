#### Univariate LSTM with Hyperopt 
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping

# For hyperopt (parameter optimization)
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll.base import scope #quniform returns float, some parameters require int; use this to force int
from tqdm.keras import TqdmCallback

import src.features.utils as utils
import argparse
import os, json
from pathlib import Path 
import random

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--window_size", type=int, required=True)
parser.add_argument("-t", "--target", type=str, required=True)
parser.add_argument("-x", "--intra", action="store_false")
arg = parser.parse_args()

# find the root directory of the project
proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

# Experiment Setup
target = [arg.target]
window_size = arg.window_size
use_intra = arg.intra
test_size = 60

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
# df_test = df.iloc[split:]

# # Scale data 
mu = np.float(df_train[target].mean())
sigma = np.float(df_train[target].std())
standardize = lambda x: (x - mu) / sigma
reverse_standardize = lambda x: x*sigma + mu

df_train = df_train[target].apply(standardize)
df_test = df[target].apply(standardize).iloc[split:]

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

space = {'rate'       : hp.uniform('rate',0.01,0.5),
         'units'      : scope.int(hp.quniform('units',10,100,5)),
         'batch_size' : scope.int(hp.quniform('batch_size',100,250,25)),
         'l1_reg'     : hp.uniform('l1_reg',0.01,0.5),
         'activation' : hp.choice('activation', ['sigmoid', 'relu', 'tanh'])
        }

save_result_path = Path(os.path.join(proj_root, "reports", "crossval", "mlp", target[0] + "_" + str(window_size)))
if not save_result_path.exists():
    save_result_path.mkdir(parents=True) 
save_result_dir = os.path.join(str(save_result_path), "")

def tune(params):

    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=15)

    btscv = utils.BlockingTimeSeriesSplit(n_splits=10, test_size=window_size + test_size)
    #btscv = BlockingTimeSeriesSplit(n_splits=2, test_size=30)
    cv_train = []
    cv_test = []
    for train_index, test_index in btscv.split(df_train):
        cv_train.append(df_train.iloc[train_index])
        cv_test.append(df_train.iloc[test_index])
    
    rmse = []
    res = {}
    for idx, (train, test) in enumerate(zip(cv_train, cv_test)):
        # Generate data for cross validation with given window
        cv_Xtrain, cv_ytrain, cv_Xtest, cv_ytest = format_data(train, test, window=window_size)
        
        # MLP model
        model = Sequential()
        model.add(Dense(units=params['units'], activation=params['activation'], input_dim=window_size, kernel_regularizer=l1(params['l1_reg'])))
        model.add(Dropout(rate=params['rate']))
        model.add(Dense(1))

        #optimizer = keras.optimizers.Adam(lr=0.001, epsilon=1e-08)
        model.compile(optimizer='Adam', loss='mean_squared_error')

        # reshape the input for MLP input (batch, window_size)
        cv_Xtrain = cv_Xtrain.reshape(cv_Xtrain.shape[0:2])
        cv_ytrain = cv_ytrain.reshape(cv_ytrain.shape[0:2])
        cv_Xtest = cv_Xtest.reshape(cv_Xtest.shape[0:2])
        cv_ytest = cv_ytest.reshape(cv_ytest.shape[0:2])
        
        result = model.fit(cv_Xtrain, cv_ytrain, verbose=0, validation_split=0.1,
                       batch_size=params['batch_size'],
                       epochs=200,
                       callbacks = [es]
                      )

        cv_predictions = model.predict(cv_Xtest).reshape(-1,)
        cv_trueValues = cv_ytest
        res[idx] = {"preds": cv_predictions.tolist(), "true": cv_trueValues.tolist(), "params": params, "history": result.history, "rmse":rmse}
        rmse.append(np.sqrt(mean_squared_error(cv_trueValues, cv_predictions)))
    num = str(random.random())[2:]
    with open(f"{save_result_dir}{num}.txt", "w") as file:
        json.dump(res, file)  
    return {'loss': np.mean(rmse), 'status': STATUS_OK, 'model': model, 'params': params}

trials = Trials()
best = fmin(tune, space, algo=tpe.suggest, max_evals=100, trials=trials)

best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']

best_model.save(f"{save_result_dir}model")
with open(f"{save_result_dir}best_params.txt", "w") as file:
    json.dump(best_params, file)

print("Hyperparameter Tuning Completed.")
