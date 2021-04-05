import src.features.utils as utils
import pandas as pd
import tensorflow as tf
from ray.tune.integration.keras import TuneReportCallback
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import logging 
import configparser

def train_mlp(config): 
    model = tf.keras.Sequential([
      tf.keras.layers.Flatten(), # convert to 1D: (time, features) => (time*features) 
      tf.keras.layers.Dense(config["hidden"], config['activation']),
      tf.keras.layers.Dense(units=1),
      tf.keras.layers.Reshape([1, -1]),
      ])
    
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=["mean_absolute_percentage_error"]) #tf.metrics.MeanAbsoluteError()]) #, tf.metrics.MeanAbsolutePercentageError(), tf.metrics.RootMeanSquaredError()])
                
    model.fit(window.train, epochs=5, validation_data=window.val,
    callbacks=[TuneReportCallback({"MAPE": "mean_absolute_percentage_error"
    })])

def read_config(no=0):
    config = configparser.ConfigParser()
    config.read('config.ini')
    sections = config.sections()
    exp = sections[no]
    dat = config[sections[no]]['data']
    inw = config[sections[no]]['input_width']
    lpw = config[sections[no]]['label_width']
    ofs = config[sections[no]]['offset']
    return exp, dat, inw, lpw, ofs

def tune(exp_name, window_size):
    ray.init(num_cpus=4)# if args.smoke_test else None)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20)

    analysis = tune.run(
        train_mlp,
        name=exp_name,
        scheduler=sched,
        metric="MAPE",
        mode="max",
        stop={
            "MAPE": 5,
            "training_iteration": 5
        },
        num_samples=10,
        resources_per_trial={
            "cpu": 2,
            "gpu": 0
        },
        config={
            "hidden": tune.grid_search([1, 2]),
            "activation": tune.choice(['relu', 'linear']),
            "window": window_size
        })
    return analysis

if __name__ == "__main__":
    # process configuration file
    exp, dat, ipw, lbw, ofs = read_config(1) # get second experiment's config
    df = pd.read_csv(dat)

    # univariate time series 
    df = df[['Close']]

    # a window with number of time step of input = 6 
    win6 = utils.DataGenerator(
        input_width=ipw, label_width=lbw, offset=ofs, 
        label_columns=df.columns.to_list(), df=df)

    exp_name = "Tune_MLP"
    analysis = tune(exp_name=exp_name, window_size=win6)

    # log tuning results  
    logging.basicConfig(filename= f"./reports/{exp_name}.log", format='%(asctime)s %(message)s')
    logger = logging.getLogger() # create a logger object
    logger.setLevel(logging.INFO)   
    logger.info("%s with configuration of Input Width: %s", exp, ipw)
    logger.info("results save in folder: %s", exp_name)    
    logger.info("Best hyperparameters: Hidden: %s; Activation: %s", analysis.best_config["hidden"], analysis.best_config['activation'])

    ray.shutdown()