from ray.tune.suggest.variant_generator import parse_spec_vars
import src.features.utils as utils
import pandas as pd
import tensorflow as tf
from ray.tune.integration.keras import TuneReportCallback
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import logging 
import configparser
import os

curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_config(self, no=0):
    config = configparser.ConfigParser()
    
    # need to store the absolute path of the config file as the module is loaded -> config != directory as this script
    # use dirname twice to repeatedly "climb higher" up to the directory, config.ini does not reside on the same level as this script
    config_path = os.path.join(curr_dir, "config.ini")
    
    config.read(config_path)
    sections = config.sections()
    exp = sections[no]
    dat = config.get(sections[no], 'data') 
    inw = config.getint(sections[no], 'input_width')
    lpw = config.getint(sections[no], 'label_width')
    ofs = config.getint(sections[no], 'offset')

    return exp, dat, inw, lpw, ofs

def run(run_name, window_size):
    ray.init(num_cpus=4)# if args.smoke_test else None)
    sched = AsyncHyperBandScheduler(time_attr="training_iteration", max_t=400, grace_period=20)

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
            "window": win6
        })
    ray.shutdown()

    return analysis

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
                
    model.fit(config["window"].train, epochs=5, validation_data=config["window"].val,
    callbacks=[TuneReportCallback({"MAPE": "mean_absolute_percentage_error"
    })])

if __name__ == "__main__":
    # process configuration file
    exp, dat, ipw, lbw, ofs = read_config(1) # get second experiment's config
    
    # retrieve file path
    csv_path = os.path.join(os.path.dirname(curr_dir), dat)
    df = pd.read_csv(csv_path)

    # univariate time series 
    df = df[['Close']]

    # a window with number of time step of input = 6 
    win6 = utils.DataGenerator(
        input_width=6, label_width=1, offset=1, 
        label_columns=df.columns.to_list(), df=df)

    exp_name = "Tune_MLP"
    analysis = run(run_name=exp_name, window_size=win6)  

    # log tuning results  
    logging.basicConfig(filename= f"./reports/{exp_name}.log", format='%(asctime)s %(message)s')
    logger = logging.getLogger() # create a logger object
    logger.setLevel(logging.INFO)   
    logger.info("%s with configuration of Input Width: %s", exp, ipw)
    logger.info("results save in folder: %s", exp_name)    
    logger.info("Best hyperparameters: Hidden: %s; Activation: %s", analysis.best_config["hidden"], analysis.best_config['activation'])
    print("Best hyperparameters found were: Hidden: %s; Activation: %s", analysis.best_config["hidden"], analysis.best_config['activation'])
    