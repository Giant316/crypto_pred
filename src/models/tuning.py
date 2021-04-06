from ray.tune import experiment
from ray.tune.suggest.variant_generator import parse_spec_vars
from ray.tune.tune import run_experiments
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
import argparse

# need to store the absolute path of the config file as the module is loaded -> config != directory as this script
# use dirname twice to repeatedly "climb higher" up to the directory, config.ini does not reside on the same level as this script
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # file path of src directory
config_file = "mlp.ini"

def read_config(num=0):
    # process the configuration file of the experiment
    config = configparser.ConfigParser()
    config_path = os.path.join(src_dir, config_file)
    
    config.read(config_path)
    sections = config.sections()
    exp = sections[num]
    dat = config.get(sections[num], 'data') 
    inw = config.getint(sections[num], 'input_width')
    lpw = config.getint(sections[num], 'label_width')
    ofs = config.getint(sections[num], 'offset')

    return exp, dat, inw, lpw, ofs

def train_mlp(config): 
    tf.debugging.set_log_device_placement(True)

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

def run_tune(exp_name, window_size):
    ray.init(num_gpus=2)
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
            "cpu": 0,
            "gpu": 2
        },
        config={
            "hidden": tune.grid_search([1, 2]),
            "activation": tune.choice(['relu', 'linear']),
            "window": window_size
        })
    ray.shutdown()

    return analysis

def run_experiment(exp_num=0):
    exp, dat, ipw, lbw, ofs = read_config(exp_num) # get second experiment's config
    
    # retrieve file path
    csv_path = os.path.join(os.path.dirname(src_dir), dat) # climb higher to crypto_predict dir
    df = pd.read_csv(csv_path)

    # univariate time series 
    df = df[['Close']]

    # a window with number of time step of input = 6 
    win = utils.DataGenerator(
        input_width=ipw, label_width=lbw, offset=ofs, 
        label_columns=df.columns.to_list(), df=df)

    analysis = run_tune(exp_name=exp, window_size=win) 

    # log tuning results  
    logging.basicConfig(filename= f"./reports/{exp}.log", format='%(asctime)s %(message)s')
    logger = logging.getLogger() # create a logger object
    logger.setLevel(logging.INFO)   
    logger.info("%s with configuration of Input Width: %s", exp, ipw)
    logger.info("results save in folder: %s", exp)    
    logger.info("Best hyperparameters: Hidden: %s; Activation: %s", analysis.best_config["hidden"], analysis.best_config['activation'])
    print("Best hyperparameters found were: Hidden: {}; Activation: {}".format(analysis.best_config["hidden"], analysis.best_config['activation']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nExp', action='store', type=int, required=True)
    args = parser.parse_args()

    # Execute tuning runs
    for n in range(args.nExp):
        run_experiment(n)

    

    
    
