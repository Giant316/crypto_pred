import src.features.utils as utils
import pandas as pd
import tensorflow as tf
from ray.tune.integration.keras import TuneReportCallback
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import logging 

def train_mlp(config):
    url = 'https://raw.githubusercontent.com/Giant316/crypto_scrapy/main/BTC.csv'
    df = pd.read_csv(url)

    # univariate time series 
    df = df[['Close']]

    # a window with number of time step of input = 6 
    window = utils.DataGenerator(
        input_width=6, label_width=1, offset=1, 
        label_columns=df.columns.to_list(), df=df)
        
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

if __name__ == "__main__":
    exp_name = "Tune_MLP"
    logging.basicConfig(filename= f"./reports/{exp_name}.log", format='%(asctime)s %(message)s')
    logger = logging.getLogger() # create a logger object
    logger.setLevel(logging.INFO)

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
            "activation": tune.choice(['relu', 'linear'])
        })
    logger.info("Best hyperparameters found were: %s", analysis.best_config)

    ray.shutdown()