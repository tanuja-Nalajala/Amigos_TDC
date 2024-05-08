import os
import re
import sys

from injector import Injector

from decai.simulation.contract.classification.perceptron import PerceptronModule
from decai.simulation.data.imdb_data_loader import ImdbDataModule
from decai.simulation.logging_module import LoggingModule
from decai.simulation.simulate import Agent, Simulator

# For `bokeh serve`.
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

datasets = dict(
    
    imdb=dict(module=ImdbDataModule(num_words=1000),
              train_size=None, test_size=None,
              )
)

models = dict(
    perceptron=dict(module=PerceptronModule,
                    baseline_accuracy=dict(
                        # train_size, test_size = 3500, 1500
                        fitness=0.9507,
                        # train_size, test_size = None, None
                        imdb=0.73,
                        # train_size, test_size = None, None
                        news=0.9003,
                    ))
)

# Set up the agents that will act in the simulation.
agents = [
    # Good
    Agent(address="Good",
          start_balance=10_000,
          mean_deposit=50,
          stdev_deposit=10,
          mean_update_wait_s=10 * 60,
          prob_mistake=0.0001,
          ),
    # Malicious: A determined agent with the goal of disrupting others.
    Agent(address="Bad",
          start_balance=10_000,
          mean_deposit=100,
          stdev_deposit=3,
          mean_update_wait_s=1 * 60 * 60,
          good=False,
          ),
    # One that just calls the model and pays to use the model.
    Agent(address="Caller",
          start_balance=30_000,
          mean_deposit=0,
          stdev_deposit=0,
          mean_update_wait_s=2 * 60 * 60,
          calls_model=True,
          pay_to_call=50
          ),
]


def main():
    global agents

    # This file is set up to use different models and datasets.
    dataset = 'offensive'
    model_type = 'nb'

    assert dataset in datasets
    assert model_type in models

    train_size = datasets[dataset]['train_size']
    test_size = datasets[dataset]['test_size']
    if train_size is None:
        init_train_data_portion = 0.08
    else:
        init_train_data_portion = 100 / train_size

    # No caller (assume free to call).
    agents = agents[:-1]

    # Set up the data, model, and incentive mechanism.
    inj = Injector([
        DefaultCollaborativeTrainerModule,
        datasets[dataset]['module'],
        MurmurHash3Module,
        LoggingModule,
        models[model_type]['module'],
        StakeableImModule,
    ])
    s = inj.get(Simulator)

    # Start the simulation.
    s.simulate(agents,
               baseline_accuracy=models[model_type]['baseline_accuracy'].get(dataset),
               init_train_data_portion=init_train_data_portion,
               train_size=train_size,
               test_size=test_size,
               filename_indicator=f"{dataset}-{model_type}"
               )


# Run with `bokeh serve PATH`.
if re.match('bk_script_|bokeh_app_', __name__):
    main()
else:
    print("`__name__` didn't match the pattern. Bokeh app will not run.")
