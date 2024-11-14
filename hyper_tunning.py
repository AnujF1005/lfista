from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from configs.configLoader import getConfig
from train import train

config = getConfig()

config["alpha_initialize"] = tune.loguniform(1e-3, 1)
# config["beta"] = tune.loguniform(1e-4, 1)
# config["learning_rate"] = tune.loguniform(1e-4, 1e-1)
# config["symLambda"] = tune.loguniform(1e-4, 1e-1)
# config["batch_size"] = tune.choice([32, 64, 128, 256])
# config["fistaIter"] = tune.choice([20, 30, 40, 50])

config["databasePath"] = os.path.abspath(config["databasePath"])

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=config["epochs"],
    grace_period=1,
    reduction_factor=2
)

reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy", "training_iteration"]
)

result = tune.run(
    partial(train,isTune=True),
    resources_per_trial={"cpu": 32, "gpu": 2},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter
)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best checkpoint directory: {}".format(best_trial.checkpoint.value))