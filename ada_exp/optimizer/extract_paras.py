import numpy as np
import panda as pd
import os
import time
import torch
import scipy2 as sci2

def read_models(model_path):
    df = pd.Dataframe()

    for sub_path in model_path:

        df = pd.series(model_path, df)

    return df
