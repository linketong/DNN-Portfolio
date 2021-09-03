import tensorflow as tf
import numpy as np
import pandas as pd
from math import isnan
import sys
import os
from sklearn.model_selection import train_test_split
from config import get_config
from equation import get_equation
from solver import Merged
import matplotlib.pyplot as plt

problem_name = "Heston"
config = get_config(problem_name)
config.total_time = 2
config.num_time_interval = 100
equation = get_equation(problem_name + "_poly", config)
trained_model = Merged(equation, y0=1.0)
trained_model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss="mean_squared_error")
total_sample_size = 10  # batch_size * n_epochs + test_size
# test_size_ratio = test_size / total_sample_size
np.random.seed(1)
inputs = np.random.normal(0, equation.sqrt_delta_t, (total_sample_size, equation.dim * equation.num_time_interval))
outcome = np.zeros([total_sample_size, 1])
trained_model.train_step(inputs, outcome, 1)
# trained_model.load_weights('./logs/model/Heston_T=2_X=0.25')
