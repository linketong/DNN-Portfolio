import numpy as np


class Config:
    def __init__(self, d=1, horizon=2, steps=20, gamma=2, psi=0.125, x0=0.25) -> None:
        self.dim = d
        self.total_time = horizon
        self.num_time_interval_per_year = steps
        self.num_time_interval = self.total_time * self.num_time_interval_per_year
        self.delta = 0.05
        self.gamma = gamma
        self.psi = psi
        self.x_init = x0
