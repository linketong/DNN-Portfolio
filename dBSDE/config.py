import numpy as np


class Config(object):
    batch_size = 128
    valid_size = 512
    logging_frequency = 100
    verbose = True
    y_init_range = [0, 1]
    min_decrease = 0.05
    init_learning_rate = 0.02


class HestonConfig(Config):
    # lr_boundaries = [200,800,1500]
    # lr_values = list(np.array([2e-2, 1e-2, 5e-3, 5e-4]))
    # num_hiddens = [dim, dim + 10, dim + 10, dim]
    # y_init_range = [0, 0]
    dim = 1
    total_time = 2
    num_time_interval = 20
    num_iterations = 2000
    y_init_value = 0
    gamma = 2
    psi = 0.125
    x_init = 1.0
    normalization = True
    handle_init_grad = True
    combine_sigmax = True


class NewEconConfig(Config):
    # lr_values = list(np.array([5e-2, 2e-2, 1e-2, 5e-4]))
    # y_init_range = [0, 0]
    # y_init_value = 0
    dim = 3
    total_time = 2
    num_time_interval = 20
    lr_boundaries = [300, 600, 1000]
    num_iterations = 2000
    normalization = True
    handle_init_grad = False
    combine_sigmax = True

    delta = 0.05
    gamma = 4.0
    psi = 0.25

    # param for alpha(x)
    kappa_r = 0.00034
    r_bar = 0.00520 * 12
    phi_r = 17224.987 / 12
    # eta_r = 0.4116
    eta_r = 0.5
    kappa_p = 0.005
    p_bar = 0.0332 * 12
    phi_p = 0
    eta_p = 0
    theta_l = np.array([1.5, 1.5, 2.5])
    theta_u = np.array([1.5, 1.5, 2.5])
    kappa_theta = np.array([0.1219, 0.5946, 0.7483])
    theta_bar = np.array([0.1562, 0.4251, 0.1669])
    delta_r = np.array([-2.7648 / 12.0, 139.0 / 12.0, 176.76 / 12.0])
    delta_p = np.array([-0.0372, 14.04, -4.44])

    # param for sigma_x(x)
    gamma_r = 0.5664
    sigma_r = [0.00986 * 12 ** (1 - gamma_r), 0, 0]
    gamma_p = 0.5
    sigma_p = np.array([-0.00103 * 12 ** 0.5, 0.01011 * 12 ** 0.5, 0.003 * 12 ** 0.5])
    sigma_theta = np.array([[-0.2032, 0.00497, 0.0356], [-0.1237, -0.0306, 0.0406], [-0.1199, -0.493, 0.402]])
    gamma1_theta = np.array([0.5, 0.5, 0.5])
    gamma2_theta = np.array([1.2158, 0.5095, 0.6440])

    x0 = np.array([0.8, 1.03, 0.65, 0.626, 1.0])
    x_init = x0 * np.array([r_bar, p_bar, theta_bar[0], theta_bar[1], theta_bar[2]])
    # x_init = np.array([r_bar, p_bar, theta_bar[0], theta_bar[1], theta_bar[2]])


class LargeScaleConfig(Config):
    dim = 10
    total_time = 2
    num_time_interval = 20
    delta = 0.05
    gamma = 4.0
    psi = 0.25
    # r_bar = 0.00407 * 12
    # theta1_bar = 0.048786
    # theta_bar = 0.119
    # p_bar = 0.0032 * 12
    x0 = np.array([1.2] * dim * 2)
    # x_init = x0 * np.array([r_bar, theta1_bar] + [theta_bar] * (dim - 1) + [p_bar] * (dim - 1))


def get_config(name):
    try:
        return globals()[name + "Config"]
    except KeyError:
        raise KeyError("Config for the required problem not found.")
