import pickle
from math import isnan

import numpy as np

import tensorflow as tf

from scipy.integrate import quad
from scipy.stats import multivariate_normal as normal
from sklearn.model_selection import train_test_split

TF_DTYPE = tf.float32


class Equation(object):
    """Base class for defining the FBSDE related functions."""

    def __init__(self, config):
        self.dim = config.dim
        self.total_time = config.total_time
        self.num_time_interval = config.num_time_interval
        self.delta_t = (self.total_time + 0.0) / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)

    def next_x(self, num_sample):
        """simulate state variables"""
        raise NotImplementedError

    def next_y(self, num_sample):
        """simulate value function"""
        raise NotImplementedError

    def z_T_matmul_sigma_x(self, z, sigma_x):
        """Calulate the matrix multiplication between Z and beta(x)"""
        return tf.squeeze(tf.matmul(tf.expand_dims(z, 0), sigma_x))

    def f_tf(self, t, x, y, z):
        """Generator function in the BSDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the BSDE."""
        raise NotImplementedError


def get_equation(name, config):
    try:
        return globals()[name](config)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")


class Heston(Equation):
    def __init__(self, config):
        super(Heston, self).__init__(config)
        self.x_init = config.x_init
        self.delta = config.delta
        self.r = 0.05
        self.kappa = 5.0
        self.var_theta = self.kappa * 0.15 ** 2
        self.beta_bar = 0.25
        self.mu_bar = 3.11
        self.rho = -0.5
        self.gamma = config.gamma
        self.psi = config.psi
        self.k = self.gamma / (self.gamma + (1 - self.gamma) * self.rho ** 2)
        self.theta = (1 - self.gamma) / (1 - 1 / self.psi)
        self.q_tilde = 1 - self.psi * self.k / self.theta
        self.dimx = self.dim

    @tf.function
    def next_x(self, x, dw):
        new_kappa = -(-self.kappa + ((1 - self.gamma) / self.gamma) * self.beta_bar * self.mu_bar * self.rho)
        new_theta = self.var_theta / new_kappa
        sigmax = self.sigma_x(x)
        dw = tf.expand_dims(dw, axis=2)
        x_new = x + new_kappa * (new_theta - x) * self.delta_t + tf.squeeze(tf.matmul(sigmax, dw), axis=2)
        x_new = tf.maximum(x_new, 0)
        return x_new

    def sigma_x(self, x):
        return tf.expand_dims(self.beta_bar * tf.sqrt(x), axis=1)

    @tf.function
    def next_y(self, t, x, y, z, dw, lb, ub, zdx=True):
        z = tf.reshape(self.sigma_x(x) @ tf.expand_dims(z, 2), [-1, 1])
        vol = tf.sqrt(tf.maximum(x, 0.0001))
        mpr = self.mu_bar * vol
        optimal_pi_sigma = (mpr + self.rho * z) / self.gamma

        pi_sigma = optimal_pi_sigma
        y_new = y - self.delta_t * self.f_u(x, y, z) + tf.reduce_sum(tf.multiply(z, dw), 1, keepdims=True)
        pi = pi_sigma / vol
        return y_new, pi

    def f_u(self, x, y, z):
        r_tilde = -(1 / self.k) * (
            self.r * (1 - self.gamma)
            + 0.5 * self.mu_bar ** 2 * x * (1 - self.gamma) / self.gamma
            - self.delta * self.theta
        )
        f = -r_tilde * y + (self.delta ** self.psi / (1 - self.q_tilde)) * tf.pow(tf.maximum(y, 0), self.q_tilde)
        return f

    def g_tf(self, t, x):
        if self.psi == 0.125 and self.gamma == 2:
            return tf.zeros(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)
        return tf.ones(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)

    def a_b(self, y, t, s):
        """
        Auxilary function used in calculating closed form solution
        """
        gamma = self.gamma
        delta = self.delta
        rho = self.rho
        r = self.r
        mu_bar = self.mu_bar
        kappa = self.kappa
        var_theta = self.var_theta
        beta_bar = self.beta_bar
        theta = self.theta
        k = self.k
        kappa_hat = kappa - ((1 - gamma) / gamma) * mu_bar * beta_bar * rho
        b = -0.5 * ((1 - gamma) / (k * gamma)) * mu_bar ** 2
        a = np.sqrt(kappa_hat ** 2 + 2 * b * beta_bar ** 2)
        a_exact = 4 * b * var_theta * (
            np.log((kappa_hat + a) * np.exp(a * s) - (kappa_hat - a) * np.exp(a * t))
            - np.log(2 * a)
            - 0.5 * ((kappa_hat + a) * s - (kappa_hat - a) * t)
        ) / (kappa_hat ** 2 - a ** 2) + (((1 - gamma) * r - delta * theta) / k) * (s - t)
        b_exact = 2 * b * (np.exp(a * (s - t)) - 1) / (np.exp(a * (s - t)) * (kappa_hat + a) - kappa_hat + a)
        return a_exact, b_exact

    def h_exact(self, x, t, T):
        """
        Auxilary function used in calculating closed form solution
        """

        def integrand(s):
            return np.exp(self.a_b(x, t, s)[0] - self.a_b(x, t, s)[1] * x)

        return self.delta ** self.psi * quad(integrand, t, T)[0]

    def hx_exact(self, x, t, T):
        """
        Auxilary function used in calculating closed form solution
        """

        def integrand(s):
            return self.a_b(x, t, s)[1] * np.exp(self.a_b(x, t, s)[0] - self.a_b(x, t, s)[1] * x)

        return -self.delta ** self.psi * quad(integrand, t, T)[0]


class LargeScale(Equation):
    def __init__(self, config):
        super(LargeScale, self).__init__(config)
        self.config = config
        self.delta = tf.constant(config.delta, TF_DTYPE)

        self.gamma = tf.constant(config.gamma, TF_DTYPE)
        self.psi = tf.constant(config.psi, TF_DTYPE)
        self.k = self.gamma
        self.theta = (1 - self.gamma) / (1 - 1 / self.psi)
        self.q_tilde = 1 - self.psi * self.k / self.theta
        self.dim = config.dim
        self.dimx = 2 * config.dim
        self.d_th = self.delta * self.theta
        self.coef_exp = (self.theta / self.psi) * self.delta ** self.psi

        # param of IR drift
        self.kappa_r = tf.constant(0.007185, TF_DTYPE)
        self.r_bar = tf.constant(0.00407 * 12, TF_DTYPE)
        self.phi_r = tf.constant(54.45 / 12, TF_DTYPE)
        # eta_r = 0.4116
        self.eta_r = tf.constant(0.3601, TF_DTYPE)
        # param of dividends drift
        self.kappa_p = tf.constant(0.02, TF_DTYPE)
        self.p_bar = tf.constant(0.0032 * 12, TF_DTYPE)
        self.phi_p = tf.constant(20.93 / 12 ** (2 * 0.244), TF_DTYPE)
        self.eta_p = tf.constant(0.244, TF_DTYPE)
        # param of MPR_1 drift
        self.theta_l = tf.constant(1.5, TF_DTYPE)
        self.theta_u = tf.constant(1.5, TF_DTYPE)
        self.kappa_theta1 = tf.constant(0.85576, TF_DTYPE)
        self.theta1_bar = tf.constant(0.048786, TF_DTYPE)
        self.delta_r1 = tf.constant(3.0708, TF_DTYPE)
        self.kappa_theta = tf.constant(0.989, TF_DTYPE)
        self.theta_bar = tf.constant(0.119, TF_DTYPE)
        self.delta_r = tf.constant(32.47 / 12, TF_DTYPE)
        self.delta_p = tf.constant(-17.46 / 12, TF_DTYPE)

        # param of IR vol
        self.gamma_r = 0.716
        self.sigma_r = tf.constant(0.0146 * 12 ** (1 - self.gamma_r), TF_DTYPE)
        # param of dividends vol
        self.gamma_p = 0.6315
        self.sigma_p = tf.constant(0.005 * 12 ** (1 - self.gamma_p), TF_DTYPE)
        # param of MPR_1
        self.sigma_theta1 = tf.constant(2.9417, TF_DTYPE)
        self.gamma1_theta1 = tf.constant(0.5, TF_DTYPE)
        self.gamma2_theta1 = tf.constant(2.8313, TF_DTYPE)
        # param of MPR_i
        self.sigma_theta = tf.constant(
            0.1626,
            TF_DTYPE,
        )
        self.gamma1_theta = tf.constant(0.5, TF_DTYPE)
        self.gamma2_theta = tf.constant(0.659, TF_DTYPE)

        x0 = np.array([config.x_init] * self.dimx)
        self.x_init = tf.cast(x0, dtype=TF_DTYPE) * tf.stack(
            [self.r_bar] + [self.p_bar] * (self.dim - 1) + [self.theta1_bar] + [self.theta_bar] * (self.dim - 1)
        )

        self.sigma_b = 0.156
        self.sigma_1 = 0.106
        self.sigma_i = 0.229
        self.rho_0 = 0.37

    def multiply_with_vol_T(self, x):
        """
        Multiply vector x with the volatility matrix of stock price
        parameter x: of dimension [M,d]
        output : of dimension [M,d]
        """
        vol_T_0_1 = np.asarray(
            [
                (1 / self.sigma_b, 0.0),
                (
                    -self.sigma_1 * self.rho_0 / (self.sigma_1 * np.sqrt(1 - self.rho_0 ** 2) * self.sigma_b),
                    1 / (self.sigma_1 * np.sqrt(1 - self.rho_0 ** 2)),
                ),
            ]
        )
        vol_T_0_1 = tf.cast(tf.stack(vol_T_0_1), dtype=TF_DTYPE)
        b_0_1 = tf.squeeze(tf.matmul(vol_T_0_1, tf.expand_dims(x[:, :2], 2)))
        b_2_10 = (1 / self.sigma_i) * x[:, 2:]
        return tf.concat([b_0_1, b_2_10], 1)

    def z_T_matmul_sigma_x(self, z, sigma_x):
        temp = z * sigma_x
        return temp[:, : self.dim] + temp[:, self.dim :]

    def sigma_x(self, x):
        """
        x:  x[0]   r
            x[1:d-1] p
            x[d]  theta1
            x[d+1:2d-1] theta
        output: beta(x) the volatility of the state variable of dimension [M x 2d]
        """
        r = tf.expand_dims(x[:, 0], 1)  # [M,1]
        p = x[:, 1 : self.dim]  # [M,d-1]
        theta1 = tf.expand_dims(x[:, self.dim], 1)  # [M,1]
        theta = x[:, self.dim + 1 : self.dim + self.dim]  # [M,d-1]
        sigma_r = -self.sigma_r * tf.maximum(r, 0.0) ** self.gamma_r
        sigma_p = -self.sigma_p * tf.maximum(p, 0.0) ** self.gamma_p
        sigma_theta1 = (
            self.sigma_theta1 * (1.5 + theta1) ** 0.5 * (1 - ((1.5 + theta1) / 3.0) ** 0.5) ** self.gamma2_theta1
        )
        sigma_theta = self.sigma_theta * (1.5 + theta) ** 0.5 * (1 - ((1.5 + theta) / 3.0) ** 0.5) ** self.gamma2_theta
        sigma_x = tf.concat([sigma_r, sigma_p, sigma_theta1, sigma_theta], axis=1)
        return sigma_x

    def alpha_x(self, x):
        """
        x:  x[0]   r
            x[1:d-1] p
            x[d]  theta1
            x[d+1:2d-1] theta
        output: the drift of the state variable M x 2d
        """
        r = tf.expand_dims(x[:, 0], 1)  # [M,1]
        p = x[:, 1 : self.dim]  # [M,d-1]
        diff_r = self.r_bar - r
        diff_p = self.p_bar - p
        theta1 = tf.expand_dims(x[:, self.dim], 1)  # [M,1]
        theta = x[:, self.dim + 1 : self.dim + self.dim]  # [M,d-1]
        alpha_r = self.kappa_r * diff_r * (1 + self.phi_r * (diff_r ** 2) * self.eta_r)
        alpha_p = self.kappa_p * diff_p * (1 + self.phi_p * (diff_p ** 2) * self.eta_p)
        alpha_theta1_r = self.delta_r * diff_r * (1.5 + theta1) * (1 - (1.5 + theta1) / 3.0)
        alpha_theta1 = self.kappa_theta1 * (self.theta1_bar - theta1) + alpha_theta1_r
        alpha_theta_r = self.delta_r * diff_r * (1.5 + theta) * (1 - (1.5 + theta) / 3.0)
        alpha_theta_p = self.delta_p * diff_p * (1.5 + theta) * (1 - (1.5 + theta) / 3.0)
        alpha_theta = self.kappa_theta * (self.theta_bar - theta) + alpha_theta_r + alpha_theta_p
        alpha_x = tf.concat([alpha_r, alpha_p, alpha_theta1, alpha_theta], 1)
        return alpha_x

    def mu_x(self, x):
        """The adjusted drift of state varialbe in the equivalent measure"""
        mpr = x[:, self.dim :]
        # mpr_stack = tf.concat([mpr, mpr], 1)
        # sigma_x = self.sigma_x(x)
        mu_x = self.alpha_x(x) + ((1 - self.gamma) / self.gamma) * (tf.concat([mpr, mpr], 1) * self.sigma_x(x))
        return mu_x

    @tf.function()
    def next_x(self, x, dw):
        mu_x = self.mu_x(x)
        sigma_x = self.sigma_x(x)  # [M,5,3]
        x_new = x + self.delta_t * mu_x + (tf.concat([dw, dw], 1) * sigma_x)
        x_new = tf.maximum(x_new, 0)
        return x_new

    @tf.function()
    def next_y(self, t, x, y, z, dw, lb, ub, zdx=True):
        if zdx:
            # temp = z * self.sigma_x(x)
            # z = temp[:, : self.dim] + temp[:, self.dim :]
            z = self.z_T_matmul_sigma_x(z, self.sigma_x(x))
        y = tf.minimum(y, 30.0)
        y = tf.maximum(y, -30.0)
        z = tf.cast(tf.minimum(z, 10.0), TF_DTYPE)
        z = tf.cast(tf.maximum(z, -10.0), TF_DTYPE)
        mpr = x[:, self.dim :]
        optimal_pi_sigma = (mpr + z) / self.gamma
        pi = self.multiply_with_vol_T(optimal_pi_sigma)
        y_new = y - self.delta_t * self.f_tf(t, x, y, z) + tf.reduce_sum(tf.multiply(z, dw), 1, keepdims=True)

        return y_new, pi

    def f_tf(self, t, x, y, z):
        r = tf.expand_dims(x[:, 0], 1)
        mpr = x[:, 2:5]
        r_tilde = -(1 / self.k) * (
            r * (1 - self.gamma)
            + 0.5 * ((1 - self.gamma) / self.gamma) * tf.reduce_sum(tf.square(mpr), 1, keepdims=True)
            - self.delta * self.theta
        )
        f = -r_tilde * y + (self.delta ** self.psi / (1 - self.q_tilde)) * tf.pow(tf.maximum(y, 0), self.q_tilde)
        return f

    def g_tf(self, t, x):
        return tf.ones(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)
