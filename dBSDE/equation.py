import pickle
from math import isnan

import numpy as np

import tensorflow as tf

# import matlab.engine
from scipy.integrate import quad
from scipy.stats import multivariate_normal as normal
from sklearn.model_selection import train_test_split

# from lambda_model import LambdaSolver
# import cvxpy as cp
# from cvxpylayers.tensorflow import CvxpyLayer

TF_DTYPE = tf.float32
MAX = float("Inf")
MIN = 0.0


# eng = matlab.engine.start_matlab()


class Equation(object):
    """Base class for defining PDE related function."""

    def __init__(self, config):
        self.dim = config.dim
        self.total_time = config.total_time
        self.num_time_interval = config.num_time_interval
        # self.steps_in_year = 40
        # self.num_time_interval = self.total_time * self.steps_in_year + 1
        self.delta_t = (self.total_time + 0.0) / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        # self.x0 = config.x_init

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def z_T_matmul_sigma_x(self, z, sigma_x):
        return tf.squeeze(tf.matmul(tf.expand_dims(z, 0), sigma_x))

    def f_tf(self, t, x, y, z):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x):
        """Terminal condition of the PDE."""
        raise NotImplementedError


def get_equation(name, config):
    try:
        return globals()[name](config)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")


class Heston_np(object):
    def __init__(self, config):
        self.dim = config.dim
        self.total_time = config.total_time
        self.num_time_interval = config.num_time_interval
        self.delta_t = (self.total_time + 0.0) / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.x0 = config.x_init
        self.x_init = config.x_init
        self.delta = 0.05
        self.r = 0.05
        self.kappa = 5.0
        self.var_theta = self.kappa * 0.15 ** 2
        self.beta_bar = 0.25
        self.mu_bar = 3.11
        self.rho = -0.5
        self.gamma = config.gamma
        self.psi = config.psi  # 0.125
        self.k = self.gamma / (self.gamma + (1 - self.gamma) * self.rho ** 2)
        # self.k = 1.
        self.theta = (1 - self.gamma) / (1 - 1 / self.psi)
        self.q_tilde = 1 - self.psi * self.k / self.theta
        self.dimx = self.dim

        self.d_th = self.delta * self.theta
        self.coef_exp = (self.theta / self.psi) * self.delta ** self.psi

    def sample(self, num_sample, seed=1):
        # dw_sample and x_sample are both [M,dim,N]
        # eng.rng(2018)
        np.random.seed(seed)
        dw_sample = np.asarray(
            normal.rvs(size=[int(num_sample / 2), self.dim, self.num_time_interval]) * self.sqrt_delta_t
        )
        dw_sample = np.concatenate((dw_sample, -dw_sample), axis=0)
        dw_sample = np.reshape(dw_sample, (num_sample, self.dim, self.num_time_interval))
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init
        new_kappa = -(-self.kappa + ((1 - self.gamma) / self.gamma) * self.beta_bar * self.mu_bar * self.rho)
        new_theta = self.var_theta / new_kappa
        sigmax = np.empty([num_sample, self.dimx, self.dim, self.num_time_interval])  # M x 1 x 1 x N

        for i in range(self.num_time_interval):
            X = x_sample[:, :, i]
            sigmax[:, :, :, i] = np.expand_dims(self.beta_bar * np.sqrt(X), axis=-1)  # M x 1 x 1 x N
            Z = dw_sample[:, :, i]
            # x_sample[:, :, i + 1] = self.euler(x_sample[:, :, i], dw_sample[:, :, i], new_kappa, new_theta)
            x_new = X + new_kappa * (new_theta - X) * self.delta_t + sigmax[:, :, 0, i] * Z
            x_new = tf.minimum(x_new, MAX)
            x_new = tf.maximum(x_new, 0.0)
            x_sample[:, :, i + 1] = x_new
        return dw_sample, x_sample, sigmax

    # def ijk(self, X, Z, kappa, theta):
    #     return (X + kappa * theta * self.delta_t + self.beta_bar * np.sqrt(X) * Z \
    #             + 0.25 * self.beta_bar**2 * (Z**2 - self.delta_t)) \
    #            / (1 + kappa * self.delta_t)
    #
    # def _euler(self, X, Z, kappa, theta):
    #     return X + kappa * (theta - X) * self.delta_t + self.beta_bar * np.sqrt(X) * Z
    #
    def next_x(self, x, dw):
        new_kappa = -(-self.kappa + ((1 - self.gamma) / self.gamma) * self.beta_bar * self.mu_bar * self.rho)
        new_theta = self.var_theta / new_kappa
        sigmax = self.sigma_x(x)
        dw = tf.expand_dims(dw, axis=2)
        x_new = x + new_kappa * (new_theta - x) * self.delta_t + tf.squeeze(tf.matmul(sigmax, dw), axis=2)
        return x_new

    def sigma_x(self, X):
        return tf.expand_dims(self.beta_bar * tf.sqrt(X), axis=1)

    def sigma_x_np(self, X):
        return self.beta_bar * np.sqrt(X)

    def f_tf(self, t, x, y, z):
        if self.q_tilde == 0:
            f = -self.r_tilde(x) * y + self.delta ** self.psi
        else:
            f = -self.r_tilde(x) * y + (self.delta ** self.psi / (1 - self.q_tilde)) * tf.pow(
                tf.maximum(y, 1e-8), self.q_tilde
            )
            # - ((1 - self.gamma) * self.k * self.rho**2 / self.gamma + self.k - 1) * z**2 / (2 * y)
        return f

    def f(self, x, y, z, lb, ub):
        z = np.squeeze(z)
        x = np.squeeze(x)
        optimal_pi_sigma = (self.mu_bar * np.sqrt(np.maximum(x, 0)) + self.rho * z) / self.gamma
        if ub is None and lb is None:
            f = (
                self.r_tilde(x)
                - self.d_th
                + self.coef_exp * np.exp(-self.psi * y / self.theta)
                + (z * z) / (2 * self.gamma)
            )
        else:
            pi_sigma = np.minimum(np.sqrt(np.maximum(x, 0)) * ub, optimal_pi_sigma)
            pi_sigma = np.maximum(np.sqrt(np.maximum(x, 0)) * lb, pi_sigma)
            f = (
                self.r_tilde(x)
                - self.d_th
                + self.coef_exp * np.exp(-self.psi * y / self.theta)
                + (z * z) / (2 * self.gamma)
                - (self.gamma * (1 - self.gamma) / 2) * (pi_sigma - optimal_pi_sigma) ** 2
            )
        return f

    def r_tilde(self, x):
        return (1 - self.gamma) * (self.r + 0.5 * (self.mu_bar ** 2 * x) / self.gamma)

    def g(self, t, x):
        if self.psi == 0.125 and self.gamma == 2:
            return 0.0
        return 0.0

    def g_tf(self, t, x):
        return tf.zeros(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)

    def a_b(self, y, t, s):
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
        # integrand = np.exp(a_exact - b_exact * y)
        return a_exact, b_exact

    def h_exact(self, x, t, T):
        def integrand(s):
            return np.exp(self.a_b(x, t, s)[0] - self.a_b(x, t, s)[1] * x)

        return self.delta ** self.psi * quad(integrand, t, T)[0]

    def hx_exact(self, x, t, T):
        def integrand(s):
            return self.a_b(x, t, s)[1] * np.exp(self.a_b(x, t, s)[0] - self.a_b(x, t, s)[1] * x)

        return -self.delta ** self.psi * quad(integrand, t, T)[0]


class Heston_exp(Equation):
    def __init__(self, config):
        # use 'super()' to refer to the base class 'Equation'
        # here is to use the base class __init__method to initiate the attributes
        super(Heston_exp, self).__init__(config)
        # self.x_init = np.ones(self.dim) * 0.15 ** 2
        self.x_init = config.x_init
        self.delta = 0.05
        self.r = 0.05
        self.kappa = 5.0
        self.var_theta = self.kappa * 0.15 ** 2
        self.beta_bar = 0.25
        self.mu_bar = 3.11
        self.rho = -0.5
        # self.rho = 1
        self.gamma = config.gamma
        self.psi = config.psi  # 0.125
        self.k = self.gamma / (self.gamma + (1 - self.gamma) * self.rho ** 2)
        # self.k = 1.
        self.theta = (1 - self.gamma) / (1 - 1 / self.psi)
        self.q_tilde = 1 - self.psi * self.k / self.theta
        self.dimx = self.dim

        self.d_th = self.delta * self.theta
        self.coef_exp = (self.theta / self.psi) * self.delta ** self.psi

    @tf.function
    def next_x(self, x, dw):
        new_kappa = -(-self.kappa + ((1 - self.gamma) / self.gamma) * self.beta_bar * self.mu_bar * self.rho)
        new_theta = self.var_theta / new_kappa
        # pdb.set_trace()
        sigmax = self.sigma_x(x)
        dw = tf.expand_dims(dw, axis=2)
        x_new = x + new_kappa * (new_theta - x) * self.delta_t + tf.squeeze(tf.matmul(sigmax, dw), axis=2)
        x_new = tf.minimum(x_new, MAX)
        x_new = tf.maximum(x_new, MIN)
        return x_new

    def sigma_x(self, x):
        # x = tf.maximum(x, MIN)
        return tf.expand_dims(self.beta_bar * tf.sqrt(x), axis=1)

    @tf.function
    def next_y(self, t, x, y, z, dw, lb, ub, zdx=True):
        if zdx:
            z = tf.reshape(self.sigma_x(x) @ tf.expand_dims(z, 2), [-1, 1])
        mpr, vol = self.mpr_vol(x)
        optimal_pi_sigma = (mpr + self.rho * z) / self.gamma

        if lb is None and ub is None:
            pi_sigma = optimal_pi_sigma
            y_new = y - self.delta_t * self.f_u(x, y, z) + tf.reduce_sum(tf.multiply(z, dw), 1, keepdims=True)
        else:
            pi_sigma = tf.minimum(optimal_pi_sigma, ub * vol)
            pi_sigma = tf.maximum(pi_sigma, lb * vol)
            y_new = (
                y
                - self.delta_t * self.f_c(x, y, z, mpr, optimal_pi_sigma, pi_sigma)
                + tf.reduce_sum(tf.multiply(z, dw), 1, keepdims=True)
            )
        pi = pi_sigma / vol
        # pdb.set_trace()
        return y_new, pi

    def f_u(self, x, y, z):
        z = tf.minimum(z, 2.0)
        z = tf.maximum(z, -2.0)
        y = tf.minimum(y, 5.0)
        y = tf.maximum(y, -5.0)
        f = (
            self.r_tilde(x)
            - self.d_th
            + self.coef_exp * tf.math.exp(-self.psi * y / self.theta)
            + (z * z) / (2 * self.gamma)
        )
        return f

    def mpr_vol(self, x):
        vol = tf.sqrt(tf.maximum(x, 0.0001))
        mpr = self.mu_bar * vol
        return mpr, vol

    def f_c(self, x, y, z, mpr, optimal_pi_sigma, pi_sigma):
        z = tf.minimum(z, 2.0)
        z = tf.maximum(z, -2.0)
        y = tf.minimum(y, 5.0)
        y = tf.maximum(y, -5.0)
        # optimal_pi_sigma = (mpr + self.rho * z) / self.gamma

        f = (
            self.r_tilde(x)
            - self.d_th
            + self.coef_exp * tf.math.exp(-self.psi * y / self.theta)
            + (z * z) / (2 * self.gamma)
            - (self.gamma * (1 - self.gamma) / 2) * (pi_sigma - optimal_pi_sigma) ** 2
        )

        # if ub is None and lb is None:
        #     f = self.r_tilde(x) - self.d_th + self.coef_exp * tf.math.exp(-self.psi * y / self.theta) + \
        #         (z * z) / (2 * self.gamma)
        #     # print('NotBound', optimal_pi_sigma[0].numpy(), lb, ub)
        # elif ub is not None:
        #     pi_sigma = tf.minimum(optimal_pi_sigma, ub * sigma)
        #     f = self.r_tilde(x) - self.d_th + self.coef_exp * tf.math.exp(-self.psi * y / self.theta) + \
        #         (z * z) / (2 * self.gamma) + ((1 - self.gamma) / self.gamma) * theta * z - \
        #         (self.gamma * (1 - self.gamma) / 2) * (pi_sigma - optimal_pi_sigma)**2
        # else:
        #     pi_sigma = tf.maximum(optimal_pi_sigma, lb * sigma)
        #     f = self.r_tilde(x) - self.d_th + self.coef_exp * tf.math.exp(-self.psi * y / self.theta) + \
        #         (z * z) / (2 * self.gamma) + ((1 - self.gamma) / self.gamma) * theta * z - \
        #         (self.gamma * (1 - self.gamma) / 2) * (pi_sigma - optimal_pi_sigma)**2
        #     # print('bound', optimal_pi_sigma[0].numpy(), pi_sigma[0].numpy(), lb, ub)
        return f

    def r_tilde(self, x):
        # return -(1 / self.k) * (self.r * (1 - self.gamma) + 0.5 * self.mu_bar**2 * x * (1 - self.gamma) / self.gamma - self.delta * self.theta)
        return (1 - self.gamma) * (self.r + 0.5 * (self.mu_bar ** 2 * x) / self.gamma)

    def g_tf(self, t, x):
        if self.psi == 0.125 and self.gamma == 2:
            return tf.zeros(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)
        return tf.zeros(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)

    def a_b(self, y, t, s):
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
        # integrand = np.exp(a_exact - b_exact * y)
        return a_exact, b_exact

    def h_exact(self, x, t, T):
        def integrand(s):
            return np.exp(self.a_b(x, t, s)[0] - self.a_b(x, t, s)[1] * x)

        return self.delta ** self.psi * quad(integrand, t, T)[0]

    def hx_exact(self, x, t, T):
        def integrand(s):
            return self.a_b(x, t, s)[1] * np.exp(self.a_b(x, t, s)[0] - self.a_b(x, t, s)[1] * x)

        return -self.delta ** self.psi * quad(integrand, t, T)[0]


class Heston_poly(Heston_exp):
    def __init__(self, config):
        # use 'super()' to refer to the base class 'Equation'
        # here is to use the base class __init__method to initiate the attributes
        super(Heston_poly, self).__init__(config)
        # self.x_init = np.ones(self.dim) * 0.15 ** 2
        self.x_init = config.x_init
        self.delta = 0.05
        self.r = 0.05
        self.kappa = 5.0
        self.var_theta = self.kappa * 0.15 ** 2
        self.beta_bar = 0.25
        self.mu_bar = 3.11
        self.rho = -0.5
        self.gamma = config.gamma
        self.psi = config.psi  # 0.125
        self.k = self.gamma / (self.gamma + (1 - self.gamma) * self.rho ** 2)
        # self.k = 1.
        self.theta = (1 - self.gamma) / (1 - 1 / self.psi)
        self.q_tilde = 1 - self.psi * self.k / self.theta
        self.dimx = self.dim

    def f_u(self, x, y, z):
        # K=1e6
        # y=tf.minimum(y,K)
        # y = tf.maximum(y, 1/K)
        if self.q_tilde == 0:
            f = -self.r_tilde(x) * y + self.delta ** self.psi
        else:
            f = -self.r_tilde(x) * y + (self.delta ** self.psi / (1 - self.q_tilde)) * tf.pow(
                tf.maximum(y, 1 / MAX), self.q_tilde
            )
            # - ((1 - self.gamma) * self.k * self.rho**2 / self.gamma + self.k - 1) * z**2 / (2 * y)
        return f

    def r_tilde(self, x):
        return -(1 / self.k) * (
            self.r * (1 - self.gamma)
            + 0.5 * self.mu_bar ** 2 * x * (1 - self.gamma) / self.gamma
            - self.delta * self.theta
        )

    def g_tf(self, t, x):
        if self.psi == 0.125 and self.gamma == 2:
            return tf.zeros(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)
        return tf.ones(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)


class NewEcon_np(Equation):
    def __init__(self, config):
        super(NewEcon_np, self).__init__(config)
        self.config = config
        self.delta = config.delta
        self.x_init = config.x_init
        self.gamma = config.gamma
        self.psi = config.psi
        self.k = self.gamma
        self.theta = (1 - self.gamma) / (1 - 1 / self.psi)
        self.q_tilde = 1 - self.psi * self.k / self.theta
        self.dimx = 5
        self.d_th = self.delta * self.theta
        self.coef_exp = (self.theta / self.psi) * self.delta ** self.psi

    def sample(self, num_sample, seed):

        # N=self.num_time_interval
        # M=num_sample
        # eng=matlab.engine.start_matlab()
        # eng.rng(2018)
        # dw_sample = eng.randn(M/2,3,N)
        # dw_sample = np.asarray(dw_sample) * self.sqrt_delta_t
        # eng.quit
        np.random.seed(seed)
        dw_sample = np.asarray(
            normal.rvs(size=[int(num_sample / 2), self.dim, self.num_time_interval]) * self.sqrt_delta_t
        )
        dw_sample = np.concatenate((dw_sample, -dw_sample), axis=0)
        dw_sample = np.reshape(dw_sample, (num_sample, self.dim, self.num_time_interval))
        # dw_sample is M x 3 x N
        # x_sample is M x 5 x (N+1)
        x_sample = np.tile(self.x_init, [num_sample, self.num_time_interval + 1, 1]).transpose(0, 2, 1)
        sigmax = np.empty([num_sample, self.dimx, self.dim, self.num_time_interval])  # M x 5 x 3 x N
        for i in range(self.num_time_interval):
            x = x_sample[:, :, i]
            mux = self.mu_x(x)
            sigmax[:, :, :, i] = self.sigma_x(x)
            dw = np.expand_dims(dw_sample[:, :, i], axis=2)
            x_new = x + self.delta_t * mux + np.squeeze(np.matmul(sigmax[:, :, :, i], dw))
            x_new = np.minimum(x_new, MAX)
            x_new = np.maximum(x_new, MIN)
            x_sample[:, :, i + 1] = x_new
        return dw_sample, x_sample, sigmax

    def mu_x(self, x):
        """
        :param x: M x 5 [r,p,mpr1,mpr2,mpr3]
        :return: M x 5
        """

        mpr = x[:, 2:5]
        mu_x = self.alpha(x) + ((1 - self.gamma) / self.gamma) * np.squeeze(
            np.matmul(self.sigma_x(x), np.expand_dims(mpr, axis=2))
        )
        return mu_x

    def alpha(self, x):
        """
        :param x: M x 5 [r,p,mpr1,mpr2,mpr3]
        :return: M x 5
        """

        M = np.shape(x)[0]
        alpha = np.empty([M, 5])
        r = x[:, 0]  # M
        p = x[:, 1]  # M
        mpr = x[:, 2:5]  # M x 3

        # r
        diff_r = self.config.r_bar - r
        alpha[:, 0] = self.config.kappa_r * diff_r * (1 + self.config.phi_r * (diff_r ** 2) ** self.config.eta_r)
        # p
        diff_p = self.config.p_bar - p
        alpha[:, 1] = self.config.kappa_p * diff_p * (1 + self.config.phi_p * (diff_p ** 2) ** self.config.eta_p)
        # mpr
        for i in range(0, 3):
            theta_r = (
                diff_r
                * self.config.delta_r[i]
                * (self.config.theta_l[i] + mpr[:, i])
                * (1 - ((self.config.theta_l[i] + mpr[:, i]) / (self.config.theta_l[i] + self.config.theta_u[i])))
            )
            theta_p = (
                diff_p
                * self.config.delta_p[i]
                * (self.config.theta_l[i] + mpr[:, i])
                * (1 - ((self.config.theta_l[i] + mpr[:, i]) / (self.config.theta_l[i] + self.config.theta_u[i])))
            )
            theta = self.config.kappa_theta[i] * (self.config.theta_bar[i] - mpr[:, i])
            alpha[:, 2 + i] = theta + theta_r + theta_p
        return alpha

    def sigma_x(self, x):
        """
        :param x: M x 5 [r,p,mpr1,mpr2,mpr3]
        :return: M x 5 x 3
        """
        M = x.shape[0]
        sigma = np.empty([M, 5, 3])
        r = x[:, 0]  # M
        p = x[:, 1]  # M
        mpr = x[:, 2:5]  # M x 3

        for i in range(0, 3):
            sigma[:, 0, i] = -np.maximum(r, 0) ** self.config.gamma_r * self.config.sigma_r[i]
            sigma[:, 1, i] = -np.maximum(p, 0) ** self.config.gamma_p * self.config.sigma_p[i]
            for j in range(0, 3):
                sigma[:, 2 + j, i] = self.config.sigma_theta[j, i] * (
                    (self.config.theta_l[j] + mpr[:, j]) ** self.config.gamma1_theta[j]
                    * np.maximum(
                        (
                            1
                            - (
                                (self.config.theta_l[j] + mpr[:, j])
                                / (self.config.theta_l[j] + self.config.theta_u[j])
                            )
                            ** (1 - self.config.gamma1_theta[j])
                        ),
                        0,
                    )
                    ** self.config.gamma2_theta[j]
                )
        return sigma

    # def f_tf(self, t, x, y, z):
    #     f = -self.r_tilde_tf(x) * y + (self.delta ** self.psi / (1 - self.q_tilde)) * tf.pow(
    #         tf.maximum(y, 0), self.q_tilde
    #     )
    #     return f
    def f(self, x, y, z, lb, ub):
        f = (
            self.r_tilde_exp(x)
            - self.d_th
            + self.coef_exp * np.exp(-self.psi * y / self.theta)
            + np.sum(np.square(z), 1, keepdims=False) / (2 * self.gamma)
        )
        return f

    def r_tilde_exp(self, x):
        """
        :param x: M x 5 x 1
        :return: M x 1 x 1
        """
        r = x[:, 0, :]
        mpr = x[:, 2:5, :]
        return (1 - self.gamma) * (r + 0.5 * np.sum(np.square(mpr), 1, keepdims=False) / self.gamma)

    # def f(self, t, x, y, z):
    #     """
    #     conjecture: y^k
    #     :param x: M x 5
    #     :param y: M x 1
    #     :param z: M x 3
    #     return : M x 1
    #     """
    #     f = -self.r_tilde(x) * y + (self.delta ** self.psi / (1 - self.q_tilde)) * np.power(
    #         np.maximum(y, 1e-10), self.q_tilde
    #     )
    #     # -(1-self.gamma)*self.k*self.rho**2/self.gamma+self.k-1)*z**2/(2*y)
    #     return f

    # def r_tilde(self, x):
    #     """
    #     conjecture: y^k
    #     :param x: M x 5
    #     :return: M x 1
    #     """
    #     r = x[:, 0]
    #     mpr = x[:, 2:5]
    #     return -(1 / self.k) * (
    #         r * (1 - self.gamma)
    #         + 0.5 * ((1 - self.gamma) / self.gamma) * np.sum(np.power(mpr, 2), axis=1)
    #         - self.delta * self.theta
    #     )

    # def r_tilde_tf(self, x):
    #     """
    #     :param x: M x 5 x 1
    #     :return: M x 1 x 1
    #     """
    #     r = tf.expand_dims(x[:, 0], 1)
    #     mpr = x[:, 2:5]
    #     return -(1 / self.k) * (
    #         r * (1 - self.gamma)
    #         + 0.5 * ((1 - self.gamma) / self.gamma) * tf.reduce_sum(tf.square(mpr), 1, keepdims=True)
    #         - self.delta * self.theta
    #     )

    # def g_tf(self, t, x):

    #     return tf.ones(shape=tf.stack([tf.shape(x)[0], 1]), dtype=TF_DTYPE)

    def g(self, t, x):
        return 0.0


class NewEcon_exp(Equation):
    def __init__(self, config):
        super(NewEcon_exp, self).__init__(config)
        self.config = config
        self.delta = tf.constant(config.delta, TF_DTYPE)
        self.x_init = tf.constant(config.x_init, TF_DTYPE)
        self.gamma = tf.constant(config.gamma, TF_DTYPE)
        self.psi = tf.constant(config.psi, TF_DTYPE)
        self.k = self.gamma
        self.theta = (1 - self.gamma) / (1 - 1 / self.psi)
        self.q_tilde = 1 - self.psi * self.k / self.theta
        self.dimx = 5

        self.delta = tf.constant(0.05, TF_DTYPE)
        self.d_th = self.delta * self.theta
        self.coef_exp = (self.theta / self.psi) * self.delta ** self.psi

        # param for alpha(x)
        self.kappa_r = tf.constant(0.00034, TF_DTYPE)
        self.r_bar = tf.constant(0.00520 * 12, TF_DTYPE)
        self.phi_r = tf.constant(17224.987 / 12, TF_DTYPE)
        # eta_r = 0.4116
        self.eta_r = tf.constant(0.5, TF_DTYPE)
        self.kappa_p = tf.constant(0.005, TF_DTYPE)
        self.p_bar = tf.constant(0.0332 * 12, TF_DTYPE)
        self.phi_p = tf.constant(0, TF_DTYPE)
        self.eta_p = tf.constant(0, TF_DTYPE)
        self.theta_l = tf.constant([1.5, 1.5, 2.5], TF_DTYPE)
        self.theta_u = tf.constant([1.5, 1.5, 2.5], TF_DTYPE)
        self.kappa_theta = tf.constant([0.1219, 0.5946, 0.7483], TF_DTYPE)
        self.theta_bar = tf.constant([0.1562, 0.4251, 0.1669], TF_DTYPE)
        self.delta_r = tf.constant([-2.7648 / 12.0, 139.0 / 12.0, 176.76 / 12.0], TF_DTYPE)
        self.delta_p = tf.constant([-0.0372, 14.04, -4.44], TF_DTYPE)

        # param for sigma_x(x)
        self.gamma_r = 0.5664
        self.sigma_r = tf.constant([0.00986 * 12 ** (1 - self.gamma_r), 0, 0], TF_DTYPE)
        self.gamma_p = 0.5
        self.sigma_p = tf.constant([-0.00103 * 12 ** 0.5, 0.01011 * 12 ** 0.5, 0.003 * 12 ** 0.5], TF_DTYPE)
        self.sigma_theta = tf.constant(
            [
                [-0.2032, 0.00497, 0.0356],
                [-0.1237, -0.0306, 0.0406],
                [-0.1199, -0.493, 0.402],
            ],
            TF_DTYPE,
        )
        self.gamma1_theta = tf.constant([0.5, 0.5, 0.5], TF_DTYPE)
        self.gamma2_theta = tf.constant([1.2158, 0.5095, 0.6440], TF_DTYPE)

        sigma_b = 0.156
        sigma_1 = 0.106
        sigma_2 = 0.229
        rho_0 = 0.37
        rho_1 = 0.32
        rho_2 = -0.1274
        rho_1_vec = np.array([rho_0, np.sqrt(1 - rho_0 ** 2), 0])
        rho_2_vec = np.array([rho_1, rho_2, np.sqrt(1 - rho_1 ** 2 - rho_2 ** 2)])
        vol = np.array([np.array([sigma_b, 0, 0]), sigma_1 * rho_1_vec, sigma_2 * rho_2_vec])
        # vol_T = np.transpose(vol)
        vol_T_inv = np.linalg.inv(np.transpose(vol))
        # var_inv = np.linalg.inv(np.matmul(np.transpose(vol), vol))
        self.vol = tf.constant(vol, dtype=TF_DTYPE)
        self.vol_T = tf.transpose(self.vol)
        self.vol_T_inv = tf.constant(vol_T_inv, dtype=TF_DTYPE)
        self.w = tf.reduce_sum(self.vol_T_inv, 0)  # [3,]
        # self.pi_model = self.get_pi_model()
        # self.var_inv = tf.constant(var_inv, dtype=TF_DTYPE)
        # x = cp.Variable(3)
        # A = cp.Parameter((m, n))
        # b = cp.Parameter(3)
        # constraints = [x >= 0, cp.sum(x) <= 0.2]
        # objective = cp.Minimize(cp.sum_squares(vol_T @ x - b))
        # problem = cp.Problem(objective, constraints)
        # self.cvxpylayer = CvxpyLayer(problem, parameters=[b], variables=[x])

    # # def get_pi_model(self):
    #     pkl_file = open("./logs/model/dataset.pkl", "rb")
    #     dataset = pickle.load(pkl_file)
    #     pkl_file.close()
    #     input_data = dataset[:, 3:]
    #     output_data = dataset[:, :3]
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         input_data, output_data, test_size=0.2, random_state=42
    #     )

    #     new_model = LambdaSolver(activation_fn="relu")
    #     # for layer in new_model.layers:
    #     #     layer.trainable = False
    #     new_model.trainable = False

    #     new_model.compile(
    #         optimizer=tf.keras.optimizers.Adam(1e-3), loss="mean_squared_error"
    #     )
    #     # This initializes the variables used by the optimizers,
    #     # as well as any stateful metric variables
    #     # optimizer = tf.keras.optimizers.Adam(1e-3)
    #     new_model.train_on_batch(X_train[:1], y_train[:1])
    #     # Load the state of the old model
    #     new_model.load_weights("./logs/model/lambda_model")
    #     return new_model

    def pi_d2(self, b, lb, ub):
        """
        b=(mpr+z)/gamma [M,3]
        return:
            pi: [M,3]
            d2: [M,1]
        """
        # a = tf.reduce_sum(tf.squeeze(tf.matmul(self.vol_T_inv, tf.expand_dims(b, axis=2))), axis=1, keepdims=True) - ub  # [M,1]
        # batches = tf.shape(b)[0]
        # c = tf.broadcast_to(tf.reduce_sum(self.var_inv, axis=1), [batches, self.dim])  # [M, 3]
        # w_T_W = tf.reduce_sum(c, axis=1, keepdims=True)  # [M,1]
        # optimal_pi = tf.squeeze(tf.matmul(self.vol_T_inv, tf.expand_dims(b, axis=2)))
        w = self.w
        k = (tf.reduce_sum(w * b, 1, keepdims=True) - ub) / tf.reduce_sum(w ** 2)  # [M,1]

        if lb is None:
            d = tf.maximum(k, 0) * w
            pi = tf.squeeze(tf.matmul(self.vol_T_inv, tf.expand_dims(b - d, axis=2)))
            d2 = tf.reduce_sum(d ** 2, axis=1, keepdims=True)

        else:
            lb = tf.cast(lb, dtype=TF_DTYPE)
            pi = self.pi_model(b)
            d2 = tf.reduce_sum(
                tf.squeeze(self.vol_T @ tf.expand_dims(pi, 2)) - b,
                axis=1,
                keepdims=True,
            )  # [M,1]

            # # First try to project pi onto the face
            # x_face = tf.maximum(lb, b)  # [M, 3]
            # # right_ub [M, 1] 1.0 means the projection on the surface is to the right of ub plane
            # right_ub = tf.cast(tf.reduce_sum(w*x_face, axis=1, keepdims=True) > ub, dtype=TF_DTYPE)
            # left_ub = 1-right_ub

            # # Second option is to project pi onto the triangle
            # # First project pi onto the plane
            # x_plane = b - k * w
            # # Project x_plane onto the closest edge
            # a1 = tf.cast(x_plane < lb, dtype=TF_DTYPE)
            # # f1=0 means two of the coordinates of x_plane is negative, in which case the closest point is the vertex
            # f1 = tf.cast(tf.reduce_sum(a1, 1, keepdims=True) == 1, dtype=TF_DTYPE)
            # w1, w23 = a1 * w, (1 - a1) * w
            # d1 = -a1 * (x_plane - lb) + (w23 / tf.norm(w23, axis=1, keepdims=True)**2) * tf.reduce_sum(w1 * (x_plane - lb), 1, keepdims=True)
            # x_edge = x_plane + d1
            # # Check if x_edge lies between the two ends of the edge, if not, further move it to the vertex
            # a2 = tf.cast(x_edge < lb, dtype=TF_DTYPE)
            # w2, w3 = a2 * w, (1 - a1 - a2) * w
            # d2 = -a2 * (x_edge - lb) + (w3 / tf.norm(w3, axis=1, keepdims=True)**2) * tf.reduce_sum(w2 * (x_edge - lb), 1, keepdims=True)
            # x_edge += d2

            # x_vertex = (1 - a1) * ub / (tf.reduce_sum((1 - a1) * w, 1, keepdims=True))

            # x_triangle = (1 - f1) * x_vertex + f1 * x_edge

            # x_new = x_face * left_ub + x_triangle * right_ub
            # pi = tf.squeeze(tf.matmul(self.vol_T_inv, tf.expand_dims(x_new, axis=2)))
            # d2 = tf.reduce_sum((x_new - b)**2, axis=1, keepdims=True)
            # pi, = self.cvxpylayer(b)
            # pi = tf.cast(pi, dtype=TF_DTYPE)
            # # pdb.set_trace()
            # d2 = tf.reduce_sum(tf.squeeze(self.vol_T @ tf.expand_dims(pi, 2)) - b, axis=1, keepdims=True)  # [M,1]

        return pi, d2

    @tf.function()
    def next_x(self, x, dw):
        """
        :param x:[M,5]
        :param dw: [M,1]
        :return: [M,5]
        """

        mu_x = self.mu_x(x)
        sigma_x = self.sigma_x(x)  # [M,5,3]
        # tf.print('x: ', tf.reduce_mean(x,axis=0), 'alpha_x: ', tf.reduce_mean(self.alpha(x),axis=0),
        #          'sigma_x: ', tf.reduce_mean(self.sigma_x(x)[:,:,0], axis=0))
        dw = tf.expand_dims(dw, 2)
        x_new = x + self.delta_t * mu_x + tf.squeeze(tf.matmul(sigma_x, dw))
        x_new = tf.minimum(x_new, MAX)
        x_new = tf.maximum(x_new, MIN)
        return x_new

    def mu_x(self, x):
        """
        :param x: M x 5 [r,p,mpr1,mpr2,mpr3]
        :return: M x 5
        """

        mpr = x[:, 2:5]
        mu_x = self.alpha(x) + ((1 - self.gamma) / self.gamma) * tf.squeeze(
            tf.matmul(self.sigma_x(x), tf.expand_dims(mpr, axis=2))
        )
        return mu_x

    def alpha(self, x):
        """
        :param x: M x 5 [r,p,mpr1,mpr2,mpr3]
        :return: M x 5
        """

        M = tf.shape(x)[0]
        alpha = tf.TensorArray(TF_DTYPE, size=5)
        r = x[:, 0]  # M
        p = x[:, 1]  # M
        mpr = x[:, 2:5]  # M x 3

        # r
        diff_r = self.r_bar - r
        alpha = alpha.write(0, self.kappa_r * diff_r * (1 + self.phi_r * (diff_r ** 2) ** self.eta_r))
        # p
        diff_p = self.p_bar - p
        alpha = alpha.write(1, self.kappa_p * diff_p * (1 + self.phi_p * (diff_p ** 2) ** self.eta_p))
        # mpr
        for i in range(3):
            theta_r = (
                diff_r
                * self.delta_r[i]
                * (self.theta_l[i] + mpr[:, i])
                * (1 - ((self.theta_l[i] + mpr[:, i]) / (self.theta_l[i] + self.theta_u[i])))
            )
            theta_p = (
                diff_p
                * self.delta_p[i]
                * (self.theta_l[i] + mpr[:, i])
                * (1 - ((self.theta_l[i] + mpr[:, i]) / (self.theta_l[i] + self.theta_u[i])))
            )
            theta = self.kappa_theta[i] * (self.theta_bar[i] - mpr[:, i])
            alpha = alpha.write(i + 2, theta + theta_r + theta_p)
        return tf.transpose(alpha.stack(), [1, 0])

    def sigma_x(self, x):
        """
        :param x: M x 5 [r,p,mpr1,mpr2,mpr3]
        :return: M x 5 x 3
        """
        M = tf.shape(x)[0]
        r = x[:, 0]  # M
        p = x[:, 1]  # M
        mpr = x[:, 2:5]  # M x 3
        sigma_r = tf.TensorArray(TF_DTYPE, size=3)  # [3,M]
        sigma_p = tf.TensorArray(TF_DTYPE, size=3)
        sigma_theta_0 = tf.TensorArray(TF_DTYPE, size=3)
        sigma_theta_1 = tf.TensorArray(TF_DTYPE, size=3)
        sigma_theta_2 = tf.TensorArray(TF_DTYPE, size=3)
        for i in range(3):
            sigma_r = sigma_r.write(i, -tf.maximum(r, 0.0) ** self.gamma_r * self.sigma_r[i])
            sigma_p = sigma_p.write(i, -tf.maximum(p, 0.0) ** self.gamma_p * self.sigma_p[i])
            sigma_theta_0 = sigma_theta_0.write(
                i,
                self.sigma_theta[0, i]
                * (
                    tf.pow((self.theta_l[0] + mpr[:, 0]), self.gamma1_theta[0])
                    * tf.maximum(
                        (
                            1
                            - ((self.theta_l[0] + mpr[:, 0]) / (self.theta_l[0] + self.theta_u[0]))
                            ** (1 - self.gamma1_theta[0])
                        ),
                        0.0,
                    )
                    ** self.gamma2_theta[0]
                ),
            )
            sigma_theta_1 = sigma_theta_1.write(
                i,
                self.sigma_theta[1, i]
                * (
                    (self.theta_l[1] + mpr[:, 1]) ** self.gamma1_theta[1]
                    * tf.maximum(
                        (
                            1
                            - ((self.theta_l[1] + mpr[:, 1]) / (self.theta_l[1] + self.theta_u[1]))
                            ** (1 - self.gamma1_theta[1])
                        ),
                        0.0,
                    )
                    ** self.gamma2_theta[1]
                ),
            )
            sigma_theta_2 = sigma_theta_2.write(
                i,
                self.sigma_theta[2, i]
                * (
                    (self.theta_l[2] + mpr[:, 2]) ** self.gamma1_theta[2]
                    * tf.maximum(
                        (
                            1
                            - ((self.theta_l[2] + mpr[:, 2]) / (self.theta_l[2] + self.theta_u[2]))
                            ** (1 - self.gamma1_theta[2])
                        ),
                        0.0,
                    )
                    ** self.gamma2_theta[2]
                ),
            )
        sigma = tf.transpose(
            tf.stack(
                [
                    sigma_r.stack(),
                    sigma_p.stack(),
                    sigma_theta_0.stack(),
                    sigma_theta_1.stack(),
                    sigma_theta_2.stack(),
                ]
            ),
            [2, 0, 1],
        )
        return sigma

    @tf.function()
    def next_y(self, t, x, y, z, dw, lb, ub, zdx=True):
        if zdx:
            # sigma_x_w = tf.squeeze(tf.matmul(self.sigma_x(x), tf.expand_dims(dw, 2)))
            z = tf.squeeze(tf.matmul(tf.transpose(self.sigma_x(x), [0, 2, 1]), tf.expand_dims(z, 2)))  # [M,3]
        y = tf.minimum(y, 30.0)
        y = tf.maximum(y, -30.0)
        z = tf.cast(tf.minimum(z, 10.0), TF_DTYPE)
        z = tf.cast(tf.maximum(z, -10.0), TF_DTYPE)
        mpr = x[:, 2:5]
        optimal_pi_sigma = (mpr + z) / self.gamma
        if lb is None and ub is None:
            pi = tf.squeeze(tf.matmul(self.vol_T_inv, tf.expand_dims(optimal_pi_sigma, axis=2)))
            y_new = y - self.delta_t * self.f_tf(t, x, y, z) + tf.reduce_sum(tf.multiply(z, dw), 1, keepdims=True)
            # [M,1]
        else:
            ub = tf.cast(ub, dtype=TF_DTYPE)
            pi, d2 = self.pi_d2(optimal_pi_sigma, lb, ub)
            # d2 = tf.minimum(d2, 100)
            # pdb.set_trace()
            f = self.f_tf(t, x, y, z) - (self.gamma * (1 - self.gamma) / 2) * d2
            y_new = y - self.delta_t * f + tf.reduce_sum(tf.multiply(z, dw), 1, keepdims=True)  # [M,1]

        return y_new, pi

    def f_tf(self, t, x, y, z):
        # y = tf.minimum(y, 50.0)
        # y = tf.maximum(y, -50.0)
        # z = tf.minimum(z, 10.0)
        # z = tf.maximum(z, -10.0)
        f = (
            self.r_tilde_tf(x)
            - self.d_th
            + self.coef_exp * tf.math.exp(-self.psi * y / self.theta)
            + tf.reduce_sum(tf.square(z), 1, keepdims=True) / (2 * self.gamma)
        )
        return f

    def r_tilde_tf(self, x):
        """
        :param x: M x 5 x 1
        :return: M x 1 x 1
        """
        r = tf.expand_dims(x[:, 0], 1)
        mpr = x[:, 2:5]
        return (1 - self.gamma) * (r + 0.5 * tf.reduce_sum(tf.square(mpr), 1, keepdims=True) / self.gamma)

    def g_tf(self, t, x):
        return tf.zeros(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)


class NewEcon_poly(NewEcon_exp):
    def f_tf(self, t, x, y, z):
        f = -self.r_tilde_tf(x) * y + (self.delta ** self.psi / (1 - self.q_tilde)) * tf.pow(
            tf.maximum(y, 0), self.q_tilde
        )
        return f

    def r_tilde_tf(self, x):
        """
        :param x: M x 5 x 1
        :return: M x 1 x 1
        """
        r = tf.expand_dims(x[:, 0], 1)
        mpr = x[:, 2:5]
        return -(1 / self.k) * (
            r * (1 - self.gamma)
            + 0.5 * ((1 - self.gamma) / self.gamma) * tf.reduce_sum(tf.square(mpr), 1, keepdims=True)
            - self.delta * self.theta
        )

    def g_tf(self, t, x):

        return tf.ones(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)


class LargeScale_poly(Equation):
    def __init__(self, config):
        super(LargeScale_poly, self).__init__(config)
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

        self.x_init = tf.cast(config.x0, dtype=TF_DTYPE) * tf.stack(
            [self.r_bar] + [self.p_bar] * (self.dim - 1) + [self.theta1_bar] + [self.theta_bar] * (self.dim - 1)
        )

        self.sigma_b = 0.156
        self.sigma_1 = 0.106
        self.sigma_i = 0.229
        self.rho_0 = 0.37

    def multiply_with_vol_T(self, x):
        """
        x: [M,d]
        output : [M,d]
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
            x[1:9] p
            x[10]  theta1
            x[11:19] theta
        output: M x 20
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
            x[1:9] p
            x[10]  theta1
            x[11:19] theta
        output: M x 20
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
        mpr = x[:, self.dim :]
        # mpr_stack = tf.concat([mpr, mpr], 1)
        # sigma_x = self.sigma_x(x)
        mu_x = self.alpha_x(x) + ((1 - self.gamma) / self.gamma) * (tf.concat([mpr, mpr], 1) * self.sigma_x(x))
        return mu_x

    @tf.function()
    def next_x(self, x, dw):
        """
        :param x:[M,20]
        :param dw: [M,10]
        :return: [M,20]
        """

        mu_x = self.mu_x(x)
        sigma_x = self.sigma_x(x)  # [M,5,3]
        x_new = x + self.delta_t * mu_x + (tf.concat([dw, dw], 1) * sigma_x)
        x_new = tf.minimum(x_new, MAX)
        x_new = tf.maximum(x_new, MIN)
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
        if lb is None and ub is None:
            # pi = tf.squeeze(tf.matmul(self.vol_T_inv, tf.expand_dims(optimal_pi_sigma, axis=2)))
            pi = self.multiply_with_vol_T(optimal_pi_sigma)
            y_new = y - self.delta_t * self.f_tf(t, x, y, z) + tf.reduce_sum(tf.multiply(z, dw), 1, keepdims=True)
            # [M,1]
        else:
            ub = tf.cast(ub, dtype=TF_DTYPE)
            pi, d2 = self.pi_d2(optimal_pi_sigma, lb, ub)
            # d2 = tf.minimum(d2, 100)
            # pdb.set_trace()
            f = self.f_tf(t, x, y, z) - (self.gamma * (1 - self.gamma) / 2) * d2
            y_new = y - self.delta_t * f + tf.reduce_sum(tf.multiply(z, dw), 1, keepdims=True)  # [M,1]

        return y_new, pi

    def f_tf(self, t, x, y, z):
        f = -self.r_tilde_tf(x) * y + (self.delta ** self.psi / (1 - self.q_tilde)) * tf.pow(
            tf.maximum(y, 0), self.q_tilde
        )
        return f

    def r_tilde_tf(self, x):
        """
        :param x: M x 5 x 1
        :return: M x 1 x 1
        """
        r = tf.expand_dims(x[:, 0], 1)
        mpr = x[:, 2:5]
        return -(1 / self.k) * (
            r * (1 - self.gamma)
            + 0.5 * ((1 - self.gamma) / self.gamma) * tf.reduce_sum(tf.square(mpr), 1, keepdims=True)
            - self.delta * self.theta
        )

    def g_tf(self, t, x):

        return tf.ones(shape=[tf.shape(x)[0], 1], dtype=TF_DTYPE)


#%%
if __name__ == "__main__":
    from config import get_config
    from equation_large import get_equation
    import tensorflow as tf
    import numpy as np

    problem_name = "LargeScale"
    config = get_config(problem_name)
    equation = get_equation(problem_name + "_poly", config)
    x0 = tf.stack([1.1] * 20)
    x_init = x0 * equation.x_init
    x = tf.broadcast_to(x_init, [2, equation.dim * 2])
    sample_dw = np.random.normal(0, equation.sqrt_delta_t, (2, equation.dim))
    dw = tf.reshape(tf.cast(sample_dw, dtype=TF_DTYPE), [-1, equation.dim])
    # print(x.numpy())
    # print(equation.multiply_with_vol_T(x).numpy)
    y = tf.cast(tf.broadcast_to(3.0, [2, 1]), dtype=tf.float32)
    z = tf.cast(tf.broadcast_to([1.0] * 20, [2, 20]), dtype=tf.float32)
    y_new, _ = equation.next_y(0, x, 3.0, z, dw, None, None, zdx=True)
    print(y_new)

# %%
# mpr = x[:, equation.dim :]
# sigma_x = equation.sigma_x(x)
# tf.concat([mpr, mpr], 1) * sigma_x
# equation.next_x(x, sample_dw)
# %%
