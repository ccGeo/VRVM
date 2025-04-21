import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, logsumexp, gamma
from numpy import linalg as la
import random
from scipy.stats import random_correlation
import scipy.linalg


class VRVM_regression():
    """
    Variational Relavance Vector Machine
    arXiv: 13013838
    Bishop and Tipping
    Developed for Regression

    """

    def __init__(self, N, input_samples, signal, max_iter=400, sigma_phi=1, a=10 ** (-6), b=10 ** (-6), c=10 ** (-6),
                 d=10 ** (-6), actual_signal=None):

        self.N = N
        self.input_samples = input_samples
        self.signal = signal
        self.max_iter = max_iter
        self.sigma_phi = sigma_phi
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.actual_signal = actual_signal

    def kernel(self, x, y, sigma):
        return np.exp(-0.5 * sigma ** (-2) * (x - y) ** 2)

    def _initialize(self):

        self.at = 1 / 2 + self.a
        self.ct = (self.N + 1) / 2 + self.c
        self.dt = self.d + 0.5 * np.sum(np.multiply(self.signal, self.signal))

        self.mu = np.random.rand(self.N)  # np.ones(self.N)
        self.eigen = np.zeros(self.N)
        for i in range(self.N - 1):
            self.eigen[i] = 0.01 * np.random.rand()
        self.eigen[-1] = self.N - np.sum(self.eigen)
        # self.eigen=np.ones(self.N)
        self.Sigma = random_correlation.rvs(self.eigen)

        # self.Sigma=-1+2 * np.random.rand(self.N,self.N)
        # self.Sigma= 0.5*(self.Sigma + np.transpose(self.Sigma))
        # self.Sigma=self.Sigma +np.eye(N)-np.diag(self.Sigma)
        # print(la.det(self.Sigma))

        self.bt = 0.5 * self.mu  # np.random.rand((N))self.b+0.5*np.multiply(self.mu,self.mu)+np.diagonal(self.Sigma))

        self.phi = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.phi[i, j] = (self.kernel(self.input_samples[i], self.input_samples[j], self.sigma_phi))

    def _compute_expectations(self):
        self.x_alpha = self.at / self.bt  # EQ. 30 x
        self.x_ln_alpha = digamma(self.at) - np.log(self.bt)  # EQ. 31 x
        self.x_tau = self.ct / self.dt  # EQ. 32 x
        self.x_ln_tau = digamma(self.ct) - np.log(self.dt)  # EQ. 33 x
        self.x_w = self.mu  # EQ. 28 x
        self.x_w_sq = np.multiply(self.mu, self.mu) + np.diagonal(self.Sigma)  # Standard, compute yourself
        self.x_w_w_T = self.Sigma + np.outer(self.mu, self.mu)  # EQ. 29, also standard

    def _update_parameters(self):

        self.at = self.a + 0.5  # EQ. 25  x
        self.bt = self.b + self.x_w_sq  # EQ. 25  x
        self.ct = self.c + 0.5 * self.N  # EQ. 26  x

        self.pre_Sigma = []
        self.pre_d = 0
        for i in range(self.N):
            self.pre_Sigma.append(np.outer(self.phi[i], self.phi[i]))
            self.pre_d += self.phi[i] @ self.x_w_w_T @ self.phi[i]

        self.pre_Sigma = np.sum(self.pre_Sigma, axis=0)

        self.Sigma = la.inv(np.diag(self.x_alpha) + self.x_tau * self.pre_Sigma)  # EQ. 23 x
        self.mu = self.x_tau * self.Sigma @ np.sum(np.transpose(np.multiply(self.phi, self.signal)), axis=0)  # EQ. 24 x
        self.dt = self.d + 0.5 * np.sum(np.multiply(self.signal, self.signal)) - self.mu @ np.sum(
            np.multiply(self.phi, self.signal), axis=1) + 0.5 * self.pre_d  # x EQ. 27

    def _compute_ELBO(self):
        self.ln_P_T_X_W_tau = 0.5 * self.N * (self.x_ln_tau - np.log(2 * np.pi)) - self.x_tau * (
                    self.dt - self.d)  # EQ. 40 x
        self.ln_P_w_alpha = -0.5 * (self.N + 1) * np.log(2 * np.pi) - 0.5 * np.sum(self.x_ln_alpha) - 0.5 * (
                    self.x_alpha @ self.x_w_sq)  # EQ. 41 x
        self.ln_P_alpha = (self.N + 1) * self.a * np.log(self.b) + (self.a - 1) * np.sum(
            self.x_ln_alpha) - self.b * np.sum(self.x_alpha) - (self.N + 1) * np.log(gamma(self.a))  # EQ. 42 x
        self.ln_P_tau = self.c * np.log(self.d) + (self.c - 1) * self.x_ln_tau - self.d * self.x_tau - np.log(
            gamma(self.c))  # EQ. 43 x
        self.ln_Q_w = -(0.5 * (self.N + 1) * (1 + np.log(2 * np.pi)) + 0.5 * np.log(la.det(self.Sigma)))  # EQ. 44 x
        self.ln_Q_alpha = -(self.at * np.sum(np.log(self.bt)) + (self.at - 1) * np.sum(
            self.x_ln_alpha) - self.bt @ self.x_alpha - np.sum(gamma(self.at)))  # EQ. 45 x
        self.ln_Q_tau = -(self.ct * np.log(self.dt) + (self.ct - 1) * self.x_ln_tau - self.dt * self.x_tau - np.log(
            gamma(self.ct)))  # EQ. 46 x

        return self.ln_P_T_X_W_tau + self.ln_P_w_alpha + self.ln_P_alpha + self.ln_P_tau - self.ln_Q_w - self.ln_Q_alpha - self.ln_Q_tau  # EQ.39

    def fit_predict(self):

        self._initialize()

        self.elbo = [1]
        training_time = self.max_iter  # self.max_iter
        for trials in range(training_time):
            self._compute_expectations()
            self.elbo.append(self._compute_ELBO())
            self._update_parameters()

            if trials % (self.max_iter / 10) == 0:
                print(f'Evidence lower bound: {self.elbo[-1]:.4f}, step {trials}')
            if np.abs(self.elbo[trials + 1] - self.elbo[trials]) < 10 ** (-15):
                break
        self._get_final_parameters()

    def _get_final_parameters(self):
        self.differences = self.phi @ self.x_w - self.signal
        self.MSE = self.N ** (-1) * np.sum(np.multiply(self.differences, self.differences))
        self.noise_estimation = (self.x_tau) ** (-1)
        self.weight_estimation = (self.x_w)

    def display(self):
        self.xs = np.linspace(-9.5, 10, 300)
        self.kerphi = np.zeros((self.N, len(self.xs)))
        for i in range(self.N):
            for j in range(len(self.xs)):
                self.kerphi[i, j] = self.kernel(self.xs[j], self.input_samples[i], self.sigma_phi)

        if type(self.actual_signal) == type(None):
            plt.figure(figsize=(6, 6))
            plt.plot(self.input_samples, self.phi @ self.x_w, marker='x', color='black')
            plt.plot(self.xs, self.x_w @ self.kerphi, color='red')
            plt.axhline(0, color='black', linewidth=.5)
            plt.axvline(0, color='black', linewidth=.5)
            plt.show()
        else:
            plt.figure(figsize=(6, 6))
            plt.plot(self.input_samples, self.phi @ self.x_w, marker='x', color='black')
            plt.plot(self.input_samples, self.actual_signal, marker='o', color='blue')
            plt.plot(self.xs, self.x_w @ self.kerphi, color='red')
            plt.axhline(0, color='black', linewidth=.5)
            plt.axvline(0, color='black', linewidth=.5)
            plt.savefig('fit.pdf')
            plt.show()

    def display_elbo(self):
        plt.figure(figsize=(6, 6))
        plt.plot(np.arange(1, len(self.elbo)), self.elbo[1:])
        plt.ylabel('ELBO')
        plt.show()


