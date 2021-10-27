# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import norm
from scipy.stats import multivariate_normal
import mglearn
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split


#generate response z
def generate_z(x, beta):
    m = np.array([0 for i in range(N)])
    epsilon = np.random.multivariate_normal(m, np.eye(N))
    return x @ beta + epsilon

# compute residuals
def residuals(x, z, sigma):
    beta_LS = np.linalg.inv(x.T @ x) @ x.T @ z
    e_hat = z - x @ beta_LS
    eps = 1e-13
    I = np.eye(p)
    h = np.diag(x @ np.linalg.inv(x.T @ x + eps * I) @ x.T)
    epsilon = list()
    for i in range(N):
        epsilon.append(e_hat[i] / math.sqrt(1 - h[i]))
    return (beta_LS,epsilon)

# bootstrap of epsilon
def scaleboot(x, n):
    l = np.array([0.0] * n)
    for i in range(n):
        l[i] = random.choice(x)
    return l

# compute z_star
def C_z_star(x, y, sigma):
    (beta_LS,epsilon) = residuals(x, y, sigma)
    epsilon_star = scaleboot(epsilon, N)
    return x @ beta_LS + math.sqrt(sigma) * epsilon_star

# z-value for region S
def C_alpha_S(x, z, sigma, j, s_j):
    t = [0.0 for i in range(length)]
    for i in range(nb):
        for k in range(length):
            z_star = C_z_star(x, z, sigma[k])
            #M_hat,s_hat = A_LASSO(x, z_star, 1)
            (M_hat, s_hat) = solve_LASSO(x, z_star, 10, maxiter=30000, tol=1e-5)
            if j in M_hat and s_hat[j] == s_j:
                t[k] += 1/nb
    return t

# LASSO
def A_LASSO(x, z, lamda):
    lasso = Lasso(alpha = lamda).fit(x, z)
    beta = lasso.coef_
    s_hat = np.sign(beta)
    M_hat = set()
    for i in range(p):
        if beta[i] != 0: M_hat.add(i)
    return M_hat, s_hat

# LASSO(another method)
def solve_LASSO(X, z, alpha, maxiter=100, tol=1.0e-4):
    x0 = np.zeros(X.shape[1])
    rho = np.max(np.sum(np.abs(X.T @ X), axis=0))
    x = []
    for i in range(maxiter):
        res = z - X @ x0
        y = x0 + (X.T @ res) / rho
        x_new = np.sign(y) * np.maximum(np.abs(y) - alpha / rho, 0.0)
        if (np.abs(x0 - x_new) < tol).all(): 
            s_hat = np.sign(x_new)
            M_hat = set()
            for j in range(p):
                if (x_new[j] != 0): M_hat.add(j)
            return M_hat,s_hat
        x0 = x_new
    raise ValueError('Not converged')

def solve_l(X,z,alpha,maxiter=100,tol=1.0e-4):
    x0 = np.zeros(X.shape[1])
    rho = np.max(np.sum(np.abs(X.T @ X), axis=0))
    x = []
    for i in range(maxiter):
        res = z - X @ x0
        y = x0 + (X.T @ res) / rho
        x_new = np.sign(y) * np.maximum(np.abs(y) - alpha / rho, 0.0)
        if (np.abs(x0 - x_new) < tol).all(): 
            return x_new
        x0 = x_new
    raise ValueError('Not converged')

#Marginal Screening
def solve_MS(X, y, k):
    z = np.array([X.T[i] @ y for i in range(p)])
    l = [i for i in zip(np.abs(z), np.sign(z), range(p))]
    l.sort(reverse=True)
    M_hat = set(l[i][2] for i in range(k))
    s_hat = np.array([l[i][1] for i in range(p)])
    return M_hat, s_hat

# bootstrap replicate
nb = 10000
# sigma
sa = 9 ** np.linspace(-1, 1, 13)
length = len(sa)

#theta = 2.0,各特徴に対してp_SIを計算
rnorm = np.random.normal(0, 1, N * p)
X = rnorm.reshape(N, p)
A = np.linalg.inv(X.T @ X) @ X.T
beta = np.array([0 for i in range(p)])
for i in range(5):
    beta[i] = 2
for j in range(p): 
    z = generate_z(X,beta)
    alpha_sigma_S = np.array(C_alpha_S(X,z,sa,j,1))
    psi_sigma_S = np.sqrt(sa) * norm.ppf(1-alpha_sigma_S,0,1)
    (beta_S_1,beta_S_0) = np.polyfit(sa,psi_sigma_S,1)
    z_H = (A[j] @ z) / np.linalg.norm(A[j],2)
    z_S = beta_S_0
    p_SI = (1-norm.cdf(z_H,0,1)) / (1-norm.cdf(z_H+z_S,0,1))
    print(j,p_SI)

#theta=2.0の時のR_j / nb(N_j:jが選択された回数,R_j:帰無仮設(beta_j == 0)が棄却された回数)
N=50
p=25
rnorm = np.random.normal(0, 1, N * p)
X = rnorm.reshape(N, p)
A = np.linalg.inv(X.T @ X) @ X.T
beta = np.array([0 for i in range(p)])
for i in range(5):
    beta[i] = 2
N_j = np.array([0] * p)
R_j = np.array([0] * p)
for i in range(nb):
    z = generate_z(X, beta)
    z_star = C_z_star(X,z,4)
    (M_hat, s_hat) = solve_LASSO(X, z_star, 10, maxiter=30000, tol=1e-5)
    for j in range(p):
        if j in M_hat and s_hat[j] == 1:
            N_j[j] += 1
            X_star = np.array([X.T[k] for k in M_hat]).T
            x = solve_l(X_star, z, 10, maxiter=30000, tol=1e-5)
            if  x[list(M_hat).index(j)] != 0: R_j[j] += 1
R_j / nb

# compute TPR and FPR
N = 50
p = 10
beta = np.array([0.0 for i in range(p)])
rnorm = np.random.normal(0, 1, N*p)
X = rnorm.reshape(N, p)
A = np.linalg.inv(X.T @ X) @ X.T
theta = np.linspace(0.5, 2.0, 16)

# TPR(j = 0)
s0 = np.array([0] * len(theta))
for i in range(len(theta)):
    for k in range(5):
        beta[k] = round(theta[i], 1)
    for it in range(nb):
        z = generate_z(X, beta)
        z_star = C_z_star(X, z, 2)
        (M_hat, s_hat) = solve_LASSO(X, z_star, 10, maxiter=30000, tol=1e-5)
        if 0 in M_hat and s_hat[0] == 1:
            #X_star = np.array([X.T[k] for k in M_hat]).T
            x = solve_l(X, z, 10, maxiter=30000, tol=1e-5)
            if  x[list(M_hat).index(0)] != 0: s0[i] += 1
s0 / nb

# FPR(j = 6)
s6 = np.array([0] * len(theta))
for i in range(len(theta)):
    N6 = 0
    R6 = 0
    for k in range(5):
        beta[k] = round(theta[i], 1)
    for it in range(nb):
        z = generate_z(X, beta)
        z_star = C_z_star(X, z, 2)
        (M_hat, s_hat) = solve_LASSO(X, z_star, 10, maxiter=30000, tol=1e-5)
        if 6 in M_hat and s_hat[6] == 1:
            #X_star = np.array([X.T[k] for k in M_hat]).T
            x = solve_l(X, z, 10, maxiter=30000, tol=1e-5)
            if  x[list(M_hat).index(6)] != 0: s6[i] += 1
s6 / nb
