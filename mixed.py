import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# 定义 t 取值范围
t = np.linspace(1e-3, 10, 1000)

# 定义结构核 p_K(t)
def p_K(t, K):
    pk = K * t**(K-1) * np.exp(-t)
    return pk / simpson(pk, t)

# 定义高斯核 f_σ(t)
def f_sigma(t, sigma):
    fs = np.exp(-t**2 / (2 * sigma**2))
    return fs / simpson(fs, t)

# 定义熵计算公式（natural log）
def entropy(p, t):
    p = np.clip(p, 1e-12, None)  # 避免 log(0)
    return -simpson(p * np.log(p), t)

# 定义联合熵 H_joint(K, sigma)
def joint_entropy(K, sigma):
    pk = p_K(t, K)
    fs = f_sigma(t, sigma)
    joint = np.outer(pk, fs)
    joint /= simpson(simpson(joint, t), t)  # 双重归一化
    joint = np.clip(joint, 1e-12, None)
    return -simpson(simpson(joint * np.log(joint), t), t)

# 定义 ΔH(K, σ) = H_joint - H_K - H_σ
def delta_H(K, sigma):
    pk = p_K(t, K)
    fs = f_sigma(t, sigma)
    Hk = entropy(pk, t)
    Hs = entropy(fs, t)
    Hjoint = joint_entropy(K, sigma)
    return Hjoint - Hk - Hs

# 使用中心差分计算混合偏导数 ∂²H/∂K∂σ
def mixed_derivative(K0, sigma0, h=1e-2):
    dH_dK_plus = (delta_H(K0 + h, sigma0 + h) - delta_H(K0 - h, sigma0 + h)) / (2 * h)
    dH_dK_minus = (delta_H(K0 + h, sigma0 - h) - delta_H(K0 - h, sigma0 - h)) / (2 * h)
    return (dH_dK_plus - dH_dK_minus) / (2 * h)

# 计算在 (K=1, σ=0.5) 处的混合偏导
K_target = 1.0
sigma_target = 0.5
mixed_d2H = mixed_derivative(K_target, sigma_target)

mixed_d2H
