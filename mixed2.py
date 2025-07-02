import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# 设置更大的时间范围与更多点数以提高积分精度
t = np.linspace(1e-6, 20, 5000)

# 构造耦合结构核 p(t; K, σ)
def p_joint(t, K, sigma):
    # 可调耦合中心位置
    mu = 2 * K
    pk = K * t**(K - 1) * np.exp(-t)
    fs = np.exp(-((t - mu)**2) / (2 * sigma**2))
    p = pk * fs
    return p / simpson(p, t)

# 定义熵计算
def entropy(p, t):
    p = np.clip(p, 1e-12, None)
    return -simpson(p * np.log(p), t)

# 定义联合熵函数 H(K, σ)
def H_joint(K, sigma):
    p = p_joint(t, K, sigma)
    return entropy(p, t)

# 使用中心差分法计算混合偏导数 ∂²H / ∂K∂σ
def mixed_partial_H(K, sigma, h=1e-4):
    Hpp = H_joint(K + h, sigma + h)
    Hpm = H_joint(K + h, sigma - h)
    Hmp = H_joint(K - h, sigma + h)
    Hmm = H_joint(K - h, sigma - h)
    return (Hpp - Hpm - Hmp + Hmm) / (4 * h**2)

# 计算中心点的混合偏导数
K0 = 1.0
sigma0 = 0.5
mixed_derivative = mixed_partial_H(K0, sigma0, h=1e-4)

mixed_derivative
