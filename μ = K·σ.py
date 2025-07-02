import numpy as np
from scipy.integrate import simpson

# 定义 t 空间
t = np.linspace(1e-6, 20, 5000)

# 概率密度
def p_joint(t, K, sigma):
    mu = K * sigma  # 核心耦合形式
    pk = K * t**(K-1) * np.exp(-t)
    fs = np.exp(-((t - mu)**2) / (2 * sigma**2))
    p = pk * fs
    return p / simpson(p, t)

# 熵函数
def entropy(p, t):
    p = np.clip(p, 1e-12, None)
    return -simpson(p * np.log(p), t)

# 混合偏导数计算（中心差分）
def mixed_partial(K0, sigma0, h=1e-3):
    def H(K, sigma):
        p = p_joint(t, K, sigma)
        return entropy(p, t)

    H_pp = H(K0 + h, sigma0 + h)
    H_pm = H(K0 + h, sigma0 - h)
    H_mp = H(K0 - h, sigma0 + h)
    H_mm = H(K0 - h, sigma0 - h)

    return (H_pp - H_pm - H_mp + H_mm) / (4 * h**2)

# 计算在中心点的混合耦合强度
K_center = 1.0
sigma_center = 0.5
coupling_strength = mixed_partial(K_center, sigma_center)
print(f"在 μ = K·σ 的耦合结构中，混合偏导数 ∂²H/∂K∂σ ≈ {coupling_strength}")
