import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson  # 替代 trapz

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

# 网格参数
K_vals = np.linspace(0.6, 1.4, 9)
sigma_vals = np.linspace(0.5, 2.0, 9)

delta_H = np.zeros((len(K_vals), len(sigma_vals)))

# 计算联合熵与分量熵差值 ΔH = H_joint - (H_K + H_sigma)
for i, K in enumerate(K_vals):
    pk = p_K(t, K)
    Hk = entropy(pk, t)
    for j, sigma in enumerate(sigma_vals):
        fs = f_sigma(t, sigma)
        Hs = entropy(fs, t)

        # 构造联合密度 p_joint(t1, t2) = p_K(t1) * f_sigma(t2)
        joint = np.outer(pk, fs)
        joint = joint / simpson(simpson(joint, t), t)  # 双重归一化

        # 计算联合熵
        joint_clipped = np.clip(joint, 1e-12, None)
        H_joint = -simpson(simpson(joint_clipped * np.log(joint_clipped), t), t)

        delta_H[i, j] = H_joint - Hk - Hs  # 耦合项

# 可视化输出 ΔH
df = pd.DataFrame(delta_H, index=np.round(K_vals, 2), columns=np.round(sigma_vals, 2))
print("联合熵耦合 ΔH (单位：nat):")
print(df)

# 绘制热力图
plt.figure(figsize=(8, 6))
plt.imshow(df.values, cmap='coolwarm', origin='lower', extent=[0.5, 2.0, 0.6, 1.4], aspect='auto')
plt.colorbar(label='ΔH = H_joint - (H_K + H_σ)')
plt.title('Joint Entropy Coupling ΔH Heatmap')
plt.xlabel('σ')
plt.ylabel('K')
plt.grid(False)
plt.show()

