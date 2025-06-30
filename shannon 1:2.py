# ✅ 第一步：导入依赖
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from numpy import log as ln
import matplotlib.pyplot as plt

# ✅ 第二步：设置参数
K = 1.0  # 固定时间吸引子
sigma_vals = np.linspace(0.1, 5, 50)  # 扫描谱域扩展 σ
t_vals = np.linspace(1e-3, 50, 100000)  # 时间区间

# ✅ 第三步：初始化记录容器
records = []

# ✅ 第四步：计算联合结构熵 H(K, σ) 与其时间导数项 dH/dt
for σ in sigma_vals:
    # 时间密度分布 p_K(t)，对应 K=1 指数衰减
    p_K = np.exp(-t_vals)

    # 谱域调制项 f_σ(t)，高斯谱域修饰
    f_sigma = np.exp(-t_vals**2 / (2 * σ**2))

    # 联合密度（未归一化）
    joint_density = p_K * f_sigma

    # 使用 Simpson 积分归一化联合密度
    norm = simpson(joint_density, t_vals)
    joint_density /= norm

    # ✅ A2：联合结构熵 H(K, σ)
    H_joint = -simpson(joint_density * ln(joint_density + 1e-20), t_vals)

    # ✅ A3：时间导数熵项 dH/dt（已修正：去掉负号）
    dpdt = -p_K
    dHdt = simpson(dpdt * ln(p_K + 1e-20), t_vals)

    records.append({
        "σ": σ,
        "A2: H(K, σ)": H_joint,
        "A3: dH/dt": dHdt
    })

# ✅ 第五步：转为 DataFrame 展示
df = pd.DataFrame(records)
print(df.head())

# ✅ 第六步：可视化结构熵与熵导数
plt.figure(figsize=(10, 6))
plt.plot(df["σ"], df["A2: H(K, σ)"], label="A2: 联合结构熵")
plt.plot(df["σ"], df["A3: dH/dt"], label="A3: 熵导数")
plt.xlabel("σ（谱域扩展）")
plt.ylabel("数值")
plt.title("A2 + A3 联合结构熵分析（K=1 吸引子）")
plt.legend()
plt.grid(True)
plt.show()
