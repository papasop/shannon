# ✅ 第一步：导入依赖
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.special import psi, gammaln
import matplotlib.pyplot as plt
from numpy import log as ln

# ✅ 第二步：设置扫描参数
K_vals = np.linspace(0.5, 1.5, 21)  # K从0.5到1.5
σ = 1.0
t_vals = np.linspace(1e-3, 50, 100000)  # 避免 t→0 发散
dt = t_vals[1] - t_vals[0]

# ✅ 第三步：准备容器
records = []

# ✅ 第四步：逐个K值验证所有公式
for K in K_vals:
    # 概率密度函数 p_K(t)
    p_K = K * t_vals**(K - 1) * np.exp(-t_vals)
    dpdt = p_K * ((K - 1)/t_vals - 1)
    
    # ✅ 公式 1：数值结构熵 H(t)
    H_t = -simpson(p_K * ln(p_K + 1e-20), t_vals)
    
    # ✅ 公式 2：修正结构熵导数 dH/dt
    dHdt = -simpson(dpdt * (ln(p_K + 1e-20) + 1), t_vals)
    
    # ✅ 公式 3：结构熵理论 H(K)
    H_K_theory = gammaln(K) + (1 - K) * psi(K) + K
    
    # ✅ 公式 4：误差一致性检验
    error_Ht_HK = abs(H_t - H_K_theory)
    error_drift_consistency = abs(dHdt + H_t)

    # ✅ 公式 5：联合结构熵 H(K, σ)
    f_sigma = np.exp(-t_vals**2 / (2 * σ**2))
    joint_density = p_K * f_sigma
    H_joint = -simpson(joint_density * ln(joint_density + 1e-20), t_vals)
    norm_joint = simpson(joint_density, t_vals)

    # ✅ 存储结果
    records.append({
        "K": K,
        "H(t) 数值": H_t,
        "dH/dt": dHdt,
        "H(K) 理论": H_K_theory,
        "误差1: |H_t - H_K|": error_Ht_HK,
        "误差2: |dH/dt + H(t)|": error_drift_consistency,
        "联合熵 H(K,σ)": H_joint,
        "联合密度归一性": norm_joint
    })

# ✅ 第五步：展示表格
df = pd.DataFrame(records)
pd.set_option("display.float_format", lambda x: f"{x:0.6f}")
print(df)

# ✅ 第六步：误差可视化
plt.figure(figsize=(10, 6))
plt.plot(df["K"], df["误差1: |H_t - H_K|"], label="误差1: |H(t) - H(K)|")
plt.plot(df["K"], df["误差2: |dH/dt + H(t)|"], label="误差2: |dH/dt + H(t)|")
plt.axvline(1.0, color='gray', linestyle='--', label='K = 1 吸引子')
plt.yscale("log")
plt.xlabel("K 值")
plt.ylabel("误差（对数尺度）")
plt.title("公式 1-4 一致性鲁棒验证（结构吸引子测试）")
plt.grid(True)
plt.legend()
plt.show()

# ✅ 第七步：联合熵归一性可视化（验证公式 5 是否单位一致）
plt.figure(figsize=(10, 6))
plt.plot(df["K"], df["联合密度归一性"], label="∫p_K * f_σ(t)")
plt.axhline(1.0, color='gray', linestyle='--', label='理论归一：1')
plt.xlabel("K 值")
plt.ylabel("联合密度归一性")
plt.title("公式 5：联合结构熵密度归一性验证")
plt.grid(True)
plt.legend()
plt.show()
