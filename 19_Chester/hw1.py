import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_clt_ultimate(m_simulations=10000, ns=(1, 5, 30, 100)):
    # 設定三種不同的母體分佈
    # 1. Uniform (0, 1) -> Mu=0.5, Var=1/12
    # 2. Exponential (lambda=2) -> Mu=0.5, Var=0.25
    # 3. Bernoulli (p=0.3) -> Mu=0.3, Var=0.21
    
    distributions = [
        {'name': 'Uniform', 'func': lambda n, m: np.random.uniform(0, 1, (m, n)), 'mu': 0.5, 'var': 1/12},
        {'name': 'Exponential', 'func': lambda n, m: np.random.exponential(0.5, (m, n)), 'mu': 0.5, 'var': 0.25},
        {'name': 'Bernoulli', 'func': lambda n, m: np.random.binomial(1, 0.3, (m, n)), 'mu': 0.3, 'var': 0.3*0.7}
    ]

    fig, axes = plt.subplots(len(distributions), len(ns), figsize=(15, 10), sharey='row')
    fig.suptitle(f"The Magic of Central Limit Theorem (Simulations: {m_simulations})", fontsize=16, fontweight='bold')

    for i, dist in enumerate(distributions):
        for j, n in enumerate(ns):
            ax = axes[i, j]
            
            # 1. 抽樣並計算平均值
            data = dist['func'](n, m_simulations)
            sample_means = data.mean(axis=1)
            
            # 2. 畫出樣本平均值的直方圖
            ax.hist(sample_means, bins=50, density=True, color='skyblue', alpha=0.7, edgecolor='white')
            
            # 3. 計算理論上的常態分佈 (CLT 預測)
            # 根據 CLT: Mean = mu, Std = sqrt(var / n)
            theory_mu = dist['mu']
            theory_std = np.sqrt(dist['var'] / n)
            
            x = np.linspace(min(sample_means), max(sample_means), 100)
            y = norm.pdf(x, theory_mu, theory_std)
            ax.plot(x, y, 'r-', lw=2, label='Normal Fit' if j == 3 else "")
            
            # 標題與格式
            if i == 0: ax.set_title(f"Sample Size n = {n}", fontsize=12, fontweight='bold')
            if j == 0: ax.set_ylabel(dist['name'], fontsize=12, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_clt_ultimate()
