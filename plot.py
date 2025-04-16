import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

# 读取数据
df = pd.read_csv('exploitability.txt')

# 创建图形
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['exploitability'], 
         marker='o', linestyle='-', linewidth=2, markersize=8)

# 设置坐标轴标签
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Exploitability', fontsize=12)

# 设置纵轴对数刻度
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
plt.gca().set_ylim(top=1, bottom=0.001)  # 根据数据范围调整纵轴范围

# 设置网格线
plt.grid(True, which='both', linestyle='--', alpha=0.7)

# 设置横轴刻度
plt.xticks(range(0, 121, 20))  # 每20个epoch显示一个刻度

# 保存为矢量图
plt.savefig('exploitability_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()