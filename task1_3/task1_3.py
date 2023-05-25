import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('../dataset/order_train1.csv')

# 按销售方式分组
online_data = data[data['sales_chan_name'] == 'online']
offline_data = data[data['sales_chan_name'] == 'offline']

# 绘制online销售方式的需求量分布图
x_range = (0, 300)
plt.figure(figsize=(8, 6))
sns.histplot(data=online_data, x='ord_qty', kde=True)
plt.xlim(x_range)
plt.xlabel('Order Quantity')
plt.ylabel('Count')
plt.title('Distribution of Order Quantity (Online Sales)')
plt.savefig('1.png')
plt.show()

# 绘制offline销售方式的需求量分布图
plt.figure(figsize=(8, 6))
sns.histplot(data=offline_data, x='ord_qty', kde=True)
plt.xlim(x_range)
plt.xlabel('Order Quantity')
plt.ylabel('Count')
plt.title('Distribution of Order Quantity (Offline Sales)')
plt.savefig('2.png')
plt.show()


# 正态性检验
_, pvalue_online = stats.normaltest(online_data['ord_qty'])
_, pvalue_offline = stats.normaltest(offline_data['ord_qty'])

# 打印正态性检验结果
print("Normality test - online: p-value =", pvalue_online)
print("Normality test - offline: p-value =", pvalue_offline)

# Mann-Whitney U检验
statistic, p_value = stats.mannwhitneyu(online_data['ord_qty'], offline_data['ord_qty'], alternative='two-sided')

print("Mann-Whitney U statistic:", statistic)
print("Mann-Whitney U p-value:", p_value)

# 绘制箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['sales_chan_name'], y=data['ord_qty'])
plt.xlabel('Sales Channel')
plt.ylabel('Order Quantity')
plt.title('Boxplot of Order Quantity by Sales Channel')

# 显示 Mann-Whitney U检验结果
plt.text(0.5, 0.9, f"Mann-Whitney U statistic: {statistic:.2f}\nMann-Whitney U p-value: {p_value:.2f}",
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.savefig('3.png')
plt.show()

# 绘制箱线图
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['sales_chan_name'], y=data['ord_qty'])
plt.xlabel('Sales Channel')
plt.ylabel('Order Quantity')
plt.title('Boxplot of Order Quantity by Sales Channel')

# 显示 Mann-Whitney U检验结果
plt.text(0.5, 0.9, f"Mann-Whitney U statistic: {statistic:.2f}\nMann-Whitney U p-value: {p_value:.2f}",
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.ylim(0, 1000)
plt.savefig('4.png')
plt.show()
