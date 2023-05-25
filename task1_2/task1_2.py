import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('../dataset/order_train1.csv')

sales_regions = data['sales_region_code'].unique()

result = stats.f_oneway(*[data[data['sales_region_code'] == region]['ord_qty'] for region in sales_regions])

# 打印方差分析结果
print("F-statistic:", result.statistic)
print("p-value:", result.pvalue)

# 绘制箱线图
sns.set()
plt.figure(figsize=(12, 8))
sns.boxplot(x='sales_region_code', y='ord_qty', data=data)
plt.xlabel('sales_region_code')
plt.ylabel('ord_qty')
plt.title('Boxplot of ord_qty by sales_region_code')
plt.savefig('1.png')
plt.show()

bins = [0, 50, 200, 500, 20000]
labels = ['0-50', '50-200', '200-500', '>500']
data['ord_qty_range'] = pd.cut(data['ord_qty'], bins=bins, labels=labels)

grouped = data.groupby(['sales_region_code', 'ord_qty_range'])
count = grouped.size().unstack()

count_long = count.stack().reset_index(name='count')

sns.set()
plt.figure(figsize=(12, 8))
sns.barplot(x='sales_region_code', y='count', hue='ord_qty_range', data=count_long)
plt.xlabel('sales_region_code')
plt.ylabel('count')
plt.title('Distribution of ord_qty by Sales Region')

plt.savefig('2.png')
plt.show()
