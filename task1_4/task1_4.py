import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm

# 读取数据
data = pd.read_csv('../dataset/order_train1.csv')

# 获取产品大类和产品小类的唯一值
first_cate_codes = data['first_cate_code'].unique()
second_cate_codes = data['second_cate_code'].unique()

model = ols('ord_qty ~ first_cate_code', data=data).fit()

anova_table = sm.stats.anova_lm(model, typ=2)

# 打印多因素方差分析结果
print(anova_table)

# 进行事后多重比较
posthoc = pairwise_tukeyhsd(data['ord_qty'], data['first_cate_code'])

# 打印事后多重比较结果
print(posthoc)

cnt = 1
# 绘制事后多重比较结果图
posthoc.plot_simultaneous(ylabel='first_cate_code', xlabel='ord_qty')
plt.savefig('{}.png'.format(cnt))
cnt += 1
plt.show()


# 存储分析结果的DataFrame
results = pd.DataFrame(columns=['first_cate_code', 'second_cate_code', 'F-statistic', 'p-value'])

# 遍历产品大类
for first_cate in first_cate_codes:
    # 获取当前产品大类下的数据
    category_data = data[data['first_cate_code'] == first_cate]

    if len(category_data['second_cate_code'].unique()) < 2:
        # 如果产品小类数量小于2，无法进行事后多重比较，跳过该产品大类
        continue

    model = ols('ord_qty ~ second_cate_code', data=category_data).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)

    # 打印多因素方差分析结果
    print(anova_table)

    # 进行事后多重比较
    posthoc = pairwise_tukeyhsd(category_data['ord_qty'], category_data['second_cate_code'])

    # 打印事后多重比较结果
    print(posthoc)

    # 绘制事后多重比较结果图
    posthoc.plot_simultaneous(ylabel='second_cate_code', xlabel='ord_qty')
    plt.title('first_cate_code: {}'.format(first_cate))
    plt.savefig('{}.png'.format(cnt))
    cnt += 1
    plt.show()

# 合并数据，计算每个大类每个小类的总需求量
merged_data = data.groupby(['first_cate_code', 'second_cate_code'])['ord_qty'].sum().reset_index()

# 创建空白图表
plt.figure(figsize=(12, 8))

# 根据大类分组绘制柱状图
sns.barplot(x='first_cate_code', y='ord_qty', hue='second_cate_code', data=merged_data, dodge=False)

plt.xlabel('first_cate_code')
plt.ylabel('Total ord_qty')
plt.title('Total Product Quantity by first_cate_code and second_cate_code')
plt.xticks(rotation=90)

plt.legend(title='second_cate_code', bbox_to_anchor=(1, 1))
plt.savefig('{}.png'.format(cnt))
cnt += 1
plt.show()
