import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import os
import scipy
matplotlib.use('TkAgg')
# 斯皮尔曼系数

data = pd.read_csv('../dataset/order_train1.csv')

print(data.head())

print(data['sales_region_code'].nunique())

print(data['item_code'].nunique())

print(data['first_cate_code'].nunique())

print(data['second_cate_code'].nunique())

filename = 'significance_tt.csv'

if not os.path.exists(filename):
    tt = []
    cnt = 0
    item_codes = []
    for item_code in data['item_code']:
        item_codes.append(item_code)
    item_codes = list(set(item_codes))

    for item_code in item_codes:
        tmp = data[data['item_code'] == item_code]
        corr, p_value = scipy.stats.spearmanr(tmp['item_price'], tmp['ord_qty'])
        tt.append((item_code, corr, p_value))
        cnt += 1
        print(cnt)

    tt = pd.DataFrame(tt)
    tt = tt.rename(columns={0: 'item_code', 1: 'corr', 2: 'p_value'})
    significance_level = 0.05
    significance_tt = tt[tt['p_value'] < significance_level]

    significance_tt = pd.DataFrame(significance_tt)
    significance_tt = significance_tt.rename(columns={0: 'item_code', 1: 'corr', 2: 'p_value'})
    significance_tt.to_csv('significance_tt.csv', index=False)

significance_tt = pd.read_csv('significance_tt.csv')
sns.set()
sns.histplot(significance_tt['corr'], kde=True)
plt.xlabel('Correlation')
plt.ylabel('Frequency')
plt.title('Distribution of Correlation Coefficients')
plt.savefig('1.png')
plt.show()
