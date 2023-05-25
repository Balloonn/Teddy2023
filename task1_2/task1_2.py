import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('../dataset/order_train1.csv')

# 创建多因素方差分析模型
model = ols('ord_qty ~ sales_region_code', data=data).fit()

# 执行多因素方差分析
anova_table = sm.stats.anova_lm(model, typ=2)

# 打印多因素方差分析结果
print(anova_table)

# 进行事后多重比较
posthoc = pairwise_tukeyhsd(data['ord_qty'], data['sales_region_code'])

# 打印事后多重比较结果
print(posthoc)

# 绘制事后多重比较结果图
posthoc.plot_simultaneous(ylabel='sales_region_code', xlabel='ord_qty')
plt.savefig('1.png')
plt.show()
