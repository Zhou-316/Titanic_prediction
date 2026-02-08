import pandas as pd
from data_preprocess import preprocess_titanic
# 加载训练集
df = pd.read_csv('train.csv')
processed_train, stats = preprocess_titanic(df, fit_mode=True)
# 1. 查看每列的缺失数量和比例
missing = processed_train.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_info = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct.round(2)
})

print("train.csv 缺失值统计：")
print(missing_info[missing_info['Missing Count'] > 0])  # 只显示有缺失的列
'''
df2 = pd.read_csv('test.csv')
# 1. 查看每列的缺失数量和比例
missing = df2.isnull().sum()
missing_pct = (missing / len(df2)) * 100

missing_info = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct.round(2)
})

print("test.csv 缺失值统计：")
print(missing_info[missing_info['Missing Count'] > 0])  # 只显示有缺失的列
'''