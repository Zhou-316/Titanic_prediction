import pandas as pd
import numpy as np

def preprocess_titanic(df, fit_mode=True, stats=None):
    """
    Args:
        df: 原始 DataFrame
        fit_mode: 是否是训练集（用于计算统计量）
        stats: 如果不是训练集，传入训练集的统计量（如 Age 均值）
    """
    df = df.copy() #copy()函数是1个浅复制方法，创建一个新的DataFrame对象
    #但其中的数据仍然引用原始DataFrame中的数据。这意味着，如果你修改了新DataFrame中的数据，原始DataFrame中的数据也会受到影响。
    # #使用copy()函数可以避免这种情况，确保新DataFrame与原始DataFrame完全独立。
    local_stats = {} # 用于存储当前数据集的统计量（如 Age 均值），如果是训练集则计算并保存，如果是测试集则从传入的 stats 中读取
    #缺失：

    # 1. 处理 Age 缺失
    if fit_mode:
        age_fill = df['Age'].median()  # 用中位数填充
        local_stats['age_fill'] = age_fill
    else:
        age_fill = stats['age_fill']
    df.fillna({'Age': age_fill}, inplace=True)
    
    # 2. 处理 Embarked 缺失
    if fit_mode:
        embarked_fill = df['Embarked'].mode()[0]   #用众数填充
        local_stats['embarked_fill'] = embarked_fill
    else:
        embarked_fill = stats['embarked_fill']
    df.fillna({'Embarked': embarked_fill}, inplace=True)
    
    # 3. Cabin 缺失太多，都设为 'U' (Unknown)
    def extract_cabin_letter(cabin):
        if pd.isna(cabin):
            return 'U'
        cabin_str = str(cabin).strip()
        return cabin_str[0].upper() if cabin_str else 'U'
    df['Cabin'] = df['Cabin'].apply(extract_cabin_letter)

    # 4. 处理 Fare 缺失（测试集中可能有）
    df['Fare'] = pd.to_numeric(df['Fare'], errors='coerce')
    if fit_mode:
        fare_fill = df['Fare'].median()
        local_stats['fare_fill'] = fare_fill
    else:
        fare_fill = stats.get('fare_fill', df['Fare'].median())
    df['Fare'].fillna(fare_fill, inplace=True)
    
    #特征工程：

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # 1. 创建新特征：FamilySize = SibSp（兄弟姐妹&配偶） + Parch（子女） + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int) # 2. 是否独自一人
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False) # 3. 从 Name 中提取 Title（Mr, Miss, Mrs, Master...）
    # 合并title，最终包含Mr, Miss, Mrs, Master 和 Rare（其他稀有称谓）
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(title_map).fillna('Rare')
    
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    df.drop(columns=cols_to_drop, errors='ignore', inplace=True)


# One-Hot 编码：对 Sex、Cabin、Embarked、Title 进行 One-Hot 编码
    categorical_columns = ['Sex', 'Cabin', 'Embarked', 'Title']

# 只对实际存在的列做 One-Hot（避免 KeyError）
    existing_categorical = [col for col in categorical_columns if col in df.columns]


    if existing_categorical:
        df = pd.get_dummies(df, columns=existing_categorical, drop_first=True)


    return df, local_stats