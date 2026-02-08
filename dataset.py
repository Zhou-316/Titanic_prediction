import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder  
# sklearn 的 LabelEncoder 能自动把字符串（如 'male', 'female'）映射成连续整数（0, 1, 2...）

import warnings
warnings.filterwarnings("ignore")
# warnings 模块用于控制 Python 中的警告信息输出，filterwarnings("ignore") 可以忽略所有警告，避免在运行过程中看到不必要的警告信息。

class TitanicDataset(Dataset):
    def __init__(self, df, is_test=False, label_encoders=None):
        self.is_test = is_test
        
        # 分离标签（训练集才有 Survived）
        if not is_test:
            self.labels = torch.tensor(df['Survived'].values, dtype=torch.long)
            feature_df = df.drop(columns=['Survived'])
        else:
            self.labels = None
            feature_df = df
        
        # 对类别变量进行 Label Encoding（训练时拟合，测试时用训练的 encoder）
        cat_cols = ['Sex', 'Embarked', 'Cabin', 'Title']
        self.label_encoders = label_encoders or {}
        
        encoded_data = []
        for col in feature_df.columns:
            if col in cat_cols:
                if col not in self.label_encoders:
                    le = LabelEncoder()
                    encoded = le.fit_transform(feature_df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # 测试集：用训练集的 encoder，未知类别设为 -1
                    le = self.label_encoders[col]
                    encoded = []
                    for val in feature_df[col]:
                        try:
                            encoded.append(le.transform([str(val)])[0])
                        except ValueError:
                            encoded.append(-1)  # 未知类别
                    encoded = np.array(encoded)
                encoded_data.append(encoded)
            else:
                # 数值特征直接使用
                encoded_data.append(feature_df[col].values)
        
        # 转为 float32 张量
        self.features = torch.tensor(np.column_stack(encoded_data), dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.is_test:
            return self.features[idx]
        else:
            return self.features[idx], self.labels[idx]