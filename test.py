import torch
import pandas as pd
import numpy as np
from dataset import TitanicDataset
from data_preprocess import preprocess_titanic
from train import TitanicNet  
import torch.nn as nn

if __name__ == "__main__":
    # 2. 加载训练时保存的信息
    checkpoint = torch.load('titanic_model.pth', weights_only=False)
    feature_columns = checkpoint['feature_columns']
    stats = checkpoint['stats']
    
    # 3. 加载并预处理测试集
    test_df = pd.read_csv('test.csv')
    processed_test, _ = preprocess_titanic(test_df, fit_mode=False, stats=stats)
    print("Processed test columns:", processed_test.columns.tolist())
    
    # 4. 关键：确保测试集特征列与训练集完全一致
    # 添加训练集中有但测试集中缺失的列（填0）
    for col in feature_columns:
        if col not in processed_test.columns:
            processed_test[col] = 0
            print(f"Added missing column: {col}")
    
    # 按训练集的列顺序重新排列，并只保留这些列
    processed_test = processed_test[feature_columns]
    print(f"Final test feature shape: {processed_test.shape}")
    
    # 5. 创建测试数据集（注意：测试集没有 'Survived' 列）
    test_dataset = TitanicDataset(processed_test, is_test=True)
    
    # 6. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TitanicNet(input_dim=len(feature_columns))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 设置为评估模式（关闭 Dropout）
    
    # 7. 预测
    predictions = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            inputs = test_dataset[i].unsqueeze(0).to(device)  # 添加 batch 维度
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            predictions.append(pred.item())
    
    # 8. 生成提交文件
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(f"Submission shape: {submission.shape}")
    print(submission.head())