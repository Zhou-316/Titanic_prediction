import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import TitanicDataset
from data_preprocess import preprocess_titanic

train_df = pd.read_csv('train.csv')   
test_df = pd.read_csv('test.csv')
# 1. 定义模型
class TitanicNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 32)
        self.fc2 = torch.nn.Linear(32, 8)
        self.fc3 = torch.nn.Linear(8, 2)  
        self.dropout = torch.nn.Dropout(0.2) ## dropout (0.3) 可以防止过拟合，随机丢弃 30% 的神经元
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# 2. 创建数据集和数据加载器，训练模型
if __name__ == "__main__":
    processed_train, stats = preprocess_titanic(train_df, fit_mode=True)
    print("Columns in processed_train:", processed_train.columns.tolist())
    train_dataset = TitanicDataset(processed_train)
    print("Features contains NaN:", torch.isnan(train_dataset.features).any())
    print("Features contains Inf:", torch.isinf(train_dataset.features).any())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = TitanicNet(input_dim=train_dataset.features.shape[1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#Adam 优化器是一种自适应学习率优化算法，结合了动量和RMSProp的优点，适用于大多数深度学习任务。
    losses=[]
    for epoch in range(1000):
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader,0):      #i代表第几个batch
            inputs, labels = inputs.to(device), labels.to(device)  #将数据移动到GPU
            y_pred=model(inputs)  
            loss=criterion(y_pred,labels)  
            epoch_loss += loss.item()
            #每十轮打印一次损失
            if epoch % 10 == 0 and i == len(train_loader) - 1:
                avg_loss = epoch_loss / len(train_loader)
                losses.append(avg_loss)
                print(f'Epoch {epoch+1}/1000 Loss: {avg_loss:.4f}')
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()
    # 绘制训练损失曲线
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, 'b', label='Training loss')
    plt.title('Training Loss per ten Epoch')
    plt.xlabel('Epochs(*10)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    feature_columns = [col for col in processed_train.columns if col != 'Survived']
    torch.save({
    'model_state_dict': model.state_dict(),
    'feature_columns': feature_columns,
    'stats': stats  
}, 'titanic_model.pth')
    print("模型已保存，特征列为:", feature_columns)