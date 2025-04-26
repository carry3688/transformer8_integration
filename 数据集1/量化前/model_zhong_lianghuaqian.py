import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import ipaddress
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os


import torch.nn.functional as F

# 添加在main()函数之前的辅助函数
def save_quantized_data(dataset, indices, filename, feature_columns, label_column):
    """
    将指定索引的数据量化并保存为CSV文件
    
    Args:
        dataset: TimeSeriesDataset对象
        indices: 要提取的数据索引
        filename: 输出CSV文件名
        feature_columns: 特征列名列表
        label_column: 标签列名
    """
    # 获取指定索引的数据
    data_list = []
    for idx in indices:
        features, label = dataset[idx]
        
        # 对每个时间步的数据进行处理
        for t in range(features.shape[0]):
            # 提取当前时间步的特征和标签
            row_features = features[t].numpy()
            row_label = label[t][0].item()
            
            # 将特征和标签组成一行数据
            row = np.append(row_features, row_label)
            data_list.append(row)
    
    # 创建DataFrame
    column_names = feature_columns + [label_column]
    df = pd.DataFrame(data_list, columns=column_names)
    
    # 保存为CSV
    df.to_csv(filename, index=False)
    print(f"量化后的数据已保存到: {filename}")
    return df

# ----------------------
# 手搓 LayerNorm
# ----------------------
class ManualLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(ManualLayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta

# ----------------------
# 手搓 Transformer Encoder Block（2-head attention + FFN）
# ----------------------
class SimpleEncoder(nn.Module):
    def __init__(self, d_model=6, nhead=2):
        super(SimpleEncoder, self).__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        # 两头的 Wq, Wk, Wv，分别是 (6, 3)
        self.wq = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(nhead)])
        self.wk = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(nhead)])
        self.wv = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(nhead)])

        # Wo: 输出的线性映射，拼接后 6 → 6
        self.wo = nn.Linear(d_model, d_model)

        # 第一层 Add & Norm
        self.norm1 = ManualLayerNorm(d_model)

        # FFN 层：6 → 12 → 6
        self.ff1 = nn.Linear(d_model, d_model * 2)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_model * 2, d_model)

        # 第二层 Add & Norm
        self.norm2 = ManualLayerNorm(d_model)

    def forward(self, x):
        # x: (B, 8, 6)
        attn_outputs = []
        for i in range(self.nhead):
            q = self.wq[i](x)  # (B, 8, 3)
            k = self.wk[i](x)  # (B, 8, 3)
            v = self.wv[i](x)  # (B, 8, 3)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, 8, 8)
            weights = F.softmax(scores, dim=-1)                                   # (B, 8, 8)
            attn = torch.matmul(weights, v)                                      # (B, 8, 3)
            attn_outputs.append(attn)

        concat = torch.cat(attn_outputs, dim=-1)  # (B, 8, 6)
        proj = self.wo(concat)                    # (B, 8, 6)

        x = self.norm1(x + proj)                  # Add & Norm → (B, 8, 6)

        ff = self.ff2(self.relu(self.ff1(x)))     # FFN: 6→12→6 → (B, 8, 6)
        x = self.norm2(x + ff)                    # Add & Norm → (B, 8, 6)

        return x

# ----------------------
# Transformer + 分类头
# ----------------------
class TransformerModel(nn.Module):
    def __init__(self, d_model: int = 6, seq_len: int = 8, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.device = torch.device("cpu")
        self.encoder1 = SimpleEncoder(d_model=d_model, nhead=2)
        self.encoder2 = SimpleEncoder(d_model=d_model, nhead=2)

        # self.fc1 = nn.Linear(d_model * seq_len, 32)  # (B, 1, 48) → (B, 1, 32)
        self.fc1 = nn.Linear(6, 6)  # (B,8,6) → (B, 8,6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(6,1)            # (B,8,6) → (B, 8,1)
        self.sigmoid = nn.Sigmoid()

    


    def forward(self, x):
        # x = quantize_input(x, x_min=0, x_max=255)
        # x: (B, 8, 6)
        x = self.encoder1(x)                          # (B, 8, 6)
        x = self.encoder2(x)                          # (B, 8, 6)
        x = self.fc1(x)                              # (B, 8, 6)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                              # (B, 8,1)
        x = self.sigmoid(x)                          # (B, 8, 1)
        return x

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, feature_columns, label_column, seq_len=8, scaler=None):
        self.data = pd.read_csv(csv_file)
        
        # 处理IP地址
        for col in feature_columns:
            if 'IP' in col:
                self.data[col] = self.data[col].apply(lambda x: self._ip_to_int(x))
        '''
        # 获取特征
        features_df = self.data[feature_columns]
        
        # 替换无穷大值为NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # 处理NaN值
        for col in features_df.columns:
            median_value = features_df[col].median()
            features_df[col] = features_df[col].fillna(median_value)
        
        # 特征标准化
        from sklearn.preprocessing import StandardScaler
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features_df.values)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(features_df.values)
        '''

            # 获取特征
        features_df = self.data[feature_columns]
        
        # 检查并替换无穷大值为NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # 删除包含NaN值的行
        nan_rows = features_df.isna().any(axis=1)
        print(f"删除了 {nan_rows.sum()} 行含有NaN的数据")
        
        # 删除NaN行
        features_df = features_df.dropna()
        # 更新原始数据框，使用相同的索引删除对应行
        self.data = self.data.loc[features_df.index]
        
        # 特征标准化
        from sklearn.preprocessing import StandardScaler
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features_df.values)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(features_df.values)




        # 处理标签
        label_map = {"TOR": 1, "nonTOR": 0}
        self.labels = self.data[label_column].map(label_map).values
        
        self.seq_len = seq_len
        
        # 生成滑动窗口数据
        self.X, self.Y = self.create_sequences(self.features, self.labels, seq_len)
    
   
    def _ip_to_int(self, ip_str):
        try:
            ip_str = str(ip_str).strip()
            full_ip = int(ipaddress.IPv4Address(ip_str))
            last_two_bytes = full_ip & 0xFFFF
            return last_two_bytes
        except:
            return 0
    '''
    def _ip_to_int(self, ip_str):
        try:
            ip_str = str(ip_str).strip()
            # 使用 IPv4Address 转换为完整的 IP 地址
            full_ip = ipaddress.IPv4Address(ip_str)
            
            # 截取IP的第3部分（z），即从 x.y.z.a 中提取 z
            z_part = str(full_ip).split('.')[2]
            
            # 将截取的 z 部分转换为整数
            return int(z_part)
        except:
            return 0
     '''
    def create_sequences(self, features, labels, seq_len):
        X, Y = [], []
        for i in range(len(features) - seq_len + 1):
            X.append(features[i:i+seq_len])
            Y.append(labels[i:i+seq_len])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, seq_len, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

# 测试函数
def test_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            # predictions = (torch.sigmoid(outputs) > 0.5).float()  # 二分类阈值
            predictions = (outputs > 0.5).float()  # 二分类阈值
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 将3D数组展平为1D用于评估
    pred_flat = all_predictions.flatten()
    target_flat = all_targets.flatten()
    
    # 计算各种评估指标
    accuracy = accuracy_score(target_flat, pred_flat)
    precision = precision_score(target_flat, pred_flat)
    recall = recall_score(target_flat, pred_flat)
    f1 = f1_score(target_flat, pred_flat)
    conf_matrix = confusion_matrix(target_flat, pred_flat)
    
    print("\n===== 测试结果 =====")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("混淆矩阵:")
    print(conf_matrix)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

# 模型保存和加载函数
def save_model(model, optimizer, scaler, filename='transformer_model.pkl'):
    model_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"模型已保存到 {filename}")

def load_model(filename='transformer_model.pkl'):
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    model = TransformerModel(d_model=6, seq_len=8)
    model.load_state_dict(model_data['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(model_data['optimizer_state_dict'])
    
    scaler = model_data['scaler']
    
    return model, optimizer, scaler

# 主程序
def main():
    # 定义特征和标签列
    feature_columns = ["Source IP", " Destination IP", " Flow Packets/s", " Flow IAT Max", " Protocol", " Flow IAT Mean"]
    label_column = "label"
    
    # 设置设备
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    if not os.path.exists("transformer_model.pkl"):
        print("未找到已保存的模型，开始训练...")
        
        # 数据集路径
        dataset_csv = "Scenario-A-merged_5s.csv"
        full_dataset = TimeSeriesDataset(dataset_csv, feature_columns, label_column, seq_len=8)
        
        # 随机划分训练集和测试集 (80% 训练, 20% 测试)
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size
        
        # 使用固定种子以确保结果可复现
        generator = torch.Generator().manual_seed(42)
        train_indices, test_indices = random_split(
            range(dataset_size), 
            [train_size, test_size],
            generator=generator
        )
        
        print("生成量化前的训练集和测试集CSV...")
        train_df = save_quantized_data(
            full_dataset, 
            train_indices.indices, 
            "before_quantized_train_data.csv", 
            feature_columns, 
            label_column
        )
        test_df = save_quantized_data(
            full_dataset, 
            test_indices.indices, 
            "before_quantized_test_data.csv", 
            feature_columns, 
            label_column
        )
        
        # 创建实际的训练集和测试集
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices.indices)
        
        print(f"总数据集大小: {dataset_size}")
        print(f"训练集大小: {train_size} (80%), 保存了 {len(train_df)} 行量化数据")
        print(f"测试集大小: {test_size} (20%), 保存了 {len(test_df)} 行量化数据")
        
        # 数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # 初始化模型
        model = TransformerModel(d_model=6, seq_len=8)
        model.to(device)
        
        # 定义损失函数和优化器
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        # 训练模型
        print("开始训练模型...")
        train_model(model, train_dataloader, criterion, optimizer, num_epochs=10)
        
        # 保存模型
        save_model(model, optimizer, full_dataset.scaler)
        
        # 测试模型
        print("\n开始测试模型...")
        metrics = test_model(model, test_dataloader)
    else:
        # 如果已经有保存的模型，加载并进行测试
        print("加载已保存的模型...")
        model, optimizer, scaler = load_model()
        model.to(device)
        
        # 加载数据集并随机划分
        dataset_csv = "Scenario-A-merged_5s.csv"
        full_dataset = TimeSeriesDataset(dataset_csv, feature_columns, label_column, seq_len=8, scaler=scaler)
        
        # 随机划分训练集和测试集 (80% 训练, 20% 测试)
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size
        _, test_dataset = random_split(full_dataset, [train_size, test_size])
        
        # 测试数据加载器
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 测试模型
        print("\n开始测试模型...")
        metrics = test_model(model, test_dataloader)

if __name__ == "__main__":
    main()