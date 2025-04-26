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

def save_quantized_data(dataset, indices, filename, feature_columns, label_column):
    """
    将数据集量化并保存为CSV文件，用于测量正确率
    """
    data_list = []
    for idx in indices:
        features, label = dataset[idx]
        quantized_features = quantize_input(features)
        
        for t in range(features.shape[0]):
            row_features = quantized_features[t].numpy()
            row_label = label[t][0].item()
            row = np.append(row_features, row_label)
            data_list.append(row)
    
    column_names = feature_columns + [label_column]
    df = pd.DataFrame(data_list, columns=column_names)
    df.to_csv(filename, index=False)
    print(f"量化后的数据已保存到: {filename}")
    return df

# 量化输入
def quantize_input(x, x_min = None, x_max = None):
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    x_clipped = torch.clamp(x, x_min, x_max)
    scale = 15.0 / (x_max - x_min)
    x_q = ((x_clipped - x_min) * scale).round()
    return x_q

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
        self.wq = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(nhead)])
        self.wk = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(nhead)])
        self.wv = nn.ModuleList([nn.Linear(d_model, self.d_head) for _ in range(nhead)])
        self.wo = nn.Linear(d_model, d_model)
        self.norm1 = ManualLayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 2)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_model * 2, d_model)
        self.norm2 = ManualLayerNorm(d_model)

    def forward(self, x):
        attn_outputs = []
        for i in range(self.nhead):
            q = self.wq[i](x) 
            k = self.wk[i](x) 
            v = self.wv[i](x) 

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  
            weights = F.softmax(scores, dim=-1)                                   
            attn = torch.matmul(weights, v)                                    
            attn_outputs.append(attn)

        concat = torch.cat(attn_outputs, dim=-1) 
        proj = self.wo(concat)                   

        x = self.norm1(x + proj)                

        ff = self.ff2(self.relu(self.ff1(x)))    
        x = self.norm2(x + ff)                   

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
        self.fc1 = nn.Linear(6, 6) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(6,1)           
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder1(x)                          
        x = self.encoder2(x)                         
        x = self.fc1(x)                             
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                              
        x = self.sigmoid(x)                         
        return x


class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, feature_columns, label_column, seq_len=8, scaler=None):
        self.data = pd.read_csv(csv_file)
        
        for col in feature_columns:
            if 'IP' in col:
                self.data[col] = self.data[col].apply(lambda x: self._ip_to_int(x))

        features_df = self.data[feature_columns]
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        nan_rows = features_df.isna().any(axis=1)
        features_df = features_df.dropna()
        self.data = self.data.loc[features_df.index]
        from sklearn.preprocessing import StandardScaler
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(features_df.values)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(features_df.values)

        label_map = {"TOR": 1, "nonTOR": 0}
        self.labels = self.data[label_column].map(label_map).values
        
        self.seq_len = seq_len
        self.X, self.Y = self.create_sequences(self.features, self.labels, seq_len)
    
    def _ip_to_int(self, ip_str):
        try:
            ip_str = str(ip_str).strip()
            full_ip = int(ipaddress.IPv4Address(ip_str))
            last_two_bytes = full_ip & 0xFFFF
            return last_two_bytes
        except:
            return 0
    
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


def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = quantize_input(inputs)
            inputs, targets = inputs.to(model.device), targets.to(model.device)         
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)        
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")


def test_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = quantize_input(inputs)
            
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            predictions = (outputs > 0.5).float() 
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    pred_flat = all_predictions.flatten()
    target_flat = all_targets.flatten()
    
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
    feature_columns = ["Source IP", " Destination IP", " Flow Packets/s", " Flow IAT Max", " Protocol", " Flow IAT Mean"]
    label_column = "label"
    
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    
    if not os.path.exists("transformer_model.pkl"):
        print("未找到已保存的模型，开始训练...")
        
        dataset_csv = "Scenario-A-merged_5s.csv"
        full_dataset = TimeSeriesDataset(dataset_csv, feature_columns, label_column, seq_len=8)
        
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size
        
        generator = torch.Generator().manual_seed(42)
        train_indices, test_indices = random_split(
            range(dataset_size), 
            [train_size, test_size],
            generator=generator
        )
        
        # print("生成量化后的训练集和测试集CSV...")
        train_df = save_quantized_data(
            full_dataset, 
            train_indices.indices, 
            "quantized_train_data.csv", 
            feature_columns, 
            label_column
        )
        test_df = save_quantized_data(
            full_dataset, 
            test_indices.indices, 
            "quantized_test_data.csv", 
            feature_columns, 
            label_column
        )
        
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices.indices)
        
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = TransformerModel(d_model=6, seq_len=8)
        model.to(device)
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        print("开始训练模型...")
        train_model(model, train_dataloader, criterion, optimizer, num_epochs=10)
        
        save_model(model, optimizer, full_dataset.scaler)
        
        print("\n开始测试模型...")
        metrics = test_model(model, test_dataloader)
    else:
        print("加载已保存的模型...")
        model, optimizer, scaler = load_model()
        model.to(device)
        
        dataset_csv = "Scenario-A-merged_5s.csv"
        full_dataset = TimeSeriesDataset(dataset_csv, feature_columns, label_column, seq_len=8, scaler=scaler)
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size
        _, test_dataset = random_split(full_dataset, [train_size, test_size])
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print("\n开始测试模型...")
        metrics = test_model(model, test_dataloader)

if __name__ == "__main__":
    main()