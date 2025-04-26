#!/usr/bin/env python3
import numpy as np
import ipaddress
from scapy.all import sendp, get_if_list, get_if_hwaddr, Ether
from scapy.fields import *
from scapy.packet import Packet
import csv
import time

# 定义自定义数据包结构
class CustomP4Headers(Packet):
    name = "CustomP4Headers"
    fields_desc = [
        # 控制标志
        ByteField("ctrl_flag", 0),
        
        # input 字段 (8x6 矩阵)
        ShortField("input_0_0", 0),
        ShortField("input_0_1", 0),
        ShortField("input_0_2", 0),
        ShortField("input_0_3", 0),
        ShortField("input_0_4", 0),
        ShortField("input_0_5", 0),
        ShortField("input_1_0", 0),
        ShortField("input_1_1", 0),
        ShortField("input_1_2", 0),
        ShortField("input_1_3", 0),
        ShortField("input_1_4", 0),
        ShortField("input_1_5", 0),
        ShortField("input_2_0", 0),
        ShortField("input_2_1", 0),
        ShortField("input_2_2", 0),
        ShortField("input_2_3", 0),
        ShortField("input_2_4", 0),
        ShortField("input_2_5", 0),
        ShortField("input_3_0", 0),
        ShortField("input_3_1", 0),
        ShortField("input_3_2", 0),
        ShortField("input_3_3", 0),
        ShortField("input_3_4", 0),
        ShortField("input_3_5", 0),
        ShortField("input_4_0", 0),
        ShortField("input_4_1", 0),
        ShortField("input_4_2", 0),
        ShortField("input_4_3", 0),
        ShortField("input_4_4", 0),
        ShortField("input_4_5", 0),
        ShortField("input_5_0", 0),
        ShortField("input_5_1", 0),
        ShortField("input_5_2", 0),
        ShortField("input_5_3", 0),
        ShortField("input_5_4", 0),
        ShortField("input_5_5", 0),
        ShortField("input_6_0", 0),
        ShortField("input_6_1", 0),
        ShortField("input_6_2", 0),
        ShortField("input_6_3", 0),
        ShortField("input_6_4", 0),
        ShortField("input_6_5", 0),
        ShortField("input_7_0", 0),
        ShortField("input_7_1", 0),
        ShortField("input_7_2", 0),
        ShortField("input_7_3", 0),
        ShortField("input_7_4", 0),
        ShortField("input_7_5", 0),
        
        # output 字段 (8个输出)
        ShortField("output0", 0),
        ShortField("output1", 0),
        ShortField("output2", 0),
        ShortField("output3", 0),
        ShortField("output4", 0),
        ShortField("output5", 0),
        ShortField("output6", 0),
        ShortField("output7", 0)
    ]

def get_if():
    """获取网络接口"""
    iface = None
    for i in get_if_list():
        if "eth0" in i:
            iface = i
            break
    if not iface:
        print("Cannot find eth0 interface")
        exit(1)
    return iface

def ip_to_int(ip_str):
    """将IP地址转换为整数（仅保留后两个字节）"""
    try:
        ip_str = str(ip_str).strip()
        full_ip = int(ipaddress.IPv4Address(ip_str))
        last_two_bytes = full_ip & 0xFFFF
        return last_two_bytes
    except:
        return 0

def standardize(data):
    """手动标准化数据"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # 避免除以0
    std = np.where(std == 0, 1, std)
    return (data - mean) / std

def load_and_preprocess_data(csv_file):
    """使用numpy加载和预处理数据集"""
    # 特征列的索引 (假设CSV结构固定)
    feature_indices = [0, 1, 10, 24, 8, 22]  # 对应您的6个特征列的索引
    label_index = -1  # 假设标签在最后一列
    
    # 读取CSV文件
    # print("正在读取CSV文件...")
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) 
        data = []
        labels = []
        
        for i, col in enumerate(header):
            print(f"{i}: {col}")
        
        for row in reader:
            try:
                features = []
                for i in feature_indices:
                    if i < len(row): 
                        value = row[i].strip()
                        if "IP" in header[i]:
                            features.append(float(ip_to_int(value)))
                        else:
                            try:
                                features.append(float(value))
                            except ValueError:
                                break
                    else:
                        break
                
                if len(features) == len(feature_indices):
                    data.append(features)
                    
                    label_value = row[label_index].strip() if label_index > -1 and label_index < len(row) else "nonTOR"
                    label = 1 if label_value == "TOR" else 0
                    labels.append(label)
            except Exception as e:
                print(f"处理行时出错: {e}")
                continue
    
    features = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    valid_indices = ~np.any(np.isnan(features) | np.isinf(features), axis=1)
    features = features[valid_indices]
    labels = labels[valid_indices]
    
    print(f"删除无效行后剩余 {len(features)} 行数据")
    features = standardize(features)
    
    return features, labels

def create_sequences(features, seq_len=8):
    """创建序列数据"""
    num_complete_sequences = len(features) // seq_len
    features = features[:num_complete_sequences * seq_len]
    sequences = features.reshape(-1, seq_len, features.shape[1])
    return sequences

def quantize_input(x, x_min=0, x_max=127):
    """量化输入"""
    x_min = np.min(x) if x_min is None else x_min
    x_max = np.max(x) if x_max is None else x_max
    x_clipped = np.clip(x, x_min, x_max)
    scale =127.0 / (x_max - x_min)
    x_q = np.round((x_clipped - x_min) * scale)
    return x_q.astype(np.int16)

def main():
    iface = get_if()
    dst_mac = "00:00:00:00:00:02"
    csv_file = "Scenario-A-merged_5s.csv"
    features, labels = load_and_preprocess_data(csv_file)
    
    seq_len = 8
    feature_sequences = create_sequences(features, seq_len)
    
    quantized_sequences = quantize_input(feature_sequences)
    
    
    # 选择前100个序列进行发送
    seq_num = 100
    max_sequences = min(seq_num, len(quantized_sequences))
    
    for i in range(max_sequences):
        inputs = {}
        sequence = quantized_sequences[i]
        
        for row in range(seq_len):
            for col in range(6):  
                field_name = f"input_{row}_{col}"
                inputs[field_name] = int(sequence[row][col])
        
        pkt = Ether(
            src=get_if_hwaddr(iface),
            dst=dst_mac,
            type=0x1234 
        ) / CustomP4Headers(
            ctrl_flag=0,
            **inputs
        )
        
        print(f"发送序列 {i+1}/{max_sequences}...")
        sendp(pkt, iface=iface, verbose=False)
        
        time.sleep(0.1)
    
    print("所有数据包发送完成!")

if __name__ == "__main__":
    main()