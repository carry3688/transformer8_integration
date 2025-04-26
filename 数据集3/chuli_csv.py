import pandas as pd
'''
def add_header_from_b_to_a(a_file='UNSW-NB15_1.csv', b_file='NUSW-NB15_features.csv'):
    """
    从b.csv的第二列提取特征名（不包括第一行），
    并将这些特征名添加为a.csv的表头
    """
    # 读取b.csv文件(使用latin1编码)
    df_b = pd.read_csv(b_file, encoding='latin1')
    
    # 获取b.csv第二列的名称
    second_col_name = df_b.columns[1]
    
    # 提取第二列除第一行外的所有值作为特征名
    feature_names = df_b[second_col_name].tolist()
    
    # 读取a.csv文件(不使用表头，使用latin1编码)
    df_a = pd.read_csv(a_file, header=None, encoding='latin1')
    
    # 检查列数是否匹配
    if len(df_a.columns) != len(feature_names):
        print(f"警告：a.csv有{len(df_a.columns)}列，但从b.csv提取了{len(feature_names)}个特征名")
        # 根据需要调整特征名或数据
    
    # 设置特征名为a.csv的列名
    df_a.columns = feature_names
    
    # 保存修改后的a.csv
    df_a.to_csv(a_file, index=False, encoding='latin1')
    
    print(f"已成功将特征名添加到{a_file}的第一行")

# 执行函数
add_header_from_b_to_a()
'''
import pandas as pd
import numpy as np

def sample_dataset(input_file='UNSW-NB15_1.csv', output_file='data.csv', 
                  total_samples=30000, positive_ratio=0.05):
    """
    从UNSW-NB15数据集随机选择指定数量的样本，确保包含一定比例的正样本（标签为1）
    
    参数:
    input_file: 输入文件路径
    output_file: 输出文件路径
    total_samples: 要选择的总样本数
    positive_ratio: 正样本(标签为1)的目标比例
    """
    print(f"正在读取数据集: {input_file}")
    # 读取数据集，假设最后一列是标签列
    try:
        df = pd.read_csv(input_file, encoding='latin1')
    except Exception as e:
        print(f"读取文件时出错: {e}")
        print("尝试不指定列名读取...")
        df = pd.read_csv(input_file, header=None, encoding='latin1')
    
    # 确定标签列
    label_col = df.columns[-1]
    print(f"使用 '{label_col}' 作为标签列")
    
    # 分离正样本和负样本
    positive_samples = df[df[label_col] == 1]
    negative_samples = df[df[label_col] == 0]
    
    print(f"数据集共有 {len(df)} 条记录")
    print(f"标签为1的记录: {len(positive_samples)} ({len(positive_samples)/len(df)*100:.2f}%)")
    print(f"标签为0的记录: {len(negative_samples)} ({len(negative_samples)/len(df)*100:.2f}%)")
    
    # 计算要选择的正负样本数量
    n_positive = int(total_samples * positive_ratio)
    n_negative = total_samples - n_positive
    
    # 确保不超过原始数据集中可用的样本数
    n_positive = min(n_positive, len(positive_samples))
    n_negative = min(n_negative, len(negative_samples))
    
    # 调整总样本数
    actual_total = n_positive + n_negative
    
    print(f"将随机选择 {n_positive} 个正样本和 {n_negative} 个负样本，总计 {actual_total} 条记录")
    
    # 随机选择样本
    sampled_positive = positive_samples.sample(n=n_positive, random_state=42)
    sampled_negative = negative_samples.sample(n=n_negative, random_state=42)
    
    # 合并样本
    sampled_data = pd.concat([sampled_positive, sampled_negative])
    
    # 随机打乱顺序
    sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存结果
    sampled_data.to_csv(output_file, index=False, encoding='latin1')
    
    print(f"已将随机选择的 {len(sampled_data)} 条记录保存至 {output_file}")
    print(f"其中标签为1的记录: {len(sampled_data[sampled_data[label_col] == 1])} ({len(sampled_data[sampled_data[label_col] == 1])/len(sampled_data)*100:.2f}%)")

# 执行采样，使标签为1的样本占20%
sample_dataset(positive_ratio=0.05)