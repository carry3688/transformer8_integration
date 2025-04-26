

import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from scapy.all import Ether, sendp, BitField, Packet
from p4utils.utils.helper import load_topo
from p4utils.utils.sswitch_p4runtime_API import SimpleSwitchP4RuntimeAPI
import pickle
import struct

# 设置control_flag
class CtrlFlag(Packet):
    name = "CtrlFlag"
    fields_desc = [BitField("ctrl_flag", 1, 8)]  

# 量化输入
def quantize_input(x, x_min = None, x_max= None):
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    x_clipped = torch.clamp(x, x_min, x_max)
    scale = 127.0 / (x_max - x_min)
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
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.encoder1 = SimpleEncoder(d_model=d_model, nhead=2)
        self.encoder2 = SimpleEncoder(d_model=d_model, nhead=2)
        self.fc1 = nn.Linear(6, 6)  # (B, 8, 6) → (B, 8, 6)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(6,1)            # (B, 8, 6) → (B, 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder1(x)                         # (B, 8, 6)
        x = self.encoder2(x)                         # (B, 8, 6)
        x = self.fc1(x)                              # (B, 8, 6)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                              # (B, 8, 1)
        x = self.sigmoid(x)                          # (B, 8, 1)
        return x
    
class SimpleController:
    
    def __init__(self, sw_name):
        self.topo = load_topo('topology.json')
        self.sw_name = sw_name
        
        # 获取设备ID和GRPC端口
        device_id = self.topo.get_p4switch_id(sw_name)
        grpc_port = self.topo.get_grpc_port(sw_name)
        sw_data = self.topo.get_p4rtswitches()[sw_name] 
        # 初始化控制器
        self.controller = SimpleSwitchP4RuntimeAPI(device_id, grpc_port,
                                                  p4rt_path=sw_data['p4rt_path'],
                                                  json_path=sw_data['json_path'])
        # 存储收到的数据包信息
        self.packet_buffer = []
        self.packet_x_tensor = []
        self.src_addr = None
        self.dst_addr = None
        # 加载模型
        self.model = TransformerModel(d_model=6, seq_len=8)
        with open("transformer_model.pkl", 'rb') as f:
            model_data = pickle.load(f)
    
        self.model.load_state_dict(model_data['model_state_dict'])
        self.scaler = model_data['scaler']
        self.model.eval()
        # 获取接口信息
        self.interfaces = {}
        for port, intf in self.topo.get_interfaces_to_node(sw_name).items():
            self.interfaces[port] = intf
        
        print(f"交换机接口: {self.interfaces}")
        self.reset()
    
    def reset(self):
        # 重置GRPC服务器
        self.controller.reset_state()

    def mac_to_string(self, mac_int):
        mac_bytes = mac_int.to_bytes(6, byteorder='big')
        return ':'.join('{:02x}'.format(b) for b in mac_bytes)
    
    def config_digest(self):
        # 配置digest接收
        self.controller.digest_enable('digest_t', 1000000, 10, 1000000)

    def packet_data_to_tensor(self, packet_dict):
        # 将数据包字典转换为张量
        vals = []
        for i in range(8):  # seq_len
            row = []
            for j in range(6):  # d_model
                key = f"input_{i}_{j}"
                row.append(packet_dict[key])
            vals.append(row)
        x = torch.tensor([vals], dtype=torch.float32)  # shape: (1, 8, 6)
        return x
    
    def unpack_digest(self, dig_list):
        packet_data = []
        for dig in dig_list.data:
            # 从digest中提取数据
            input_0_0 = int.from_bytes(dig.struct.members[0].bitstring, byteorder='big')
            input_0_1 = int.from_bytes(dig.struct.members[1].bitstring, byteorder='big')
            input_0_2 = int.from_bytes(dig.struct.members[2].bitstring, byteorder='big')
            input_0_3 = int.from_bytes(dig.struct.members[3].bitstring, byteorder='big')
            input_0_4 = int.from_bytes(dig.struct.members[4].bitstring, byteorder='big')
            input_0_5 = int.from_bytes(dig.struct.members[5].bitstring, byteorder='big')
            input_1_0 = int.from_bytes(dig.struct.members[6].bitstring, byteorder='big')
            input_1_1 = int.from_bytes(dig.struct.members[7].bitstring, byteorder='big')
            input_1_2 = int.from_bytes(dig.struct.members[8].bitstring, byteorder='big')
            input_1_3 = int.from_bytes(dig.struct.members[9].bitstring, byteorder='big')
            input_1_4 = int.from_bytes(dig.struct.members[10].bitstring, byteorder='big')
            input_1_5 = int.from_bytes(dig.struct.members[11].bitstring, byteorder='big')
            input_2_0 = int.from_bytes(dig.struct.members[12].bitstring, byteorder='big')
            input_2_1 = int.from_bytes(dig.struct.members[13].bitstring, byteorder='big')
            input_2_2 = int.from_bytes(dig.struct.members[14].bitstring, byteorder='big')
            input_2_3 = int.from_bytes(dig.struct.members[15].bitstring, byteorder='big')
            input_2_4 = int.from_bytes(dig.struct.members[16].bitstring, byteorder='big')
            input_2_5 = int.from_bytes(dig.struct.members[17].bitstring, byteorder='big')
            input_3_0 = int.from_bytes(dig.struct.members[18].bitstring, byteorder='big')
            input_3_1 = int.from_bytes(dig.struct.members[19].bitstring, byteorder='big')
            input_3_2 = int.from_bytes(dig.struct.members[20].bitstring, byteorder='big')
            input_3_3 = int.from_bytes(dig.struct.members[21].bitstring, byteorder='big')
            input_3_4 = int.from_bytes(dig.struct.members[22].bitstring, byteorder='big')
            input_3_5 = int.from_bytes(dig.struct.members[23].bitstring, byteorder='big')
            input_4_0 = int.from_bytes(dig.struct.members[24].bitstring, byteorder='big')
            input_4_1 = int.from_bytes(dig.struct.members[25].bitstring, byteorder='big')
            input_4_2 = int.from_bytes(dig.struct.members[26].bitstring, byteorder='big')
            input_4_3 = int.from_bytes(dig.struct.members[27].bitstring, byteorder='big')
            input_4_4 = int.from_bytes(dig.struct.members[28].bitstring, byteorder='big')
            input_4_5 = int.from_bytes(dig.struct.members[29].bitstring, byteorder='big')
            input_5_0 = int.from_bytes(dig.struct.members[30].bitstring, byteorder='big')
            input_5_1 = int.from_bytes(dig.struct.members[31].bitstring, byteorder='big')
            input_5_2 = int.from_bytes(dig.struct.members[32].bitstring, byteorder='big')
            input_5_3 = int.from_bytes(dig.struct.members[33].bitstring, byteorder='big')
            input_5_4 = int.from_bytes(dig.struct.members[34].bitstring, byteorder='big')
            input_5_5 = int.from_bytes(dig.struct.members[35].bitstring, byteorder='big')
            input_6_0 = int.from_bytes(dig.struct.members[36].bitstring, byteorder='big')
            input_6_1 = int.from_bytes(dig.struct.members[37].bitstring, byteorder='big')
            input_6_2 = int.from_bytes(dig.struct.members[38].bitstring, byteorder='big')
            input_6_3 = int.from_bytes(dig.struct.members[39].bitstring, byteorder='big')
            input_6_4 = int.from_bytes(dig.struct.members[40].bitstring, byteorder='big')
            input_6_5 = int.from_bytes(dig.struct.members[41].bitstring, byteorder='big')
            input_7_0 = int.from_bytes(dig.struct.members[42].bitstring, byteorder='big')
            input_7_1 = int.from_bytes(dig.struct.members[43].bitstring, byteorder='big')
            input_7_2 = int.from_bytes(dig.struct.members[44].bitstring, byteorder='big')
            input_7_3 = int.from_bytes(dig.struct.members[45].bitstring, byteorder='big')
            input_7_4 = int.from_bytes(dig.struct.members[46].bitstring, byteorder='big')
            input_7_5 = int.from_bytes(dig.struct.members[47].bitstring, byteorder='big')
            # 解析其他字段
            src_addr = int.from_bytes(dig.struct.members[48].bitstring, byteorder='big')
            dst_addr = int.from_bytes(dig.struct.members[49].bitstring, byteorder='big')
            
            packet_data.append({
                'input_0_0': input_0_0,
                'input_0_1': input_0_1,
                'input_0_2': input_0_2,
                'input_0_3': input_0_3,
                'input_0_4': input_0_4,
                'input_0_5': input_0_5,
                'input_1_0': input_1_0,
                'input_1_1': input_1_1,
                'input_1_2': input_1_2,
                'input_1_3': input_1_3,
                'input_1_4': input_1_4,
                'input_1_5': input_1_5,
                'input_2_0': input_2_0,
                'input_2_1': input_2_1,
                'input_2_2': input_2_2,
                'input_2_3': input_2_3,
                'input_2_4': input_2_4,
                'input_2_5': input_2_5,
                'input_3_0': input_3_0,
                'input_3_1': input_3_1,
                'input_3_2': input_3_2,
                'input_3_3': input_3_3,
                'input_3_4': input_3_4,
                'input_3_5': input_3_5,
                'input_4_0': input_4_0,
                'input_4_1': input_4_1,
                'input_4_2': input_4_2,
                'input_4_3': input_4_3,
                'input_4_4': input_4_4,
                'input_4_5': input_4_5,
                'input_5_0': input_5_0,
                'input_5_1': input_5_1,
                'input_5_2': input_5_2,
                'input_5_3': input_5_3,
                'input_5_4': input_5_4,
                'input_5_5': input_5_5,
                'input_6_0': input_6_0,
                'input_6_1': input_6_1,
                'input_6_2': input_6_2,
                'input_6_3': input_6_3,
                'input_6_4': input_6_4,
                'input_6_5': input_6_5,
                'input_7_0': input_7_0,
                'input_7_1': input_7_1,
                'input_7_2': input_7_2,
                'input_7_3': input_7_3,
                'input_7_4': input_7_4,
                'input_7_5': input_7_5,
                'src_addr': src_addr,
                'dst_addr': dst_addr,
            })
        
        return packet_data

    
    def process_digest(self, dig_list):
        # 解析digest数据
        packet_data = self.unpack_digest(dig_list)
        
        # 将收到的包添加到缓冲区
        for pkt in packet_data:
            self.packet_buffer.append(pkt)
            self.src_addr = pkt['src_addr']
            self.dst_addr = pkt['dst_addr']
            x_tensor = self.packet_data_to_tensor(pkt)  # shape: (1, 8, 6)
            self.packet_x_tensor.append(x_tensor) # 存储张量
            # print(f"收到数据包: src={self.mac_to_string(pkt['src_addr'])}, dst={self.mac_to_string(pkt['dst_addr'])}, 入端口={pkt['ingress_port']}")
            # print(f"收到数据面的输入: input_0_0={pkt['input_0_0']}")
        # 当收集到5个数据包后处理它们
        if len(self.packet_buffer) >= 10:
            # print(f"已收集{len(self.packet_buffer)}个数据包，等待处理...")
            time.sleep(5)  # 等待5秒
            self.process_and_forward_packets()
    
    def packet_out_scapy(self, packet_data, egress_port):
        """使用Scapy发送数据包到特定端口"""
        # 检查端口是否存在于接口映射中
        if egress_port not in self.interfaces:
            print(f"错误: 端口 {egress_port} 没有对应的接口")
            return False
        
        interface = self.interfaces[egress_port]
        
        try:
            # 如果packet_data是原始字节，先转换为Scapy的以太网包
            if isinstance(packet_data, bytes):
                pkt = Ether(packet_data)
            else:
                pkt = packet_data
                
            # 使用sendp发送数据包
            sendp(pkt, iface=egress_port, verbose=False)
            print(f"成功通过接口 {interface} 发送数据包")
            return True
        except Exception as e:
            print(f"发送数据包时出错: {e}")
            return False
    
    def process_and_forward_packets(self):
        # print("处理并转发数据包...")
        
        # 处理每个数据包并发送回交换机
        for pkt in self.packet_x_tensor:   
            with torch.no_grad():
                output = self.model(pkt)  # 输出: (1, 8, 1)
                print(f"Raw output: {output}")
                prediction = (output > 0.5).int()

            print(f"[Packet] Prediction:\n{prediction.squeeze(-1)}") 

            egress_port = 's1-eth1'
            prediction_np = prediction.squeeze(-1).numpy() #(8,)

            print(f"预测结果: {prediction_np}")
            # 重建原始以太网帧
            src_mac_str = self.mac_to_string(self.src_addr)
            dst_mac_str = self.mac_to_string(self.dst_addr)
            ethertype = 0x0800
            
            # 创建Scapy以太网包
            eth_pkt = Ether(src=src_mac_str, dst=dst_mac_str, type=ethertype)
            ctrlflag = CtrlFlag(ctrl_flag=1)
            
            input_fields = pkt.flatten().numpy()  
            input_fields = [int(value) for value in input_fields]
            input_header = struct.pack("!48H", *input_fields)
            prediction_np = prediction_np.squeeze(0)
            prediction_np = [int(value) for value in prediction_np]
            output_header = struct.pack("!8H", *prediction_np)
            complete_pkt = eth_pkt/ctrlflag/input_header/output_header
            
            # print(f"转发数据包到端口 {egress_port}，源MAC={src_mac_str}，目的MAC={dst_mac_str}")
            self.packet_out_scapy(complete_pkt, egress_port)
        
        # 清空缓冲区
        self.packet_buffer = []
    
    def run_digest_loop(self):
        # 配置digest接收
        self.config_digest()
        print("控制器已启动，等待接收数据包...")
        
        # 循环接收digest
        while True:
            dig_list = self.controller.get_digest_list()
            self.process_digest(dig_list)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python controller.py <switch_name>")
        sys.exit(1)
    
    sw_name = sys.argv[1]
    controller = SimpleController(sw_name)
    controller.run_digest_loop()