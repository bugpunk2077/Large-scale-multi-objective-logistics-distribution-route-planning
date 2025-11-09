import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj=None):
        """
        h: 输入特征 [batch_size, n_nodes, in_features]
        adj: 邻接矩阵 [n_nodes, n_nodes]
        """
        batch_size, n_nodes, _ = h.size()
        
        # 线性变换
        Wh = torch.matmul(h, self.W)  # [batch_size, n_nodes, out_features]
        
        # 计算注意力分数
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        # 应用邻接矩阵掩码
        if adj is not None:
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
        else:
            attention = e
            
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 应用注意力权重
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size, n_nodes, out_features = Wh.size()
        
        # 重复Wh以创建所有节点对
        Wh_repeated_in_chunks = Wh.repeat_interleave(n_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, n_nodes, 1)
        
        # 拼接所有节点对
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        
        return all_combinations_matrix.view(batch_size, n_nodes, n_nodes, 2 * out_features)

class GraphAttentionEncoder(nn.Module):
    """图注意力编码器"""
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=4, dropout=0.1):
        super(GraphAttentionEncoder, self).__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        
        # 多头注意力层
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout) 
            for _ in range(n_heads)
        ])
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim * n_heads, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x, adj=None):
        # x: [batch_size, n_nodes, input_dim]
        head_outputs = []
        for attn in self.attentions:
            head_output = attn(x, adj)
            head_outputs.append(head_output)
        
        # 拼接多头输出
        x = torch.cat(head_outputs, dim=2)  # [batch_size, n_nodes, hidden_dim * n_heads]
        x = F.elu(x)
        x = self.output_proj(x)
        x = self.layer_norm(x)
        
        return x

class PolicyNetwork(nn.Module):
    """策略网络 - 用于生成路径决策"""
    def __init__(self, node_dim, hidden_dim, n_actions):
        super(PolicyNetwork, self).__init__()
        self.encoder = GraphAttentionEncoder(node_dim, hidden_dim, hidden_dim)
        
        # LSTM解码器
        self.decoder_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )
        
        # 上下文投影
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_features, current_node, visited_mask, adj_matrix=None):
        """
        node_features: [batch_size, n_nodes, node_dim]
        current_node: [batch_size] - 当前节点索引
        visited_mask: [batch_size, n_nodes] - 已访问节点掩码
        """
        batch_size, n_nodes, node_dim = node_features.size()
        
        # 编码节点特征
        encoded_nodes = self.encoder(node_features, adj_matrix)  # [batch_size, n_nodes, hidden_dim]
        
        # 获取当前节点编码
        current_node_encoding = encoded_nodes[torch.arange(batch_size), current_node]  # [batch_size, hidden_dim]
        
        # 获取全局上下文
        global_context = encoded_nodes.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 合并当前节点和全局上下文
        decoder_input = torch.cat([current_node_encoding, global_context], dim=1).unsqueeze(1)  # [batch_size, 1, hidden_dim*2]
        
        # LSTM解码
        h0 = torch.zeros(1, batch_size, encoded_nodes.size(-1)).to(node_features.device)
        c0 = torch.zeros(1, batch_size, encoded_nodes.size(-1)).to(node_features.device)
        
        decoder_output, (hn, cn) = self.decoder_lstm(decoder_input, (h0, c0))  # [batch_size, 1, hidden_dim]
        
        # 计算动作概率
        action_logits = self.output_layer(decoder_output.squeeze(1))  # [batch_size, n_actions]
        
        # 应用掩码 - 将已访问节点的概率设为负无穷
        action_logits = action_logits.masked_fill(visited_mask, -1e9)
        
        # 计算概率分布
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, action_logits

class ValueNetwork(nn.Module):
    """价值网络 - 用于评估解决方案质量"""
    def __init__(self, node_dim, hidden_dim, n_objectives=3):
        super(ValueNetwork, self).__init__()
        self.encoder = GraphAttentionEncoder(node_dim, hidden_dim, hidden_dim)
        
        # 解决方案编码器
        self.solution_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, n_objectives)
        )
        
    def forward(self, node_features, solution_routes, adj_matrix=None):
        """
        node_features: [batch_size, n_nodes, node_dim]
        solution_routes: [batch_size, max_route_length] - 解决方案路径
        """
        batch_size, n_nodes, node_dim = node_features.size()
        
        # 编码节点特征
        encoded_nodes = self.encoder(node_features, adj_matrix)  # [batch_size, n_nodes, hidden_dim]
        
        # 编码解决方案
        route_embeddings = []
        for i in range(batch_size):
            route = solution_routes[i]
            valid_nodes = route[route >= 0]  # 去除填充值
            if len(valid_nodes) > 0:
                route_embedding = encoded_nodes[i, valid_nodes.long()].mean(dim=0)
            else:
                route_embedding = torch.zeros(encoded_nodes.size(-1)).to(node_features.device)
            route_embeddings.append(route_embedding)
        
        route_embeddings = torch.stack(route_embeddings)  # [batch_size, hidden_dim]
        
        # 全局图嵌入
        global_embedding = encoded_nodes.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 合并特征
        combined = torch.cat([route_embeddings, global_embedding], dim=1)  # [batch_size, hidden_dim * 2]
        solution_encoding = self.solution_encoder(combined)  # [batch_size, hidden_dim // 2]
        
        # 预测价值
        value_estimate = self.value_head(solution_encoding)  # [batch_size, n_objectives]
        
        return value_estimate

class NeuralVRPSolver(nn.Module):
    """完整的神经网络VRP求解器"""
    def __init__(self, node_dim, hidden_dim=128, n_actions=100, n_objectives=3):
        super(NeuralVRPSolver, self).__init__()
        self.policy_net = PolicyNetwork(node_dim, hidden_dim, n_actions)
        self.value_net = ValueNetwork(node_dim, hidden_dim, n_objectives)
        self.hidden_dim = hidden_dim
        
    def forward(self, node_features, current_node, visited_mask, solution_routes=None, adj_matrix=None):
        # 策略网络前向传播
        action_probs, action_logits = self.policy_net(node_features, current_node, visited_mask, adj_matrix)
        
        # 价值网络前向传播（如果提供了解决方案）
        if solution_routes is not None:
            value_estimate = self.value_net(node_features, solution_routes, adj_matrix)
            return action_probs, action_logits, value_estimate
        
        return action_probs, action_logits

def save_model(model, filepath):
    """保存模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'node_dim': model.policy_net.encoder.attentions[0].in_features,
            'hidden_dim': model.hidden_dim,
            'n_actions': model.policy_net.output_layer[2].out_features,
            'n_objectives': model.value_net.value_head[2].out_features
        }
    }, filepath)
    print(f"模型已保存到: {filepath}")

def load_model(filepath, device='cpu'):
    """加载模型"""
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['config']
    
    model = NeuralVRPSolver(
        node_dim=config['node_dim'],
        hidden_dim=config['hidden_dim'],
        n_actions=config['n_actions'],
        n_objectives=config['n_objectives']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"模型已从 {filepath} 加载")
    
    return model