import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientAttention(nn.Module):
    def __init__(self, d_model):
        super(EfficientAttention, self).__init__()
        self.d_model = d_model
        # Linear transformations for Q, K, V
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)
        
        # Calculate attention scores with proper scaling
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        attn_scores = F.softmax(attn_scores, dim=-1)
        
        # Apply attention scores to V
        E = torch.matmul(attn_scores, V)
        return E

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        d_model = 64
        self.d_model = d_model
        num_linear_transforms = 3
        
        self.linears = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(num_linear_transforms)])
        self.efficient_attention = EfficientAttention(d_model)
        self.mix = nn.Linear((num_linear_transforms + 1) * d_model, self.pred_len)
        
    def forward(self, x):
        # print(f'{x.shape = }\n\n\n\n\n\n\n\n\n\n\n')
        batch_size = x.shape[0]
        x = x.permute(0,2,1)
        parallel_transforms = [linear(x) for linear in self.linears]
        # print(self.linears[0](x))
        # exit()
        
        # Concatenate all parallel transforms along the last dimension
        concatenated = torch.cat(parallel_transforms, dim=-1)
        
        # Compute attention
        attended = self.efficient_attention(x)
        
        # Concatenate the attended vector with the parallel_transforms
        concatenated = torch.cat([concatenated, attended], dim=-1)
        
        # Pass through mix layer
        mixed = self.mix(concatenated)
        
        # No permute needed, just reshape to (batch_size, pred_len, d_model)
        pred = mixed.reshape(batch_size, self.pred_len, self.d_model)
        
        return pred

