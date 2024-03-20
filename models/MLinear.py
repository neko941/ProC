import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientAttention(nn.Module):
    def __init__(self, d_model):
        super(EfficientAttention, self).__init__()
        # Linear transformations for Q, K, V
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # Assuming x is of shape (batch_size, num_channels, sequence_length)
        # Compute Q, K, V
        Q = self.linear_q(x)  # Shape: (batch_size, num_channels, d_model)
        K = self.linear_k(x)  # Shape: (batch_size, num_channels, d_model)
        V = self.linear_v(x)  # Shape: (batch_size, num_channels, d_model)
        
        # Compute attention scores
        KV = torch.matmul(K.transpose(-2, -1), V) / self.d_model**0.5  # Shape: (batch_size, d_model, d_model)
        E = torch.matmul(Q, KV)  # Shape: (batch_size, num_channels, d_model)
        
        return E

class MLinear(nn.Module):
    def __init__(self, input_dim, sequence_length, num_channels, num_linear_transforms, d_model):
        super(MLinear, self).__init__()
        # Linear layers to transform the input sequence
        self.linears = nn.ModuleList([nn.Linear(input_dim, sequence_length) for _ in range(num_linear_transforms)])
        # Efficient attention mechanism
        self.efficient_attention = EfficientAttention(d_model)
        # A mixing layer to combine the outputs of the linear layers and the attention mechanism
        self.mix = nn.Sequential(
            nn.Linear((num_linear_transforms + 1) * sequence_length, sequence_length),
            nn.ReLU()
        )
        # Final output linear layers
        self.output_linears = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(3)])
    
    def forward(self, x):
        # x shape: (batch_size, num_channels, input_dim)
        parallel_transforms = [linear(x) for linear in self.linears]
        # Each element of parallel_transforms has shape: (batch_size, num_channels, sequence_length)

        # Attended shape before concatenation: (batch_size, num_channels, d_model)
        attended = self.efficient_attention(x)
        # Reshape to make it compatible with other parallel_transforms
        attended = attended.view(x.size(0), x.size(1), -1)
        # attended shape after view: (batch_size, num_channels, sequence_length)

        # Concatenate all the transformed sequences with the attended sequence
        # concatenated shape: (batch_size, num_channels, (num_linear_transforms + 1) * sequence_length)
        concatenated = torch.cat(parallel_transforms + [attended], dim=-1)

        # Pass through the mix layer
        # mixed shape: (batch_size, num_channels, sequence_length)
        mixed = self.mix(concatenated)

        # Pass through each of the final linear layers
        outputs = [output_linear(mixed) for output_linear in self.output_linears]
        # Each output shape: (batch_size, num_channels, sequence_length)

        return outputs

# # Example instantiation and forward pass
# input_dim = 512  # Example input feature dimension
# sequence_length = 128  # Example sequence length
# num_channels = 3  # Example number of channels
# num_linear_transforms = 8  # Example number of parallel linear transformations
# d_model = 512  # Dimensionality of the model