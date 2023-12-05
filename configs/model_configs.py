class VanillaLSTMConfig():
    def __init__(self):
        self.units = [64, 64]
        self.weight_path = None
        
class PatchTSTConfig():
    def __init__(self):
        self.enc_layers = 3
        self.n_heads = 16
        self.d_model = 128
        self.d_ff = 256
        self.d_k = self.d_v = None
        
        self.attn_dropout = 0.0
        self.dropout = 0.2  
        self.head_dropout = 0.0
        self.fc_dropout = 0.2
        self.norm = 'BatchNorm'
        self.activation = 'gelu'
        
        self.patch_len = 16
        self.padding_patch = 'end'
        self.stride = 8
        
        self.store_attention = False
        self.res_attention = True
        self.pre_norm = False
        self.pe = 'zeros'
        self.learn_pe = True
        
        self.weight_path = None