import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(TSMamba, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        d_model = 16
        n_layer = 1

        self.proj = nn.Sequential(
            nn.Linear(in_features=configs.enc_in, out_features=d_model//2),
            nn.SiLU(),
            nn.Linear(in_features=d_model//2, out_features=d_model),
            nn.LayerNorm(normalized_shape=d_model, eps=1e-5),
            nn.ReLU()
        )

        self.mamba = MixerModel(d_model=d_model,
                                n_layer=n_layer,
                                vocab_size=self.seq_len,
                                rms_norm=False,
                                fused_add_norm=False,
                                residual_in_fp32=False,
                                )
        self.linear = nn.Linear(self.seq_len, self.pred_len)

        self.head = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=configs.enc_in),
        )

        # Initialize weights and apply final processing
        initializer_cfg = None
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def forward(self, x):
        # x: [Batch, Input length, Channel]

        x = self.proj(x)
        x = self.mamba(x)
        x = self.linear(x.permute(0,2,1)).permute(0,2,1)
        x = self.head(x)

        return x # [Batch, Output length, Channel]