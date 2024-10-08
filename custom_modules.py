# This code is newly added and different from RETFound
# Partly revised by YP @Anhui Medical University
# --------------------------------------------------------
import torch
import torch.nn as nn
from timm.models.layers import DropPath

class Attention(nn.Module):
    def __init__(self, 
                 dim,   #    token  dim
                 num_heads=8,  #  number of heads   
                 qkv_bias=False,
                 attn_drop=0., 
                 proj_drop=0.):
        super().__init__()
        r = 4 # the rank of LoRA in FMUE
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads  #Dimension of the Head
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # LoRA layers added in FMUE
        self.linear_a_q = nn.Linear(dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, dim, bias=False)
        self.linear_a_v = nn.Linear(dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):#   򴫲     
        #      [batch_size, 
        #      num_patches+1,   base16ģ ͵        14*14  
        #      total_embed_dim  base16ģ ͵        768  ]
        B, N, C = x.shape
        qkv = self.qkv(x)

        # LoRA layers
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # make torchscript happy (cannot use tensor as tuple)
        #q  k  v  С  [batchsize, num_heads, num_patches+1, embed_dim_per_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
      
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
        x = self.proj(x) 
        x = self.proj_drop(x)
  
        return x

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #mlp                 4    ʵ    һ  MLPģ   ʱ    Ҫ    mlp_hidden_dim              ڴ   ǰ    
        self.mlp = Mlp(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        residual = x
        x = self.drop_path(self.mlp(self.norm2(x)))

        x = residual + x
        return x

