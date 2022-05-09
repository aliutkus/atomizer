import torch
from torch import nn
import timm 
from timm.models.layers import Mlp, DropPath
import pytorch_lightning as pl
from torch import Tensor
import typing
from einops import rearrange


class Attention(nn.Module):
    """modified from timm to allow for kv and q to have different lengths and for linear attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_drop=0., linear = False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # queries and key/values don't have the same lengths in general
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = linear

    def forward(self, reference, target=None):
        """
        reference and target are dicts {'data':Tensor, 'masks':None or Tensor}
        """
        B, N, C = reference.shape
        if target is None:
            # self attention
            target = reference

        assert (
            (reference['masks'] is None and target['masks'] is None)
            or (reference['masks'] and target['masks'])
        ), 'the masks must both be either provided or None'

        M = target.shape[1]

        kv = self.kv(reference['data']).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = self.q(target['data']).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (B, H, N, D)

        if not self.linear:
            # explicitly compute attention matrix
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if reference['masks'] is not None:
                # mask if needed (due to padding)
                attn_mask = (
                    rearrange(target['masks'], 'b i -> b 1 i 1') *
                    rearrange(reference['masks'], 'b j -> b 1 1 j')
                )
                mask_value = -torch.finfo(attn.dtype).max
                attn = attn.masked_fill(~attn_mask, mask_value)

            attn = attn.softmax(dim=-1)

            # get output by multiplying attention matrix with values 
            x = attn @ v
        else:
            # need linear attention. apply feature maps on queries/keys
            q = torch.nn.functional.elu(q) + 1
            k = torch.nn.functional.elu(k) + 1

            if reference['masks'] is not None:
                # mask if needed (due to padding). Maybe we could mask q and k directly up there
                # instead of masking attention even in the nonlinear case ?
                mask_value = -torch.finfo(attn.dtype).max

                # q and k are (b, h, n, d)
                q_mask = rearrange(target['masks'], 'b i -> b 1 i 1')
                k_mask = rearrange(reference['masks'], 'b i -> b 1 i 1')
                q.masked_fill(~q_mask, mask_value)
                k.masked_fill(~k_mask, mask_value)

            x = k.transpose(-2, -1) @ v
            x = q @ x * self.scale

            # Using normalization over the lines of the attention
            x = x / (1e-4+q @ k.transpose(-2, -1).sum(dim=-1, keepdim=True))

        # reshape properly
        x = x.transpose(1, 2).reshape(B, M, C)

        # final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    """modified from timm to allow for (cross+/linear) attention. Just hard-wiring LayerNorm"""
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, linear_attention=False, 
            drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop, linear=linear_attention)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, reference, target=None):
        # if target is None, it will default to the reference in the attention module
        target = target + self.drop_path1(self.ls1(self.attn(self.norm1(reference), target)))
        target = target + self.drop_path2(self.ls2(self.mlp(self.norm2(target))))
        return target


class PointSineEmbedder(nn.Module):
    """
    an n-dimensional position embedder with sinusoidal encoding
    """
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.dim = dim
        self.embed = nn.Linear(dim, embed_dim // 2)
 
    def forward(self, locations):
        original_shape = locations.shape[:-1]

        locations = locations.view(-1, self.dim)

        # first have some sinusoidal pos encoding
        x = self.embed(locations) # (batchize, embed_dim // 2)
        x = torch.concat([torch.cos(x), torch.sin(x)], dim=-1) # (batchsize, embed_dim)
        return x.view(original_shape + (self.embed_dim,))

class Atomizer(pl.LightningModule):
    def __init__(
        self,
        num_chans,
        num_velocities = 256,
        dynamic_range = 90,
        num_freqs=1024,
        num_times= 1024,
        embed_dim=512,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        linear_attention=True
    ):
        super().__init__()

        # --------------------------------------------------------------------------

        # learnable velocity embeddings
        self.dynamic_range = dynamic_range
        self.num_velocities = num_velocities
        self.velocity_embed = nn.Embedding(self.num_velocities, embed_dim)

        # (channel, time, frequency, sign) atoms support encoding.
        self.num_chans = num_chans
        self.num_freqs = num_freqs
        self.num_times = num_times
        self.tf_spe = PointSineEmbedder(2, embed_dim)
        self.freqs_ape = nn.Embedding(num_freqs+1, embed_dim) # last entry for padding
        self.times_ape = nn.Embedding(num_times+1, embed_dim) # last entry for padding
        self.signs_ape = nn.Embedding(2 + 1, embed_dim) # 0 (positive), 1 (negative), 2 (padding)
        self.chans_ape = nn.Embedding(num_chans+1, embed_dim) # last entry for padding
        self.mlp_pe = Mlp(embed_dim)
        self.mask_token = torch.Parameter(torch.randn(1, 1, embed_dim)*0.02) # std 0.02 from timm

        # each predictors as a different mlp (classif problem)
        self.times_predictor = Mlp(embed_dim, embed_dim*4, num_times)
        self.freqs_predictor = Mlp(embed_dim, embed_dim*4, num_freqs)
        self.chans_predictor = Mlp(embed_dim, embed_dim*4, num_chans)
        self.signs_predictor = Mlp(embed_dim, embed_dim*4, 2)

        # encoder blocs
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    linear_attention=linear_attention
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    linear_attention=linear_attention
                )
                for i in range(depth)
            ]
        )

        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, context, target):
        """
            context is a dict with keys 'freqs', 'times', 'chans', 'signs', 'velocities', 'masks','lens'. All values
            are (batchsize, max_context_len) Tensors except 'lens' that just provide each sample individual length.
                * freqs: in [0, self.num_freqs-1]
                * times: int frame of the atoms. unbounded
                * chans: in [0, self.num_chans-1]
                * signs: in [-1, 1]
                * velocities: lower than 0
                * masks: just the masks for the sequence: masks[b, i]= 1 if i<lens[b], 0 otherwise

            target: Dict with keys 'velocities', 'masks', 'lens'.
                * 'velocities' is (batchsize, max_target_len)
                * 'masks': (batchsize, max_target_len), definied similarly as for context
                * 'lens' is (batchsize,). It gives the original length of each sample

        """
        batchsize = context['len'].shape[0]

        # making velocities integer
        velocities = []
        for current in context, target:
            vel = current['velocities']
            vel = vel.clamp(-self.dynamic_range, 0)
            vel = (-vel) * self.num_velocities / self.dynamic_range
            vel = vel.to(int)

            # setting the padded velocities index to the padding embedding
            for sample, sample_len in enumerate(current['lens']):
                vel[sample, sample_len:] = self.num_velocities
            velocities.append(vel)
        context_embed, target_embed = velocities

        # make signs in [0 (negative), 1 (positive), 2 (padding)]
        # making signs in [0, 1]
        context_signs = context['signs'] # originally -1 (negative), 1 (positive), 0 (padding)
        context_signs[context_signs==0] = 2 # padded => 2
        context_signs = torch.where(context_signs < 0, 1, context_signs) # -1 => 1 (positive)
        context_signs = context_signs.to(int)

        # setting the padded chan index to the padding embedding
        context_chans = context['chans']
        for sample, sample_len in enumerate(context['lens']):
            context_chans[sample, sample_len:] = self.num_chans
        context_chans = context_chans.to(int)

        # compute velocity embeddings 
        context_embed = self.velocity_embed(context_embed) # (batchsize, context_max_len, embed_dim)
        target_embed = self.velocity_embed(target_embed) # (batchsize, target_max_len, embed_dim)

        # compute context embeddings (batchsize, context_max_len, embed_dim)
        context_embed = context_embed + self.tf_spe(
            torch.stack( (context['times'], context['freqs']) , dim=-1)
        ) # joint (time_frequency) sine positional embedding
        context_embed = context_embed + self.times_ape(context['times']) # absolute time embedding
        context_embed = context_embed + self.freqs_ape(context['freqs']) # absolute frequency embedding
        context_embed = context_embed + self.signs_ape(context_signs) # absolute signs embedding
        context_embed = context_embed + self.chans_ape(context_chans) # absolute chans embedding
        context_embed = self.mlp_pe(context_embed) # and finally some additional MLP
        context_embed = {'data':context_embed, 'masks':context['masks']}

        # add the mask embedding to the target embedding
        target_embed = target_embed + self.mask_token
        target_embed = {'data':target_embed, 'masks': target['masks']}
        
        # we are ready. apply encoder
        for blk in self.encoder_blocks:
            context_embed['data'] = blk(context_embed)

        # now, apply decoder. target_embed is (batchsize, target_max_len, embed_dim)
        for blk in self.decoder_blocks:
            target_embed['data'] = blk(context_embed, target_embed)

        # now, decode
        times = self.times_predictor(target_embed)
        freqs = self.freqs_predictor(target_embed)
        chans = self.chans_predictor(target_embed)
        signs = self.signs_predictor(target_embed)

        return times, freqs, chans, signs