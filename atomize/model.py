import torch
from torch import nn
import torch.nn.functional as F
import timm 
from timm.models.layers import Mlp, DropPath
import pytorch_lightning as pl
from torch import Tensor
import typing
from einops import rearrange
from omegaconf import OmegaConf
import hydra

class Attention(nn.Module):
    """modified from timm to allow for:
    * kv and q to have different lengths
    * linear attention
    * returning the attention (no values)
    """
    def __init__(self,
            dim, num_heads=8, qkv_bias=False,
            proj_drop=0., linear = False, values=True):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        assert values or (not values and not linear), 'cannot return attention (values=False) and do linear attention'

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # queries and key/values don't have the same lengths in general
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.values = values
        if values:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = linear

    def forward(self, target, reference):
        """
        reference and target are dicts {'data':Tensor, 'masks':None or Tensor}
        """
        if reference is None:
            # self attention
            reference = target
        B, N, C = reference['data'].shape

        assert (
            (reference['masks'] is None and target['masks'] is None)
            or (reference['masks'] is not None and target['masks'] is not None)
        ), 'the masks must both be either provided or None'

        M = target['data'].shape[1]

        if self.values:
            # we need k and v
            kv = self.kv(reference['data']).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        else:
            # we don't need values
            k = self.k(reference['data']).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.q(target['data'])
        if q.shape[0] == 1:
            # there may be a bug here if q does not have the same batchsize, but 1.
            # this may happen if the target does not comprise any information, but that everything
            # is masked. in that case, we expand the query
            q = q.repeat(B, 1, 1)
        #import ipdb; ipdb.set_trace()
        q = q.reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (B, H, N, D)
        inf = torch.finfo(q.dtype).max

        if not self.linear:
            # explicitly compute attention matrix
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if reference['masks'] is not None:
                # mask if needed (due to padding)
                attn_mask = (
                    rearrange(target['masks'], 'b i -> b 1 i 1') *
                    rearrange(reference['masks'], 'b j -> b 1 1 j')
                )
                attn = attn.masked_fill(~attn_mask.to(bool), -inf)

            attn = attn.softmax(dim=-1)

            if not self.values:
                return attn

            # get output by multiplying attention matrix with values 
            x = attn @ v
        else:
            # need linear attention. apply feature maps on queries/keys
            q = torch.nn.functional.elu(q) + 1
            k = torch.nn.functional.elu(k) + 1

            if reference['masks'] is not None:
                # q and k are (b, h, n, d). filling the masked ones
                # with zeros so that they don't count in the computation
                q.masked_fill(~(target['masks'][:, None, :, None].to(bool)), 0)
                k.masked_fill(~(reference['masks'][:, None, :, None].to(bool)), 0)

            x = k.transpose(-2, -1) @ v
            x = q @ x * self.scale

            if reference['masks'] is not None:
                # and finally set the masked ones to -inf
                x.masked_fill(~(target['masks'][:, None, :, None].to(bool)), -inf)

            # Using normalization over the lines of the attention
            x = x / (1e-4+q @ (k.transpose(-2, -1).sum(dim=-1, keepdim=True)))

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


def norm(x, norm_fn):
    """
    applies a norm_fn to the `data` entry of a dict `x`, with keys 'data' and 'masks'
    """
    return {'data':norm_fn(x['data']), 'masks':x.get('masks')}


class Block(nn.Module):
    """modified from timm to allow for :
    - being and encoder or decoder block (decoder=True/False)
    - allowing linear Attention

    Just hard-wiring LayerNorm"""
    def __init__(
            self, embed_dim, num_heads, mlp_ratio=4., qkv_bias=False, linear_attention=False, 
            drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU,
            encoder=True):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(
            embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop, linear=linear_attention)
        self.ls1 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.encoder = encoder
        if not encoder:
            self.norm_cross_ref = nn.LayerNorm(embed_dim)
            self.norm_cross_target = nn.LayerNorm(embed_dim)

            self.attn_cross = Attention(
                embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_drop=drop, linear=linear_attention)
            self.ls_cross = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path_cross = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, other=None):
        # the first part is the same for encoder and decoder: applies to x only
        x_norm = norm(x, self.norm1)

        x_masks = x['masks']
        x = x['data'] + self.drop_path1(self.ls1(self.attn(x_norm, x_norm)))

        if self.encoder is False:
            # for the decoder, there is cross attention
            x_norm = {'data':self.norm_cross_target(x), 'masks':x_masks}
            other_norm = norm(other, self.norm_cross_ref)
            x = x + self.drop_path_cross(self.ls_cross(self.attn_cross(x_norm, other_norm)))

        # now mlp
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class SinusoidalEmbedder(nn.Module):
    """
    an n-dimensional position embedder with sinusoidal encoding
    """
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.embed = nn.Linear(dim, embed_dim // 2)
 
    def forward(self, locations):

        if self.dim > 1:
            # input has shape, say (batchsize, ...., dim)
            original_shape = locations.shape[:-1]
        else:
            # input has shape, say (batchsize, ....)
            original_shape = locations.shape

        locations = locations.view(-1, self.dim)

        # first have some sinusoidal pos encoding
        x = self.embed(locations.to(self.embed.weight.dtype)) # (batchize, embed_dim // 2)
        x = torch.concat([torch.cos(x), torch.sin(x)], dim=-1) # (batchsize, embed_dim)
        return x.view(original_shape + (self.embed_dim,))


class AtomsLocator(nn.Module):
    def __init__(
        self,
        atomizer=None,
        depth=12,
        num_heads=5,
        embed_dim=360,
        mlp_ratio=4,
        linear_attention=True,
    ):
        super().__init__()
        # saving the atomizer
        self.atomizer = atomizer

        # dimensions given by the atomizer
        self.features = atomizer.features

        # Feature-wise encodings:
        # when cardinality is given, feature-wise encoding is done through a standard embedding
        self.featurewise_encoding = nn.ModuleDict({
            key:nn.Embedding(self.features[key].cardinality, embed_dim)
            for key in self.features if self.features[key].cardinality
        })
        # When cardinality is None (dynamic features), feature-wise encoding is done with sinusoidal pos encoding
        self.featurewise_encoding.update({
            key:SinusoidalEmbedder(1, embed_dim)
            for key in self.features if self.features[key].cardinality is None
        })

        # joint-sinusoidal encoding for location features
        self.location_features = [name for name in self.features if self.features[name].is_location]
        if len(self.location_features):
            self.joint_location_encoding = SinusoidalEmbedder(len(self.location_features), embed_dim)
        else:
            self.joint_location_encoding = None

        # mask encoding for the missing information
        self.masks = nn.ParameterDict({
            key:nn.Parameter(torch.randn(1, 1, embed_dim)*0.02) 
            for key in self.features})

        # final MLP for the embedding
        self.mlp_pe = Mlp(embed_dim, embed_dim * mlp_ratio, embed_dim)

        # each predictors as a different mlp to convert the joint encoding to an estimated
        # feature-wise codes
        self.predictors = nn.ModuleDict({
            key:Mlp(
                embed_dim,
                embed_dim * mlp_ratio,
                embed_dim
            )
            for key in self.features})

        """self.decoding_attentions = nn.ModuleDict({
            key: Attention(
                embed_dim, num_heads=1, qkv_bias=True,
                linear = False, values=False)
            for key in self.features
        })"""
        self.decoder_norm_q = nn.LayerNorm(embed_dim)
        self.decoder_norm_k = nn.LayerNorm(embed_dim)

        # encoder blocs
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    linear_attention=linear_attention,
                    encoder=True,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    linear_attention=linear_attention,
                    encoder=False
                )
                for i in range(depth)
            ]
        )
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

    def forward(self, context, target=None):
        """
            1. context
            provides a description for the known information to serve as a conditionner
            this information may be incomplete, with some keys missing, but the typical
            usecase is to have a completely known context, as in the case of generation

            context is a dict of keys:values.
                The keys that will be used are those found in the atomizer.features.
                    The values are (batch_size, num_atoms).
                There is an additional 'masks' key, which is a binary vector
                    masks[b, i]= 1 if i<original length[b], 0 otherwise

                There may be other entries we don't care about.

            2. target
            the target provides a description of the information that is available regarding
            the data to generate.

                same structure as context.
        
        
        Output:
        a Tensor with shape (batchsize, target_length, embed_dim) provides the encoded
        to be used with the decoder to actually predict desired keys.
        """
        # batchsize is the first dimension of any of the entries
        batchsize = context[list(context.keys())[0]].shape[0]
        embeddings = []
        predict = {}

        # build the embedding for context and target
        for current in (context, target):
            # handling the case target is None
            if not current:
                embeddings.append(None)
                continue

            # create the embeddings as a sum of all the individual embeddings
            embed = 0
            for key in self.features:
                value = current.get(key)
                if value is not None:
                    if key in self.featurewise_encoding:
                        # add the absolute embedding for this key
                        embed = embed + self.featurewise_encoding[key](value)
                else:
                    # not present: add a mask and mark this key as needing predict
                    if current is target: predict[key] = True
                    embed = embed + self.masks[key]

            # joint sinusoidal positional encoding for the locations
            locations = [current.get(feature) for feature in self.location_features]
            if  all([v is not None for v in locations]):
                # all locations are given. add their joint embedding
                embed = embed + self.joint_location_encoding(
                    torch.stack( locations , dim=-1)
                )

            # finally some additional MLP
            embed = self.mlp_pe(embed)
            embeddings.append({'data':embed, 'masks':current.get('masks')})

        context_embed, target_embed = embeddings

        # we are ready. apply encoder
        for blk in self.encoder_blocks:
            context_embed['data'] = blk(context_embed)

        if target_embed is None:
            # if we're not provided with a target, just apply the predictors to the context
            target_embed = context_embed
        else:
            # we have a target. apply decoder target_embed is (batchsize, target_max_len, embed_dim)
            for blk in self.decoder_blocks:
                target_embed['data'] = blk(target_embed, context_embed)

        return target_embed['data']

    def decode(self, embeddings, queries, all_scores=True):
        """
        decodes embeddings to provide scores for the different queries.

        embeddings: A Tensor with shape (batchsize, length, embed_dim)
                    that has been produced by the model to encode a 
                    batch of (context, target) pairs.
        queries: A dict of key: values representing the features we want to
                 predict.

            key: one of the entries of the atomizer.features
            values: a Tensor with shape (num_classes) indicating the candidate values
                    to check for that key. 
                    if None: if key is a feature with known cardinality, will check
                               all values in [0...feature.cardinality-1] by default.
                               will error otherwise (we don't know which values to test
                               for dynamic features like time).     
        """
        # making embeddings (batchsize*length, embed_dim)
        (batch_size, length, embed_dim) = embeddings.shape
        embeddings = embeddings.view(-1, embed_dim)

        result = {}
        for feature in queries:
            if feature not in self.features:
                continue
            if queries[feature] is None:
                # user wants to test all the possibilities for this key
                if self.features[feature].cardinality is None:
                    # the feature cannot be dynamic in that case
                    raise ValueError(f'Query for {feature} cannot have a None value because {feature} is a dynamic feature (None cardinality)')
                queries[feature] = torch.arange(
                    0, self.features[feature].cardinality,
                    device=embeddings.device
                )

            # we have a list of query values. compute their encodings
            # q is (num_queries, embed_dim) Tensor
            q = self.featurewise_encoding[feature](queries[feature])
            q = self.decoder_norm_q(q)

            # get the decoded embeddings for that particular feature
            # (batchsize*len, embed_dim)
            decoded = self.predictors[feature](embeddings)
            decoded = decoded.view(batch_size, length, embed_dim)
            decoded = self.decoder_norm_k(decoded)

            """score = self.decoding_attentions[feature](
                {'data':q[None], 'masks':None},
                {'data':decoded, 'masks':None}
            ) # (batchsize, 1, len, num_queries)
            score = score[:, 0].transpose(-2, -1) # (batchsize, num_queries, len)

            """
            # dot product between embedings and query (batchsize*len, num_queries)
            score = (decoded @ q.transpose(-2, -1)) * (embed_dim ** -0.5)
            

            ## result is softmax over the query dimension (to make it a probability)
            result[feature] = score.softmax(dim=-1).view(batch_size, length, -1)

        return result

class System(pl.LightningModule):
    def __init__(
            self,
            model,
            optimizer_partial,
            scheduler_partial,
            lr_frequency
        ):
        super().__init__()        
        self.model = model
        self.optimizer = optimizer_partial(params=model.parameters())
        self.scheduler = {
            "scheduler": scheduler_partial(optimizer=self.optimizer),
            "monitor": "train/loss",  # Default: val_loss
            "interval": "step",
            "frequency": lr_frequency,
        }

        
    def training_step(self, batch, batch_idx):
        context, target, ground_truth = batch
        embeddings = self.model(context, target)

        # building the feature queries for decoding. If the feature has
        # a known cardinality, let the model choose the queries.
        # otherwise (for dynamic cardinality e.g. time), let the query lie
        # inside the context range.
        # doing this for all features that are not in target (the ones to predict)
        queries = {
            feature: (None if self.model.features[feature].cardinality is not None
            else torch.arange(
                ground_truth[feature].min(), ground_truth[feature].max()+1,
                device = ground_truth[feature].device))
            for feature in self.model.features if feature not in target
        }

        # get the output class likelihood
        estimate = self.model.decode(embeddings, queries)

        loss = 0
        for i, key in enumerate(estimate):
            cardinality = estimate[key].shape[-1]
            key_loss = F.cross_entropy(
                estimate[key].contiguous().view(-1, cardinality), # estimate (batch*natoms, num_classes)
                ground_truth[key].flatten() # target
            ) / torch.log(torch.tensor(cardinality))
            self.log(f"train/{key}_loss", key_loss)
            loss = loss + key_loss
        pred = {**target, **{key:estimate[key].argmax(dim=-1) for key in estimate}}
        self.log("train/loss", loss)
        return {"loss": loss, "pred": pred}

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


def get_system(system_cfg, atomizer):
    model = hydra.utils.instantiate(system_cfg.model, atomizer=atomizer)
    system = hydra.utils.instantiate(system_cfg, model=model)
    return system