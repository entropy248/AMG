import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=False)
        )


def PositionEmbeddingSine(h, w, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi
    mask = torch.ones([1, h, w], device='cuda')
    y_embed = mask.cumsum(1, dtype=torch.float32)
    x_embed = mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device='cuda')
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    b, c, h, w = pos.shape
    pos = pos.reshape(b, c, h*w).permute(0, 2, 1)
    class_pos = torch.zeros([1, 1, c], device='cuda')
    pos = torch.cat((class_pos, pos), dim=1)

    return pos


class PositionEmbs(nn.Module):
    def __init__(self, h, w, num_patches, emb_dim, dropout_rate=0.1, embedding_type='learned'):
        super(PositionEmbs, self).__init__()
        if embedding_type == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
            # print(self.pos_embedding.shape)
        elif embedding_type == 'sine':
            n_steps = emb_dim // 2
            self.pos_embedding = PositionEmbeddingSine(h, w, n_steps, normalize=True)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = x + self.pos_embedding

        if self.dropout:
            out = self.dropout(out)

        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out

    def flops(self, N):
        flops = 0
        (in_dim, mlp_dim) = self.fc1.weight.data.shape
        flops += N * in_dim * mlp_dim
        flops += N * mlp_dim * in_dim
        return flops


class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()

        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class ConvCompress(nn.Module):
    def __init__(self, in_dim, out_dim, ratio, groups):
        super(ConvCompress, self).__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, ratio, stride=ratio, groups=groups)

    def forward(self, mem):
        b, N, h, dv = mem.shape
        mem = mem.reshape(b, N, -1).permute(0, 2, 1)
        com_mem = self.conv(mem)
        return com_mem.permute(0, 2, 1).reshape(b, -1, h, dv)


class SelfAttention(nn.Module):
    def __init__(self, block_dim, in_dim, num_patches, channel=None, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()

        self.head_dim = in_dim // heads
        assert block_dim % self.head_dim == 0
        self.heads = block_dim // self.head_dim
        self.scale = self.head_dim ** 0.5

        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))

        if channel is None:
            self.channel = num_patches
        else:
            self.channel = channel

        mask = torch.ones(self.channel + 1)
        mask[0] = 0
        self.index = nn.Parameter(torch.nonzero(mask).reshape(-1), requires_grad=False)

        # to save the features and the ranks
        self.q = None
        self.k = None
        self.v = None

        self.ck = None
        self.cv = None
        self.attn = None
        self.cnt = 0

        self.seq_ranks = None
        self.head_ranks = None

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x,
                compute_taylor_h=False,
                compute_taylor_n=False,
                compute_taylor_attn=False,
                compute_entropy=False):
        # x:(b,N,d)
        b, n, _ = x.shape
        # w(d,h,dv) (x[2] dot w[0]) --> (b,N,h,dv)
        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        # index layer, choose the kept tokens. The class token is excluded for importance estimation
        k_h, k = k[:, 0:1, :, :], k[:, self.index, :, :]
        v_h, v = v[:, 0:1, :, :], v[:, self.index, :, :]

        if compute_taylor_n:
            self.ck = k
            self.cv = v
            k.register_hook(self.compute_rank_ck)
            v.register_hook(self.compute_rank_cv)
        # print(head_value.shape, v.shape)
        k = torch.cat([k_h, k], dim=1)
        v = torch.cat([v_h, v], dim=1)

        # (b,N,h,dv) --> (b,h,N,dv)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if compute_taylor_h:
            self.q = q
            self.k = k
            self.v = v
            q.register_hook(self.compute_rank_q)
            k.register_hook(self.compute_rank_k)
            v.register_hook(self.compute_rank_v)

        # (b,h,N,dv) matmul (b,h,dv,N) -->(b,h,N,N)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        if compute_taylor_attn:
            self.attn = attn_weights
            attn_weights.register_hook(self.compute_rank_attn)

        if compute_entropy:
            if self.attn is None:
                self.attn = torch.zeros(attn_weights.shape).cuda()
            if self.attn.shape[0] == attn_weights.shape[0]:
                self.attn += attn_weights
                self.cnt += 1
        # (b,h,N,N) matmul (b,h,N,dv) --> (b,h,N,dv)
        out = torch.matmul(attn_weights, v)
        # (b,h,N,dv) --> (b,N,h,dv)
        out = out.permute(0, 2, 1, 3)
        # out:(h,dv,d)  (b,N,h,dv)[2,3] dot (h,dv,d)[0,1] --> (b,N,d)
        out = self.out(out, dims=([2, 3], [0, 1]))

        return out

    def compute_rank_q(self, grad):
        values = torch.sum((grad * self.q), dim=0, keepdim=True)\
            .sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
        values = values / (self.q.size(0) * self.q.size(2) * self.q.size(3))

        if self.head_ranks is None:
            self.head_ranks = torch.zeros(grad.size(1)).cuda()

        self.head_ranks += values

    def compute_rank_k(self, grad):
        values = torch.sum((grad * self.k), dim=0, keepdim=True) \
                     .sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
        values = values / (self.k.size(0) * self.k.size(2) * self.k.size(3))

        if self.head_ranks is None:
            self.head_ranks = torch.zeros(grad.size(1)).cuda()

        self.head_ranks += torch.abs(values)

    def compute_rank_v(self, grad):
        values = torch.sum((grad * self.v), dim=0, keepdim=True) \
                     .sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
        values = values / (self.v.size(0) * self.v.size(2) * self.v.size(3))

        if self.head_ranks is None:
            self.head_ranks = torch.zeros(grad.size(1)).cuda()

        self.head_ranks += torch.abs(values)

    def compute_rank_ck(self, grad):

        values = torch.sum((grad * self.ck), dim=0, keepdim=True) \
                     .sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
        values = values / (self.cv.size(0) * self.cv.size(2) * self.cv.size(3))

        if self.seq_ranks is None:
            self.seq_ranks = torch.zeros(grad.size(1)).cuda()

        self.seq_ranks += torch.abs(values)

    def compute_rank_cv(self, grad):

        values = torch.sum((grad * self.cv), dim=0, keepdim=True) \
                     .sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
        values = values / (self.cv.size(0) * self.cv.size(2) * self.cv.size(3))
        if self.seq_ranks is None:
            self.seq_ranks = torch.zeros(grad.size(1)).cuda()

        self.seq_ranks += torch.abs(values)


    def compute_rank_attn(self, grad):
        # calculate gradient-weighted simmilarity and sum along query dimension
        values = torch.sum((grad * self.attn), dim=0, keepdim=True) \
                      .sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)[0, 0, 0, 1:].data

        values = values / (self.attn.size(0) * self.attn.size(1) * self.attn.size(2))

        if self.seq_ranks is None:
            self.seq_ranks = torch.zeros(grad.size(3)-1).cuda()

        self.seq_ranks += torch.abs(values)

    def reset_rank(self):
        self.q = None
        self.k = None
        self.v = None
        self.ck = None
        self.cv = None
        self.attn = None
        self.seq_ranks = None
        self.head_ranks = None
        self.cnt = 0

    def flops(self):
        flops = 0
        total_dim = self.head_dim * self.heads
        # q.k.v dot x
        flops += 3 * self.channel * total_dim * total_dim
        # attn = q matmul k.transpose
        flops += self.channel * total_dim * self.channel
        # softmax
        flops += self.heads * self.channel * self.channel
        # out = attn matmul v
        flops += self.channel * total_dim * self.channel
        # self.out dot out
        flops += self.channel * total_dim * total_dim
        return flops


class EncoderBlock(nn.Module):
    def __init__(self, block_dim, in_dim, mlp_dim, num_heads, num_patches, channel=None, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.num_patches =num_patches
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(block_dim, in_dim, num_patches, channel=channel, heads=num_heads, dropout_rate=attn_dropout_rate)
        # self.attn.init_conv()
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x,
                compute_taylor_h=False,
                compute_taylor_n=False,
                compute_taylor_attn=False,
                compute_entropy=False):
        # out = residual = self.res1(x)
        residual = x
        # (b,n,d)    (d)
        out = self.norm1(x)

        out = self.attn(out, compute_taylor_h, compute_taylor_n, compute_taylor_attn, compute_entropy)
        if self.dropout:
            out = self.dropout(out)
        out += residual

        residual = out

        out = self.norm2(out)

        out = self.mlp(out)

        out += residual
        return out

    def flops(self):
        flops = 0
        flops += self.attn.flops()
        flops += self.mlp.flops(self.num_patches+1)
        return flops


class Encoder(nn.Module):
    def __init__(self, h, w, num_patches, emb_dim, mlp_dim, channels=None,
                 block_settings=[768]*12, num_layers=12, num_heads=12,
                 dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()

        # positional embedding
        self.pos_embedding = PositionEmbs(h, w, num_patches, emb_dim, dropout_rate)

        if channels is None:
            channels = [None] * num_layers

        # encoder blocks
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(block_settings[i], in_dim, mlp_dim, num_heads,
                                 num_patches, channels[i], dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x,
                compute_taylor_h=False,
                compute_taylor_n=False,
                compute_taylor_attn=False,
                compute_entropy=False):

        out = self.pos_embedding(x)

        for layer in self.encoder_layers:
            out = layer(out, compute_taylor_h, compute_taylor_n, compute_taylor_attn, compute_entropy)
        out = self.norm(out)
        return out

    def flops(self):
        flops = 0
        for layer in self.encoder_layers:
            flops += layer.flops()
        return flops


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self,
                 image_size=(384, 384),
                 patch_size=(16, 16),
                 emb_dim=768,
                 mlp_dim=3072,
                 num_heads=12,
                 num_layers=12,
                 num_classes=1000,
                 block_settings=[768]*12,
                 channels=None,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 feat_dim=None):
        super(VisionTransformer, self).__init__()
        h, w = image_size

        # embedding layer
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # transformer
        self.transformer = Encoder(
            gh, gw,
            num_patches=num_patches,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            block_settings=block_settings,
            channels=channels,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate)

        # classfier
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x,
                compute_taylor_h=False,
                compute_taylor_n=False,
                compute_taylor_attn=False,
                compute_entropy=False):
        emb = self.embedding(x)     # (n, c, gh, gw)
        emb = emb.permute(0, 2, 3, 1)  # (n, gh, hw, c)
        b, h, w, c = emb.shape
        emb = emb.reshape(b, h * w, c)

        # prepend class token
        cls_token = self.cls_token.repeat(b, 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)

        # transformer
        feat = self.transformer(emb, compute_taylor_h, compute_taylor_n, compute_taylor_attn, compute_entropy)

        # classifier
        logits = self.classifier(feat[:, 0])
        return logits

    def flops(self):
        return self.transformer.flops()


if __name__ == '__main__':
    model = VisionTransformer(num_layers=2)
    print(model.flops())

    x = torch.randn((2, 3, 384, 384))
    out = model(x)

    state_dict = model.state_dict()

    for key, value in state_dict.items():
        print("{}: {}".format(key, value.shape))
    # pos = PositionEmbeddingSine(24, 24, 384, normalize=True)
    # print(pos.shape)










