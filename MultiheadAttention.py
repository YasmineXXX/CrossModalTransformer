import torch, math
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value.transpose(-1, -2)), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, d_model, h_w, config):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        self.config = config
        assert d_model % head == 0
        # print(d_model)
        # We assume d_v always equals d_k
        self.d_k = d_model // head
        self.head = head
        self.h_w = h_w
        self.linears_q = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.linears_k_v = nn.Sequential(
            nn.Linear(h_w, h_w),
            nn.Linear(h_w, h_w),
            nn.Linear(h_w, h_w),
            nn.Linear(h_w, h_w)
        )
        self.attn = None
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, query, key, value, mask=None):
        # query: bsz, words, d
        # key, value: bsz, d, H*W
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.linears_q(query)
        seg_bsz = query.size(0)
        query = query.view(seg_bsz, -1, self.head, self.d_k).transpose(1, 2)
        key, value = \
            [l(x).view(seg_bsz, self.head, self.d_k, self.h_w)
             for l, x in zip(self.linears_k_v, (key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(seg_bsz, -1, self.head * self.d_k)
        return self.linears_q[-1](x)