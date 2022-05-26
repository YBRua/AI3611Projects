import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # pe: 1, L, D

    def forward(self, x: torch.Tensor):
        # x: B, L, D
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            input_dim: int,
            h_dim: int,
            n_heads: int):
        super().__init__()

        if h_dim % n_heads != 0:
            raise ValueError('h_dim must be divisible by n_heads')

        self.k_linear = nn.Linear(input_dim, h_dim)
        self.q_linear = nn.Linear(input_dim, h_dim)
        self.v_linear = nn.Linear(input_dim, h_dim)
        self.n_heads = n_heads
        self.d_k = h_dim // n_heads

    def forward(
            self,
            k: torch.Tensor,
            q: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor = None):
        # B, L, nheads, D
        k = self.k_linear(k).reshape(*k.shape[:-1], self.n_heads, self.d_k)
        q = self.q_linear(q).reshape(*q.shape[:-1], self.n_heads, self.d_k)
        v = self.v_linear(v).reshape(*v.shape[:-1], self.n_heads, self.d_k)

        attn_w = torch.einsum('...mhd, ...nhd -> ...mnh', k, q)
        attn_w = attn_w / math.sqrt(self.d_k)  # B, L, L, nheads
        if mask is not None:
            mask = mask.unsqueeze(-1)
            attn_w = attn_w.masked_fill(mask, float('-inf'))
        attn_w = torch.softmax(attn_w, dim=-2)

        # B, nheads, D
        out = torch.einsum('...mnh, ...nhd -> ...mhd', attn_w, v)
        return out.flatten(-2)


class BottlenectedAttention(nn.Module):
    def __init__(
            self,
            audio_embd: int,
            video_embd: int,
            num_classes: int,
            hidden_dim: int = 320,
            n_heads: int = 10,
            btnk_len: int = 4):
        super().__init__()

        if audio_embd != video_embd:
            raise ValueError(
                '唉. audio_embd != video_embd is not supported 啊.')

        self.audio_tok = nn.Parameter(
            torch.zeros((1, 1, audio_embd)),
            requires_grad=True)
        self.video_tok = nn.Parameter(
            torch.zeros((1, 1, audio_embd)),
            requires_grad=True)
        self.bottleneck_toks = nn.Parameter(
            torch.zeros((1, btnk_len, audio_embd)))
        self.btnk_len = btnk_len

        self.pos_embd = PositionalEncoding(audio_embd)

        self.attn = MultiHeadAttention(audio_embd, hidden_dim, n_heads)
        self.ln = nn.LayerNorm(hidden_dim)

        self.aud_pred = nn.Sequential(
            nn.Linear(hidden_dim, num_classes))
        self.vid_pred = nn.Sequential(
            nn.Linear(hidden_dim, num_classes))

    def _get_btnk_mask(self, L: int):
        mask = torch.zeros((L, L))
        mask[L+1: L+1+self.btnk_len, :] = 1
        mask[:, L+1: L+1+self.btnk_len] = 1
        return (mask == 1).unsqueeze(0)  # 1, L, L

    def forward(
            self,
            audio_feat: torch.Tensor,
            video_feat: torch.Tensor):
        B, L_AUD = audio_feat.shape[:2]
        L_VID = video_feat.shape[1]

        # append CLS tokens
        # B, 1, D
        aud_cls = self.audio_tok.expand(B, -1, -1)
        vid_cls = self.video_tok.expand(B, -1, -1)
        # bottleneck tokens
        # B, btnk_len, D
        bottleneck = self.bottleneck_toks.expand(B, -1, -1)

        # B, L+1, D
        audio_feat = torch.cat([audio_feat, aud_cls], dim=1)
        video_feat = torch.cat([video_feat, vid_cls], dim=1)

        # B, (L+1)+Btnk+(L+1), D
        feats = torch.cat([audio_feat, bottleneck, video_feat], dim=1)
        feats = self.pos_embd(feats)

        mask = self._get_btnk_mask(feats.shape[1]).to(feats.device)

        out = self.attn(feats, feats, feats, mask)
        out = self.ln(out)

        aud_cls = out[:, 0]
        aud_outs = out[:, 1:L_AUD+1]
        vid_cls = out[:, L_AUD + 1 + self.btnk_len]
        vid_outs = out[:, L_AUD + 2 + self.btnk_len:]
        # aud_outs = out[:, :L_AUD]
        # vid_outs = out[:, L_AUD + self.btnk_len:]

        assert aud_outs.shape[1] == L_AUD
        assert vid_outs.shape[1] == L_VID

        aud_out = aud_cls
        vid_out = vid_cls
        # aud_out = torch.mean(aud_outs, dim=1)
        # vid_out = torch.mean(vid_outs, dim=1)

        aud_pred = self.aud_pred(aud_out)
        vid_pred = self.vid_pred(vid_out)

        return (aud_pred + vid_pred) / 2
