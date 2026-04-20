import torch.nn as nn
import torch.nn.functional as F
import torch


# class ConstOutput(nn.Module):
#     def __init__(self, d, N=1):
#         """
#         d: embedding dimension
#         N: number of dummy attention tokens
#         """
#         super().__init__()
#         self.register_buffer("score", torch.zeros(1))
#         self.register_buffer("repr", torch.zeros(d))
#         self.register_buffer("attn_w", torch.ones(N))

#     def forward(self, query_emb, feats):
#         q = query_emb[0] if type(query_emb) is list else query_emb
#         B = q.size(0)
#         device = q.device

#         return (
#             self.score.expand(B).to(device),      # [B]
#             # self.score.expand(B, -1).to(device),      # [B,1]
#             #self.repr.expand(B, -1).to(device),      # [B,d]
#             self.attn_w.expand(B, -1).to(device),    # [B,N]
#         )


class ScalarEmbed(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(1, d)
    def forward(self, x):                      # x: [B] ή [B,1]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return F.gelu(self.fc(x))              # [B,d]

class VecProj(nn.Module):
    def __init__(self, d_in, d):
        super().__init__()
        self.fc = nn.Linear(d_in, d)
        self.layer_norm = nn.LayerNorm(d)
    def forward(self, x):                      # x: [B,d_in]
        x = self.fc(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        return x            # [B,d]

def sample_token_mask(x, drop_prob):
    # x: [B, T, D]
    B, T, _ = x.shape
    keep = torch.rand(B, T, device=x.device) > drop_prob
    idx = torch.randint(0, x.shape[1], (x.shape[0],), device=x.device)
    keep[torch.arange(x.shape[0]), idx] = True
    return keep  # True = keep token


def phoc_tokenizer(phoc, num_chars = 38, num_segments = 6):
    B = phoc.size(0)

    return phoc.view(B, num_segments, num_chars)

    return [tokens[:, i, :] for i in range(num_segments)]


class AttentionScoringModule(nn.Module):
    def __init__(self, d_tkn, d_hid, feature_projections, num_heads):
        super().__init__()
        self.feature_projections = feature_projections

        self.qry_norm = nn.LayerNorm(d_tkn)
        self.tkn_norm = nn.LayerNorm(d_tkn)
        self.attn = nn.MultiheadAttention(embed_dim=d_tkn, num_heads=num_heads, dropout=0.1, batch_first=True)
        self.scoring = nn.Sequential(
            nn.Linear(d_tkn, d_hid),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(d_hid, 1),
        )

        self.num_phoc_tokens = 6
        self.pos_embed = nn.Embedding(self.num_phoc_tokens, d_tkn)

        self.a = nn.Parameter(torch.ones(()))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, query_emb, feats): ## TODO  q_emb visual and semantic emb separate tokens ???
                                         ##  edit convert_rows_to_feat_dict to enable this behaviour
        query = torch.stack(query_emb, dim=1).to(query_emb[0].device) #[B,2,d]
        query = query.mean(dim=1).unsqueeze(1) # single token

        # Build token list (excluding query)
        tokens = []
        for k, proj in self.feature_projections.items():
            if k in feats:
                if k == 'v_emb_tok': #phoc embedding
                    x = phoc_tokenizer(feats['v_emb']) #[B,6,38]
                    y = proj(x) #[B,6,d_tkn]
                    positions = torch.arange(self.num_phoc_tokens, device=y.device)
                    y = y + self.pos_embed(positions)
                    tokens.extend([y[:,i,:] for i in range(self.num_phoc_tokens)])
                    continue

                y = proj(feats[k])
                tokens.append(y) #[B,d]
                if y.dim()!=2:
                    raise ValueError("Provided feats have wrong number of dimensions")

        if len(tokens) == 0:
            raise ValueError("No tokens provided in feats.")

        T = torch.stack(tokens, dim=1).to(query_emb[0].device)  # [B,N,d]
        T = self.tkn_norm(T)
        
        if self.training:
            keep_mask = sample_token_mask(T, drop_prob=0.2) # [B, N]
        else:
            keep_mask = torch.ones(
                T.size(0), T.size(1),
                dtype=torch.bool,
                device=T.device
            )

        # Cross-attention: q attends to candidate tokens
        # attn_out: [B,1,d], attn_w: [B,1,N]
        norm_query = self.qry_norm(query)
        attn_out, attn_w = self.attn(query=norm_query, key=T, value=T, need_weights=True, key_padding_mask=~keep_mask)
        attn_out = attn_out + query # skip connection

        #single token
        attn_out = attn_out.view(attn_out.size(0), -1)
        attn_w = attn_w.view(attn_w.size(0), -1)
        # two tokens
        # attn_out = attn_out.view(attn_out.size(0), 2, -1)
        # attn_w = attn_w.view(attn_w.size(0), 2, -1)

        #single token
        out = self.a*self.scoring(attn_out).squeeze(-1)+self.b # [B]
        #two tokens
        # out = self.a*self.scoring(attn_out).mean(dim=1).squeeze(1)+self.b # [B]
        #out = out * torch.exp(-self.log_t)

        return out, attn_w
        return attn_out, attn_w


def attn_entropy(w):
    eps = 1e-8
    return -(w * torch.log(w + eps)).sum(dim=-1)  # [B]



class QueryConditionedTokenMixer(nn.Module):
    """
    Query-aware fusion:
    - project mixed tokens to common dim d
    - do cross-attention (query attends to tokens)
    - produce fused score + interpretable attn weights over tokens
    """
    def __init__(
        self,
        d = 32,
        d_scalar=16,
        d_vec=256,
        query_specs= (('q_emb', 228+384),),#(("qv_emb", 228), ("qs_emb", 384)), #(q_emb, 228+384)
        scalar_keys=("v_sim", "s_sim", "rank_prior"),
        ocr_keys=("ocr_entropy", "levenshtein", "cer"),
        vec_specs=(("v_emb", 228), ("s_emb", 384)),# ("c_emb", 228+384),
    ):
        super().__init__()
        self.d = d
        self.d_scalar = d_scalar
        self.d_vec = d_vec
        self.query_specs = list(query_specs)
        self.query_keys = list(dict(query_specs).keys())
        self.scalar_keys = list(scalar_keys)
        self.ocr_keys = list(ocr_keys)
        self.vec_specs = list(vec_specs)

        self.query_projs = nn.ModuleDict({
            f"{k} {m}": VecProj(dim_in, dim_out)
                for k, dim_in in self.query_specs
                    for m, dim_out in [('scalar',d_scalar), ('ocr',d_scalar), ('vec',d_vec)]
        })

        self.experts = nn.ModuleDict()

        # Cross-attention: query -> tokens
        if len(scalar_keys) > 0:
            scalar_embeds = nn.ModuleDict({k: ScalarEmbed(d_scalar) for k in self.scalar_keys})
            self.experts['scalar'] = AttentionScoringModule(d_scalar, d, scalar_embeds, 1)

        if len(ocr_keys) > 0:
            ocr_embeds = nn.ModuleDict({k: ScalarEmbed(d_scalar) for k in self.ocr_keys})
            self.experts['ocr'] = AttentionScoringModule(d_scalar, d, ocr_embeds, 1)

        if len(vec_specs) > 0:
            vec_projs = nn.ModuleDict({k: VecProj(dim_in, d_vec) for k, dim_in in self.vec_specs})
            self.experts['vec'] = AttentionScoringModule(d_vec, d, vec_projs, 2)

        self.n_experts = len(self.experts)

        if self.n_experts == 0:
            raise ValueError('At least one expert is required')

        D_gate = 10  # 10 reliability + evidence dim

        self.gate = nn.Sequential(
            nn.Linear(D_gate, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 2)
        )

        self.ocr_delta = nn.Sequential(
            nn.Linear(D_gate, 32),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False)
        )

        self.vec_delta = nn.Sequential(
            nn.Linear(D_gate, 32),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False)
        )

        nn.init.zeros_(self.ocr_delta[-1].weight)
        nn.init.zeros_(self.vec_delta[-1].weight)


    def forward(self, feats):
        """
        feats dict must contain:
          - q_emb: [B,Dt]  (query embedding)
          - token features for each candidate (same B)
        We assume B = number of candidates for a single query batch
        (or you can flatten (query,candidate) pairs).
        """

        attn_scr = {}
        attn_maps = {}

        for k, expert in self.experts.items():
            q_emb = [ self.query_projs[f"{qs} {k}"](feats[qs])  for qs in self.query_keys ]
            score, attn_w = expert(q_emb, feats)
            attn_scr[k] = score
            attn_maps[k] = attn_w

        s_scalar = attn_scr['scalar']
        s_ocr = attn_scr['ocr']
        s_vec = attn_scr['vec']

        evidence_feats = torch.stack([
            s_scalar,
            s_ocr,
            s_vec,
            (s_scalar - s_vec).abs(),
            (s_scalar - s_ocr).abs(),
            (s_vec - s_ocr).abs(),
        ], dim=-1)

        reliability_feats = torch.stack([
            feats['cer'],
            feats['ocr_entropy'],
            feats['levenshtein'],
            (feats['v_sim'] - feats['s_sim']).abs(),
        ], dim=-1).squeeze(1)

        gate_input = torch.cat([evidence_feats, reliability_feats], dim=-1)
        gate_input = F.layer_norm(gate_input, gate_input.shape[-1:])

        w = F.sigmoid(self.gate(gate_input))               # [B, 3]

        delta_ocr = self.ocr_delta(gate_input).squeeze(-1)
        delta_vec = self.vec_delta(gate_input).squeeze(-1)

        delta = w[:, 0] * delta_ocr + w[:, 1] * delta_vec

        s_final = s_scalar + delta

        return {
            'final_score': s_final,
            'scalar_score': s_scalar,
            'ocr_score': s_ocr,
            'vec_score': s_vec,
            'weights': w,
            'attn_maps': attn_maps
        }   
