from .ode_jump import ODEJump,ODEFunc
import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0, kdim=None):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            kdim=kdim or d_model,
            vdim=kdim or d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # (B, T, C)
        )
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,              # (B, T_q, C) — p.ex., saídas do decoder
        kv: torch.Tensor,             # (B, T_kv, C) — p.ex., saídas do encoder
        kv_key_padding_mask=None,     # (B, T_kv) True = PADDING/ignorar
        attn_mask=None                # (T_q, T_kv) ou (B*n_heads, T_q, T_kv)
    ):
        # MHA já faz: attn = softmax(QK^T/sqrt(d_k)) V
        attn_out, attn_weights = self.mha(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=kv_key_padding_mask,  # opcional
            attn_mask=attn_mask,                   # opcional (p.ex., máscara causal)
            need_weights=True
        )
        # Residual + norm (padrão Transformer)
        x = self.norm(q + self.out(attn_out))
        return x, attn_weights  # attn_weights: (B, n_heads, T_q, T_kv) a partir do PyTorch 2.6+

class TransformerGate(nn.Module):
    """
    Produz α_t,c em (0,1) a partir de [x, x_denoised, (x_denoised-x), mask].
    Saída tem shape (B,T,C). Use dropout de fonte para forçar o uso de ambos sinais.
    """
    def __init__(self, hidden_dim: int, p_src_dropout: float = 0.1):
        super().__init__()
        self.p_src_dropout = p_src_dropout
        # features por canal: [x, x_denoised, delta, mask] -> 4*C
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim*6),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*6, hidden_dim)   # logits por canal
        )

    def forward(self, h_gru, h_trf, train_mode: bool = True):
        # concat por canal
        delta = h_gru - h_trf
        h = torch.cat([h_gru, h_trf, delta], dim=-1)  # (B,T,4*C)
        logits = self.mlp(h)                            # (B,T,C)
        alpha  = torch.sigmoid(logits)                      # (0,1)

        # "source dropout": às vezes força só raw ou só denoised (melhora generalização)
        if train_mode and self.p_src_dropout > 0.0:
            # mesma máscara para todo o batch/time-step (poderia ser por amostra)
            if torch.rand(1, device=h_gru.device) < (self.p_src_dropout / 2):
                alpha = alpha.detach()*0  # 100% raw
            elif torch.rand(1, device=h_gru.device) < (self.p_src_dropout / 2):
                alpha = alpha.detach()*0 + 1  # 100% denoised

        # fusão final
        x_fused = alpha * h_gru + (1.0 - alpha) * h_trf
        return x_fused, alpha

class AttnMemory(nn.Module):
    """
    Atenção causal sobre uma memória local de estados (ou embeddings de x).
    Q = h (B,H) -> (B,1,H), K,V = mem (B,K,H). Retorna contexto c (B,H).
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4, attn_dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=attn_dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)  # estabiliza Q antes da MHA

    def forward(self, h: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # h: (B,H), mem: (B,K,H) (K pode ser 0)
        if mem is None or mem.size(1) == 0:
            return torch.zeros_like(h)
        q = self.norm(h).unsqueeze(1)      # (B,1,H)
        k = mem                            # (B,K,H) passado apenas (causal por construção)
        v = mem
        ctx, _ = self.mha(q, k, v, need_weights=False)  # (B,1,H)
        return ctx.squeeze(1)              # (B,H)



class JumpODEEncoder(nn.Module):
    """
    Self‑Attentive Jump‑ODE simplificado:
    - GRUCell executa o *jump* g_ψ na chegada de cada evento (x_i, t_i)
    - ODEFunc integra h(t) entre eventos.
    - Self‑attention usa máscara para faltantes (opcional).
    """
    def __init__(self, in_dim, hidden_dim, mem_len: int = 32,
                 n_heads: int = 4, attn_dropout: float = 0.1,
                 attn_heads=None, num_layers=2, dropout=0.2, ff_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.mem_len = mem_len
        self.gru = nn.GRUCell(in_dim, in_dim)
        self.odefunc = ODEFunc(in_dim)
        if attn_heads is None:
            attn_heads = n_heads
        enc_layer = nn.TransformerEncoderLayer(
            nhead=attn_heads,
            d_model=in_dim,
            norm_first=True,  # normalização antes da atenção
            dropout=dropout,  # dropout opcional
            dim_feedforward= self.hidden_dim * 4 or ff_dim,  # feedforward dimension
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(in_dim)
        )    
        self.t_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, 2*in_dim)  # gamma, beta
        )  
        self.time_encoding = TimeHybridEncoding(in_dim)    
        self.transformer_gate = TransformerGate(in_dim)
        self.norm_gru = nn.LayerNorm(in_dim)
        self.norm_H = nn.LayerNorm(in_dim)
        self.norm_ode = nn.LayerNorm(in_dim)

        if in_dim != hidden_dim:
            self.encoder=nn.Sequential(
                nn.Linear(in_dim,hidden_dim*4),
                nn.GELU(),
                nn.Linear(hidden_dim*4,hidden_dim)
            )

    @staticmethod
    def _causal_mask(L, device):
        # Máscara triangular superior (impede olhar para o futuro)
        return torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)    

    @staticmethod
    def _rk4_step(h, dt, f, c):
        # RK4 com contexto c fixo no intervalo
        k1 = f(h, c)
        k2 = f(h + 0.5*dt*k1, c)
        k3 = f(h + 0.5*dt*k2, c)
        k4 = f(h + dt*k3, c)
        return h + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def forward(self, x, ts, only_gru=False):
        B, T, C = x.shape
        #eps = 1e-6
        h = torch.zeros(B, self.in_dim, device=x.device)
        #mem = torch.zeros(B, 0, self.hidden_dim, device=x.device)  # memória causal curta
        states = []

        for i in range(T):
            if i > 0 and not only_gru:
                dt = (ts[:, i] - ts[:, i-1]).float().unsqueeze(-1)  # (B,1)
                # controle: mantenha simples. Use input anterior constante no intervalo
                u = (x[:, i],x[:, i-1],(x[:, i]-x[:, i-1]) / dt)  # (B, C) — assuma C == hidden_dim quando este bloco recebe embedding
                # RK4 em h, condicionado por u
                k1 = self.odefunc(h, *u)
                k2 = self.odefunc(h + 0.5*dt*k1, *u)
                k3 = self.odefunc(h + 0.5*dt*k2, *u)
                k4 = self.odefunc(h + dt*k3, *u)
                h  = h + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                h = self.norm_ode(h)

            # JUMP no evento i (como no esquema original)
            h = self.gru(x[:, i], h)
            h = self.norm_gru(h)
            states.append(h)

        h = torch.stack(states, dim=1)  # (B, T, hidden_dim)
        #h,_ = self.cross_attn(h+tme,x+tme)
        if not only_gru:
            gb = self.t_film(h)                       # (B,1,2D)
            gamma, beta = gb.chunk(2, dim=-1)          # (B,1,D), (B,1,D)
            tme = self.time_encoding(ts)
            tme = (1.0 + gamma) * tme + beta         # FiLM no tempo contínuo
            H = self.transformer(h+tme)
            #H,_ = self.cross_attn(h_gru,h_trf)
        else:
            H=h
        return H if self.in_dim == self.hidden_dim else self.encoder(H)


class TimeHybridEncoding(nn.Module):
    """
    Odd dims  -> ramp(t) = a*t + b   (aprendido)
    Even dims -> sinusoidal [sin, cos] com frequências log-espalhadas
    Usa timestamps normalizados globalmente: (ts - t0)/TS_SPAN
    """
    def __init__(self, d_model: int,
                 min_period: float = 4.0,
                 max_period: float = 10_000.0,
                 ramp_gain: float = 1.0):
        super().__init__()
        self.d_model = d_model

        # Máscaras pares/ímpares
        even_idx = torch.arange(d_model) % 2 == 0
        odd_idx  = ~even_idx
        self.register_buffer("even_mask", even_idx, persistent=False)
        self.register_buffer("odd_mask",  odd_idx,  persistent=False)

        self.n_even = int(even_idx.sum().item())
        self.n_odd  = int(odd_idx.sum().item())

        # --- Ramp nos ímpares ---
        self.a = nn.Parameter(torch.zeros(self.n_odd, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(self.n_odd, dtype=torch.float32))
        if ramp_gain != 1.0:
            with torch.no_grad():
                self.a.mul_(ramp_gain)

        # --- Sin/cos nos pares ---
        self.n_freq = max(self.n_even // 2, 0)
        if self.n_freq > 0:
            freqs = torch.exp(
                -torch.linspace(0, math.log(max_period/min_period), self.n_freq, dtype=torch.float32)
            ) * (2.0 * math.pi / min_period)
            self.register_buffer("freqs", freqs, persistent=False)
        else:
            self.register_buffer("freqs", torch.empty(0, dtype=torch.float32), persistent=False)

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        """
        ts: (B,T) — timestamps já normalizados: (ts - t0)/TS_SPAN
        retorna: (B,T,D)
        """
        B, T = ts.shape
        device, dtype = ts.device, ts.dtype
        out = torch.zeros(B, T, self.d_model, device=device, dtype=ts.dtype)

        # ----- RAMP nas dims ímpares -----
        if self.n_odd > 0:
            ramp = ts.unsqueeze(-1) * self.a.view(1,1,-1) + self.b.view(1,1,-1)  # (B,T,n_odd)
            odd_positions = self.odd_mask.nonzero(as_tuple=False).squeeze(-1)
            out.index_copy_(dim=2, index=odd_positions, source=ramp)

        # ----- SIN/COS nas dims pares -----
        if self.n_even > 0 and self.n_freq > 0:
            even_positions = self.even_mask.nonzero(as_tuple=False).squeeze(-1)
            phase = ts.unsqueeze(-1) * self.freqs.view(1,1,-1)   # (B,T,n_freq)
            S = torch.sin(phase)
            C = torch.cos(phase)
            sc = torch.stack([S, C], dim=-1).reshape(B, T, -1)  # (B,T,2*n_freq)
            n_fill = min(sc.size(-1), self.n_even)
            out[:, :, even_positions[:n_fill]] = sc[:, :, :n_fill]

        return out
    
class ODEJumpEncoder(ODEJump):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        denoised: bool = False,
        lam: list[float,float] = [0.9, 0.1],
        n_heads: int = 4,
        n_layers: int = 4,
        cost_columns: list = None                  
        ):
        super().__init__(in_channels, hidden_dim, static_dim, denoised, lam, cost_columns)  # <- chama ODEJump.__init__
        # (daqui pra baixo você pode sobrescrever/estender o que quiser)
        
        self.encoder_ode = JumpODEEncoder(hidden_dim, hidden_dim, n_heads=n_heads,num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None,
        x_denoised: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        # Embedding de entrada
        if not already_latent:
            if x_denoised is not None:
                # aplique gate (treino=True quando model.training)
                x_fused, _ = self.denoise_gate(x, x_denoised, mask, train_mode=self.training)
                h_in = torch.cat([x_fused, mask], dim=-1)
            else:
                h_in = torch.cat([x, mask], dim=-1)
            h = self.encoder(h_in)  # (B,T,hidden_dim)
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        #tm_e = self.time_encoding(timestamps.to(h.dtype)).to(h.dtype)  # tempo contínuo
        #h = h + tm_e
        h = self.encoder_ode(h, timestamps)
        state = h
        x_hat = self.decoder(state) if return_x_hat else None
        return x, state, x_hat
                
    def train_cognite(self,*args,**kwargs):
        # exemplo para ODEJumpEncoder: ajuste nomes conforme sua classe
        transformer_params = []
        base_params = []
        for n,p in self.named_parameters():
            if "transformer" in n or "encoder_ode" in n:  # bloco de atenção
                transformer_params.append(p)
            else:
                base_params.append(p)

        optimizer = torch.optim.AdamW([
            {"params": base_params, "lr": 3e-4, "weight_decay": 1e-4},
            {"params": transformer_params, "lr": 1.5e-4, "weight_decay": 1e-4},
        ], betas=(0.9, 0.98))
        kwargs['optimizer']=optimizer
        
        for res in super().train_cognite(*args,**kwargs):
            yield res
