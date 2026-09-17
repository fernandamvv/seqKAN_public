from .ode_jump_encoder import ODEJumpEncoder,JumpODEEncoder
from .ode_jump import TS_SPAN
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import RobustScaler

max_drop = 0.3

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Beta schedule baseado em Nichol & Dhariwal (2021):
    β_t = 1 − ᾱ_t / ᾱ_{t−1}, onde
    ᾱ_t = cos²( (t/T + s) / (1 + s) * π/2 )
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    # ᾱ desde t=0 até t=T
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # β_t = 1 − ᾱ_t / ᾱ_{t−1}
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5,max=0.999)  # evita valores muito altos

class DiffTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (estilo Vaswani et al.) + projeção linear.
    """
    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        half_dim = model_dim // 2
        # Registrar as frequências como buffer (não são parâmetros treináveis)
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) 
            * math.log(10000.0) / half_dim
        )
        self.register_buffer('freqs', freqs)  # shape: (half_dim,)
        # Projeção final: mantém dimensão
        self.lin = nn.Linear(model_dim, model_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor de shape (batch,) ou (batch, 1) com timesteps inteiros.
        Returns:
            emb: (batch, model_dim) embedding do timestep.
        """
        # garante shape (batch,)
        t = t.view(-1)
        # multiplica t pelas frequências: result -> (batch, half_dim)
        args = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        # concatena seno e cosseno -> (batch, model_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # projeta de volta ao espaço de dimensão model_dim
        return self.lin(emb)
    
class TSDiffusion(ODEJumpEncoder):
    """
    Modelo de difusão para séries temporais contínuas, baseado em
    "Diffusion Models for Implicit Imputation of Time Series Data"
    (https://arxiv.org/abs/2205.14217) e
    "Score-Based Generative Modeling in Latent Space" (https://arxiv.org/abs/2206.00364).
    Usa Transformer com máscara causal para codificar a série temporal.
    """
    optimizers: dict[torch.optim.Optimizer] = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'radam': torch.optim.RAdam
    }
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        status_dim: int = 0,
        lam: list[float,float,float,float,float,float] = [0.9, 0.0, 0.0, 0.1, 0.0, 0.0],
        n_heads: int = 4,
        n_layers: int = 4,
        num_steps: int = 1000,
        cost_columns: list = None,
        n_heads_g: int = 4,
        n_layers_g: int = 4,
        log_likelihood: bool = True,
        sigma_temp: float = 0.7,
        x_min: float = None,
        x_max: float = None
        ):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            static_dim=static_dim,
            lam=lam,
            n_heads=n_heads,
            n_layers=n_layers,
            cost_columns=cost_columns
        )
        self.x_min = x_min
        self.x_max = x_max
        self.log_likelihood = log_likelihood
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.status_dim = status_dim
        self.sigma_temp = float(sigma_temp)
        self.t_embed = DiffTimeEmbedding(hidden_dim)
        if self.lam[2]>0:
            self.noise_head = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim*4),
                nn.GELU(),
                nn.Linear(hidden_dim*4,hidden_dim),
                )
            self.denoise_x = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim*4),
                nn.GELU(),
                nn.Linear(hidden_dim*4,hidden_dim)
            )
        if self.lam[0]>0:
            self.encoder_ode_x = JumpODEEncoder(hidden_dim, hidden_dim, n_heads=n_heads_g,
                                                num_layers=n_layers_g)        # (a) λ(t)  — intensity do ponto de observação
            if self.log_likelihood:
                self.lambda_head = nn.Sequential(
                    nn.Linear(hidden_dim*2, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1)       # escalar
                )
        if status_dim > 0:
            self.tmax_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )
            self.encoder_ode_tmax = JumpODEEncoder(hidden_dim, hidden_dim, n_heads=n_heads_g,num_layers=n_layers_g)
            if self.log_likelihood:
                self.lambda_tmax_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 1)       # escalar
                )        
        if self.lam[4] > 0:
            self.vae_latent = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2)
            )
            self.vae_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, in_channels)
            )
            # variância observacional para vae_x
            self.vae_sigma_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, in_channels)
            )
        if self.lam[5] > 0 and status_dim > 0:
            self.vae_tmax_latent = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2)
            )
            self.vae_tmax_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )
            # variância observacional para vae_tmax
            self.vae_tmax_sigma_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, status_dim)
            )
        # Schedule de difusão
        betas = cosine_beta_schedule(num_steps)
        alphas = 1 - betas
        self.register_buffer('beta', betas)
        self.register_buffer('alpha', alphas)
        self.register_buffer('alpha_bar', torch.cumprod(alphas, dim=0))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask: torch.Tensor = None,
        mask_ts: torch.Tensor = None,
        test: bool=True,
        only_gru: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        noise = None
        noise_hat = None
        hg = None
        ht = None
        tmax_hat = None
        vae_x = None
        vae_mu = None
        vae_logvar = None
        vae_logvar_obs = None
        vae_tmax = None
        vae_tmax_mu = None
        vae_tmax_logvar = None
        vae_tmax_logvar_obs = None
        t = t if t is not None else torch.randint(0, self.num_steps, (x.size(0),), device=x.device)
        if mask_ts is None:
            mask_ts = mask.any(dim=2, keepdim=True).float() if mask is not None else torch.ones_like(x[..., :1])
        # Embedding de entrada
        if not already_latent:
            h = self.encoder(torch.cat([x, mask], dim=-1))
            if not test and self.lam[1]>0:
                noise = torch.randn_like(h) * mask_ts
                ab = self.alpha_bar[t].view(-1, 1, 1)
                h = torch.sqrt(ab) * h + torch.sqrt(1 - ab) * noise
            else:
                t = torch.zeros((x.size(0),), device=x.device, dtype=torch.long)
                noise = None
        else:
            h = x
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        if not only_gru:
            te = self.t_embed(t).unsqueeze(1)          # (b,1,model_dim)
            h = h + te * mask_ts    
        if self.lam[2]>0:
            hg = self.encoder_ode(h, timestamps, only_gru)
            noise_hat = self.noise_head(hg)
            hd = self.denoise_x(noise_hat)
            h = h + hd
        else:
            hg = None
            noise_hat = None
        if self.lam[0]>0:
            h = self.encoder_ode_x(h, timestamps, only_gru)
        if self.lam[2]>0:
            ht = self.encoder_ode_tmax(h, timestamps, only_gru)
            tmax_hat = self.tmax_head(ht)
        else:
            ht = None
            tmax_hat = None

        if self.lam[4] > 0:
            mu_logvar = self.vae_latent(h)
            vae_mu, vae_logvar = torch.chunk(mu_logvar, 2, dim=-1)
            std = torch.exp(0.5 * vae_logvar)
            eps = torch.randn_like(std)
            z_vae = vae_mu + eps * std
            vae_x = self.vae_decoder(z_vae)
            vae_logvar_obs = self.vae_sigma_head(z_vae).clamp(min=-5.0, max=5.0)
        if self.lam[5] > 0 and ht is not None:
            mu_logvar_t = self.vae_tmax_latent(ht)
            vae_tmax_mu, vae_tmax_logvar = torch.chunk(mu_logvar_t, 2, dim=-1)
            std_t = torch.exp(0.5 * vae_tmax_logvar)
            eps_t = torch.randn_like(std_t)
            z_tmax = vae_tmax_mu + eps_t * std_t
            vae_tmax = self.vae_tmax_decoder(z_tmax)
            vae_tmax_logvar_obs = self.vae_tmax_sigma_head(z_tmax).clamp(min=-5.0, max=5.0)

        x_hat = self.decoder(h) if return_x_hat and self.lam[0] > 0 else None

        return (
            h,
            hg,
            ht,
            x_hat,
            tmax_hat,
            noise,
            noise_hat,
            vae_x,
            vae_mu,
            vae_logvar,
            vae_tmax,
            vae_tmax_mu,
            vae_tmax_logvar,
            vae_logvar_obs,
            vae_tmax_logvar_obs,
        )

    def impute(
            self, 
            **kwargs
            ):
        state=kwargs['state'] 
        device=kwargs['device']
        timestamps=kwargs['timestamps']
        static_feats=kwargs['static_feats']
        mask=kwargs['mask']
        x0=kwargs['x0']
        return_x_hat=kwargs.get('return_x_hat',True)
        t=torch.full((state.size(0),), 0, device=device, dtype=torch.long)
        mask_ts = mask.any(dim=2, keepdim=True).float()
        (
            z,
            _,
            _,
            x_hat_step,
            _,
            _,
            _,
            vae_x,
            *_,
        ) = self.forward(
            state,
            t=t,
            timestamps=timestamps,
            static_feats=static_feats,
            already_latent=True,
            return_x_hat=True,
            mask_ts=mask_ts,
        )
        # Se não há cabeça de reconstrução L1 (lam[0]==0) mas há VAE (lam[4]>0),
        # use a saída do VAE como estimativa.
        if x_hat_step is None and self.lam[4] > 0 and vae_x is not None:
            x_hat_step = vae_x

        x_clamped  = torch.where(mask.bool(), x0, x_hat_step)
        if return_x_hat:
            return x_clamped
        z = self.encoder(torch.cat([x_clamped, torch.ones_like(mask)], dim=-1))
        return z

    def denoise(self, state, timestamps, static_feats, device, steps,
                x0: torch.Tensor | None = None, mask: torch.Tensor | None = None,
                enforce_data_consistency: bool = True):
        
        t=torch.full((state.size(0),), 0, device=device, dtype=torch.long)
        mask_ts = mask.any(dim=2, keepdim=True).float() 
        (
            z,
            _,
            _,
            x_hat_step,
            _,
            _,
            _,
            vae_x,
            *_,
        ) = self.forward(
            state,
            t=t,
            timestamps=timestamps,
            static_feats=static_feats,
            already_latent=True,
            return_x_hat=True,
            mask_ts=mask_ts,
        )
        # Se não há cabeça L1 mas há VAE, use vae_x como estimativa inicial
        if x_hat_step is None and self.lam[4] > 0 and vae_x is not None:
            x_hat_step = vae_x

        x_clamped  = torch.where(mask.bool(), x0, x_hat_step)
        z = self.encoder(torch.cat([x_clamped, torch.ones_like(x_clamped)], dim=-1))
        for i in reversed(range(steps)):
            a, ab = self.alpha[i], self.alpha_bar[i]
            t = torch.full((z.size(0),), i, device=device, dtype=torch.long)
            # pred noise em latente
            _, _, _, _, _, _, eps_hat, *_ = self.forward(
                z, t=t, timestamps=timestamps, static_feats=static_feats,
                already_latent=True, return_x_hat=False, mask_ts=mask_ts, test=False
            )

            # passo de reverse (DDPM em latente)
            z = (1/torch.sqrt(a)) * (z - ((1-a)/torch.sqrt(1-ab)) * eps_hat)
            if i > 0:
                z = z + torch.sqrt(self.beta[i]) * torch.randn_like(z)

            # --- DATA CONSISTENCY opcional ---
            if enforce_data_consistency and (x0 is not None) and (mask is not None):
                with torch.no_grad():
                    x_hat_step = self.decoder(torch.cat([z,torch.zeros_like(z)], dim=-1))  # (B,T,C)
                    x_clamped  = torch.where(mask.bool(), x0, x_hat_step)
                    # re-encode para latente mantendo a máscara
                    z = self.encoder(torch.cat([x_clamped, mask], dim=-1))

        return x_clamped,z
    
    def _compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        tmax: torch.Tensor,
        state: torch.Tensor,
        state_tmax: torch.Tensor,
        ts_batch: torch.Tensor,
        mask: torch.Tensor,
        mask_train: torch.Tensor,
        cc: torch.Tensor,
        state_pred: torch.Tensor,
        status_pred_window: np.float32,
        noise,
        noise_hat,
        lambda1,
        weight_tmax: float,
        vae_mu: torch.Tensor | None = None,
        vae_logvar: torch.Tensor | None = None,
        vae_x: torch.Tensor | None = None,
        vae_tmax: torch.Tensor | None = None,
        vae_tmax_mu: torch.Tensor | None = None,
        vae_tmax_logvar: torch.Tensor | None = None,
        vae_logvar_obs: torch.Tensor | None = None,
        vae_tmax_logvar_obs: torch.Tensor | None = None,
        kl_scale: float = 1.0
        
    ):
        if x_hat is not None:
            #L1
            mask_err = mask * (1 - mask_train) # erro ao longo dos C canais observados 
            sse = (((x_hat - x)**2) * mask_err * cc).sum(dim=-1) # (B,T) 
            #nobs =x_hat.numel() # -½ λ ||x-μ||^2 + ½ log λ
            nobs = (mask_err*cc).sum(dim=-1).clamp(min=1e-8) # -½ λ ||x-μ||^2 + ½ log λ
            if self.log_likelihood:
                lam_t = F.softplus(
                    self.lambda_head(
                        torch.cat(
                            [state,noise_hat if noise_hat is not None else torch.zeros_like(state)],dim=-1
                            ))).clamp(min=1/(2*math.pi),max=2*math.pi) # (B,T,1) 
                lam2 = lam_t.squeeze(-1) # (B,T)  
                log_px = -0.5 * lam2 * sse + 0.5 * nobs * torch.log(lam2) - 0.5 * nobs * math.log(2*math.pi) # (B,T) 
                # Se quiser normalizar para não depender de C/T, use média por observação: # loss por (B,T) normalizada por nobs: 
                neg_log_px = -(log_px) # (B,T) 
                L1 = neg_log_px.sum() # escalar
                L1_div = nobs.sum().clamp(min=1.0)
            else:
                L1 = sse.sum()  # escalar
                L1_div = nobs.sum().clamp(min=1.0)
        else:
            L1 = torch.tensor(0.0, device=state.device)
            L1_div = torch.tensor(1.0, device=state.device)
        

        if self.lam[2]>0:
            #L3
            mask_tmax = mask.new_zeros(mask.size(0), mask.size(1), 1)  # usar máscara completa para VAE
            mask_tmax[:,-1,:] = 1.0
            offset_state_pred = (state_pred - ts_batch.unsqueeze(-1)).clamp(min=0,max=status_pred_window) / status_pred_window
            offset_tmax = tmax
            changing_state = ((offset_state_pred<1) & (offset_state_pred>0)).float()
            err = mask_tmax * (offset_tmax - offset_state_pred * changing_state)   # (B,T,S)
            err_no_change = err * (1-changing_state)
            err_change = err * changing_state * weight_tmax
            if self.log_likelihood:
                lam_t_tmax = F.softplus(self.lambda_tmax_head(state_tmax)).clamp(min=1/(2*math.pi), max=2*math.pi)  # (B,T,1)
                lam2_tmax  = lam_t_tmax.squeeze(-1)                                            # (B,T)
                lam2_tmax_clamped = lam2_tmax.clamp(min=0.1)  # novo tensor, sem in-place
                sse_tmax_no_change = (err_no_change**2).sum(dim=-1)              # (B,T)  soma sobre S
                sse_tmax_change = (err_change**2).sum(dim=-1)
                S_bt = torch.full_like(lam2_tmax, float(err.size(-1)))   # (B,T)
                log_ptmax = (
                    - 0.5 * lam2_tmax * sse_tmax_no_change
                    - 0.5 * lam2_tmax_clamped * sse_tmax_change
                    + 0.5 * S_bt * torch.log(lam2_tmax_clamped)
                    - 0.5 * S_bt * math.log(2 * math.pi)
                )                                       # (B,T)

                # Média por timestep (B,T). Se preferir, some e divida por B*T explicitamente.
                L3 = -(log_ptmax.sum())
                L3_div = float(err_change.size(0)*err_change.size(2))
            else:
                sse_tmax_change = ((err_change**2)+(err_no_change**2)).sum(dim=-1)
                L3 = sse_tmax_change.sum()
                L3_div = float(err_change.size(0)*err_change.size(2))
                 
        else:
            L3 = torch.tensor(0.0, device=state.device)
            L3_div = torch.tensor(1.0, device=state.device)   

        if self.lam[3]>0:
            # ----- L4 (máscara) -----
            # máscara binária: 1 se ao menos um canal está presente no timestep
                    # (B, T, 1)
            m_t = mask_train.any(dim=2, keepdim=True).float()              # (B,T,1)
            L4 = torch.nn.functional.binary_cross_entropy_with_logits(self.miss_head(state), m_t, reduction='sum')
            L4_div = float(m_t.numel())
        else:
            L4 = torch.tensor(0.0, device=state.device)
            L4_div = 1.0

        if self.lam[4] > 0 and vae_x is not None and vae_mu is not None and vae_logvar is not None:
            mask_vae = mask
            if vae_logvar_obs is not None:
                logvar_obs = (vae_logvar_obs + math.log(self.sigma_temp)).clamp(min=-5.0, max=5.0)
                nll_obs = 0.5 * (logvar_obs + ((x - vae_x) ** 2) / torch.exp(logvar_obs) + math.log(2 * math.pi))
                vae_recon = (nll_obs * mask_vae * cc).sum()
            else:
                vae_recon = ((x - vae_x).pow(2) * mask_vae * cc).sum()
            vae_recon_div = (mask_vae * cc).sum().clamp(min=1.0)
            vae_kl = -0.5 * torch.sum(1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp())
            vae_kl_div = vae_logvar.numel()
            L5 = vae_recon / vae_recon_div + kl_scale * (vae_kl / vae_kl_div)
            L5_div = torch.tensor(1.0, device=x.device)
        else:
            L5 = torch.tensor(0.0, device=state.device)
            L5_div = torch.tensor(1.0, device=state.device)

        if self.lam[5] > 0 and vae_tmax is not None and vae_tmax_mu is not None and vae_tmax_logvar is not None:
            mask_vae = mask.new_zeros(mask.size(0), mask.size(1), 1)  # usar máscara completa para VAE
            mask_vae[:,-1,:] = 1.0            
            offset_state_pred = (state_pred - ts_batch.unsqueeze(-1)).clamp(min=0,max=status_pred_window) / status_pred_window
            offset_tmax = (vae_tmax)
            changing_state = ((offset_state_pred<1) & (offset_state_pred>0)).float()
            err = (offset_tmax - offset_state_pred * changing_state)   # (B,T,S)
            err_no_change = err * (1-changing_state)
            err_change = err * changing_state
            if vae_tmax_logvar_obs is not None:
                logvar_obs_t = (vae_tmax_logvar_obs + math.log(self.sigma_temp)).clamp(min=-5.0, max=5.0)
                base_nll_t = 0.5 * (logvar_obs_t + (err ** 2) / torch.exp(logvar_obs_t) + math.log(2 * math.pi))
                base_nll_t = base_nll_t.clamp_min(0.0)
                weight_t = 1.0 + (100.0 - 1.0) * changing_state
                vae_recon_t = (base_nll_t * weight_t * mask_vae).sum()
            else:
                vae_recon_t = ((err_change ** 2 * 1000 + err_no_change ** 2)*mask_vae).sum()
            vae_kl_t = -0.5 * torch.sum((1 + vae_tmax_logvar - vae_tmax_mu.pow(2) - vae_tmax_logvar.exp()) * mask_vae) 
            L6_div = mask_vae.sum().clamp(min=1.0)
            L6 = (vae_recon_t / L6_div) + kl_scale * (vae_kl_t / L6_div)
        else:
            L6 = torch.tensor(0.0, device=state.device)
            L6_div = torch.tensor(1.0, device=state.device)

        if self.lam[1] == 0:
            L2 = torch.tensor(0.0, device=state.device)
            L2_div = torch.tensor(1.0, device=state.device)
            loss = self.lam[0]*L1/L1_div + self.lam[2]*L3/L3_div + self.lam[3]*L4/L4_div + self.lam[4]*L5 + self.lam[5]*L6/L6_div
        else:
            # noise_hat may be None for models that do not predict noise
            if noise_hat is None:
                L2 = torch.tensor(0.0, device=state.device)
                L2_div = torch.tensor(1.0, device=state.device)
                loss = self.lam[0]*L1/L1_div + self.lam[2]*L3/L3_div + self.lam[3]*L4/L4_div + self.lam[4]*L5 + self.lam[5]*L6/L6_div
            else:
                L2 = F.mse_loss(noise, noise_hat, reduction='sum')
                L2_div = (torch.ones_like(state) * m_t).sum()

                if L2 / L2_div > 0.1:
                    loss = + self.lam[1] * L2 / L2_div + (self.lam[0]*L1/L1_div + self.lam[2]*L3/L3_div + self.lam[3]*L4/L4_div + self.lam[4]*L5 + self.lam[5]*L6/L6_div) * 1e-3
                else:
                    loss = self.lam[0]*L1/L1_div + self.lam[1] * L2 / L2_div + self.lam[2]*L3/L3_div + self.lam[3]*L4/L4_div + self.lam[4]*L5 + self.lam[5]*L6/L6_div

        if lambda1 > 0:
            l1 = sum(
                p.abs().sum() for n, p in self.named_parameters()
                if p.requires_grad and "bias" not in n
                )
            loss = loss + lambda1 * l1

        return (
            loss,
            (float(L1.item()), float(L1_div.item())),
            (float(L2.item()),float(L2_div.item())),
            (float(L3.item()), float(L3_div)),
            (float(L4.item()), float(L4_div)),
            (float(L5.item()), float(L5_div.item())),
            (float(L6.item()), float(L6_div.item()))
            )
    
    def _make_random_mask(self,x,ts_batch,m,device,max_drop=None):
        if max_drop is None:
            max_drop = globals().get("max_drop", 0.3)
        t_mask = torch.randint(0, self.num_steps, (x.size(0),), device=device)
        t_mask_ts = torch.randint(0, self.num_steps, (x.size(0),), device=device)
        # 2) probabilidade de *extra-missing* cresce com t
        p_drop_t = (t_mask.float() / (self.num_steps - 1)) * max_drop   # (B,)
        p_drop_t = p_drop_t.view(-1, 1, 1)                         # broadcast
        p_drop_ts = (t_mask_ts.float() / (self.num_steps - 1)) * max_drop   # (B,)
        p_drop_ts = p_drop_ts.view(-1, 1)                         # broadcast

        rand_mask = (torch.rand_like(m) > p_drop_t).float()
        rand_mask_ts = (torch.rand_like(ts_batch) > p_drop_ts).unsqueeze(-1).float()
        rand_mask_ts[:,-1,0]=0
        return m * rand_mask * rand_mask_ts

    def _kl_scale(self, epoch: int, kl_start: float, kl_end: float, kl_warmup_epochs: int) -> float:
        """Exponencial suave do peso de KL ao longo das épocas."""
        if kl_warmup_epochs <= 0:
            return float(kl_end)
        progress = min(1.0, max(epoch - 1, 0) / kl_warmup_epochs)
        if kl_start <= 0:
            return float(kl_end)
        scale = kl_start * ((kl_end / kl_start) ** progress)
        return float(scale)
    
    def train_cognite(self,
        df: pd.DataFrame,
        feature_cols: list,
        static_features_cols: list,
        timestamp_col: str,
        states_col: str | list,
        predict_state_cols: str | list = None,
        status_pred_window: int = 600,
        batch_size: int = 32,
        lr: float = 3e-4,
        window_size: int = None,
        window_step: int = 1,
        epochs: int = 10,
        validate: bool = True,
        early_stopping: bool = True,
        patience: int = 15,
        device: torch.device = None,
        label_at: str = "end",
        fixed_test_idx: np.ndarray | None = None,
        seed_split: int = 42,
        lr_t: float = 1.5e-4,
        only_gru: bool = False,
        reconstruction_test: bool = True,
        optimizer_name: str = 'adamw',
        optimizer_params: dict = {'weight_decay': 1e-4},
        warmup_steps: int = 0,
        min_lr_factor: float = 0.1,
        lambda1: float = 0.0,
        rebuild: bool = True,
        kl_start: float = 0.001,
        kl_end: float = 1.0,
        kl_warmup_epochs: int = 50,
        train_fraction: float = 0.6,
        weight_tmax: float = 1.0,
        max_drop: float | None = None
    ):
        delta_pred_window = np.float32(status_pred_window / TS_SPAN)
        # exemplo para ODEJumpEncoder: ajuste nomes conforme sua classe
        transformer_params = []
        base_params = []
        for n,p in self.named_parameters():
            if "transformer" in n or "encoder_ode" in n:  # bloco de atenção
                transformer_params.append(p)
            else:
                base_params.append(p)
        base_params = {'params': base_params}
        transformer_params = {'params': transformer_params}
        base_params.update(optimizer_params)
        transformer_params.update(optimizer_params)
        base_params['lr'] = lr
        transformer_params['lr'] = lr_t

        optimizer = self.optimizers[optimizer_name]([
            base_params,
            transformer_params,
        ], betas=(0.9, 0.98))



        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        grid = None
        grid_eps = None
        if hasattr(self, "encoder_ode_x") and hasattr(self.encoder_ode_x, "kan_params"):
            kp = self.encoder_ode_x.kan_params or {}
            cell = kp.get("cell", kp.get("hidden", {}))
            grid = cell.get("grid", kp.get("grid", None))
            grid_eps = cell.get("grid_eps", kp.get("grid_eps", None))
        run_cfg = {
            "model": self.__class__.__name__,
            "grid": grid,
            "grid_eps": grid_eps,
            "in_channels": getattr(self, "in_channels", None),
            "hidden_dim": getattr(self, "hidden_dim", None),
            "x_min": getattr(self, "x_min", None),
            "x_max": getattr(self, "x_max", None),
        }
        print("RUN CONFIG:", run_cfg)
        df_sorted = df if timestamp_col == "index" else df.sort_values(timestamp_col).reset_index(drop=True)

        # Dataset (sem y) e rótulos de grupo por janela (para split/oversampling/relato)
        ds = self._make_dataset(
            df_sorted, 
            timestamp_col, 
            window_size, 
            feature_cols, 
            static_features_cols, 
            window_step,
            predict_state_cols=predict_state_cols
            )
        y_win, _starts = self._states_from_df_windows(df_sorted, states_col, window_size, window_step, label_at)
        all_groups = np.unique(y_win)

        N = ds.tensors[0].shape[0]
        if N != len(y_win):
            raise ValueError(f"Inconsistência: dataset={N} vs rótulos={len(y_win)}.")
        val_frac = (1.0 - train_fraction) / 2
        train_frac, val_frac, test_frac = train_fraction, val_frac, val_frac
        # --- Split por grupo (proporcional): 60/20/20 ou 80/0/20
        if fixed_test_idx is not None:
            test_idx = np.asarray(fixed_test_idx, dtype=int)
            remain_mask = np.ones(N, dtype=bool); remain_mask[test_idx] = False
            tr_idx_rel, va_idx_rel, _ = self._split_by_group_proportions(
                y_win[remain_mask], validate=validate,
                train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed_split
            )
            base = np.where(remain_mask)[0]
            train_idx = base[tr_idx_rel]
            val_idx   = base[va_idx_rel]
        else:
            train_idx, val_idx, test_idx = self._split_by_group_proportions(
                y_win, validate=validate, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, seed=seed_split
            )
        x_all_scaled = self.scale_tensor(ds.tensors[0][train_idx], ds.tensors[0])
        ds.tensors = (torch.tensor(x_all_scaled, dtype=torch.float32),) + ds.tensors[1:]
        # --- Clamp counts (feature-specific) after clamp on model input
        target_feat = "PH (CBM) 1st Stage ExpCompr Poly Eff"
        if target_feat in feature_cols:
            feat_idx = feature_cols.index(target_feat)
            x_min = getattr(self, "x_min", None)
            x_max = getattr(self, "x_max", None)

            def _print_clamp_counts(split_name, idx):
                if x_min is None or x_max is None:
                    print(f"CLAMP COUNTS {split_name}: x_min/x_max not set")
                    return
                if idx is None or len(idx) == 0:
                    print(f"CLAMP COUNTS {split_name}: empty split")
                    return
                x_split = ds.tensors[0][idx]  # (B,T,C)
                vals = x_split[:, :, feat_idx]
                vals_clamped = vals.clamp(min=x_min, max=x_max)
                total = vals_clamped.numel()
                n_min = int((vals_clamped == x_min).sum().item())
                n_max = int((vals_clamped == x_max).sum().item())
                p_min = (n_min / total * 100.0) if total > 0 else 0.0
                p_max = (n_max / total * 100.0) if total > 0 else 0.0
                print(
                    f"CLAMP COUNTS {split_name} | {target_feat} | "
                    f"= {x_min}: {n_min}/{total} ({p_min:.2f}%) | "
                    f"= {x_max}: {n_max}/{total} ({p_max:.2f}%)"
                )

            _print_clamp_counts("train", train_idx)
            if validate and len(val_idx):
                _print_clamp_counts("valid", val_idx)
            _print_clamp_counts("test", test_idx)
        else:
            print(f"CLAMP COUNTS: feature not found: {target_feat}")
        # --- Oversampling APENAS no treino
        train_sampler = self._make_weighted_sampler_from_classes(y_win[train_idx]) if len(train_idx) else None

        # --- DataLoaders
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size,
                                sampler=train_sampler if train_sampler is not None else None,
                                shuffle=False, pin_memory=True)
        # avaliador do treino na distribuição real (sem oversampling)
        val_loader  = DataLoader(Subset(ds, val_idx),  batch_size=batch_size,
                                shuffle=False, pin_memory=True) if validate and len(val_idx) else None
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size,
                                shuffle=False, pin_memory=True)

        # --- Logs de cobertura de grupos
        def _count(y):
            keys, cnts = np.unique(y, return_counts=True)
            return dict(zip(keys.tolist(), cnts.tolist()))
        # reduzido: não imprime grupos por padrão

        # --- Treino + ES sempre no TESTE
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)  # linear warmup
            # cosine decay até min_lr_factor
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_factor + (1 - min_lr_factor) * cosine
        self.to(device)
        best_score = float("inf"); best_epoch = 0; wait = patience
        steps_per_epoch = max(len(train_loader), 1)
        total_steps     = max(epochs * steps_per_epoch, 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        def _print_x_stats(t, feat_cols=None, raw=None, x_min=None, x_max=None):
            if t is None:
                return
            tt = t.detach()
            if tt.numel() == 0:
                return
            flat = tt.reshape(-1)
            q = torch.tensor([0.01, 0.50, 0.99], device=flat.device)
            p1, p50, p99 = torch.quantile(flat, q)
            over2 = (flat.abs() > 2).float().mean() * 100.0
            over3 = (flat.abs() > 3).float().mean() * 100.0
            over4 = (flat.abs() > 4).float().mean() * 100.0
            print(f">% |x| > 2 : {over2.item():.2f}%")
            print(f">% |x| > 3 : {over3.item():.2f}%")
            print(f">% |x| > 4 : {over4.item():.2f}%")
            print(f"min: {flat.min().item():.1f} max: {flat.max().item():.1f}")
            print(f"p1/p50/p99: [{p1.item():.8f} {p50.item():.8f} {p99.item():.8f}]")
            if raw is not None and x_min is not None and x_max is not None:
                raw_flat = raw.detach().reshape(-1)
                clipped = ((raw_flat < x_min) | (raw_flat > x_max)).float().sum().item()
                total = float(raw_flat.numel())
                pct = (clipped / total * 100.0) if total > 0 else 0.0
                print(f"clipped: {int(clipped)} / {int(total)} ({pct:.2f}%)")
                if raw.dim() == 3 and feat_cols is not None and len(feat_cols) == raw.shape[2]:
                    raw_bt = raw.detach()
                    per_feat_pct = (
                        ((raw_bt < x_min) | (raw_bt > x_max))
                        .float()
                        .mean(dim=(0, 1))
                        .cpu()
                        .numpy()
                    )
                    print("clipped% per feature:")
                    for name, val in zip(feat_cols, per_feat_pct):
                        print(f"{name}: {val * 100:.2f}%")
            if tt.dim() == 3 and feat_cols is not None and len(feat_cols) == tt.shape[2]:
                feat_min = tt.amin(dim=(0, 1))
                feat_max = tt.amax(dim=(0, 1))
                idx_min = int(torch.argmin(feat_min).item())
                idx_max = int(torch.argmax(feat_max).item())
                print(f"worst_min_feature: {feat_cols[idx_min]} {feat_min[idx_min].item():.1f}")
                print(f"worst_max_feature: {feat_cols[idx_max]} {feat_max[idx_max].item():.1f}")
        print_x_stats = False

        for ep in range(1, epochs + 1):
            epoch_start = time.time()
            # update epoch for SeqKAN top-k scheduling
            for m in self.modules():
                if hasattr(m, "current_epoch"):
                    m.current_epoch = ep
            self.train()
            total_train = [[0.0, 0.0] for _ in range(6)]  # L1..L6
            scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
            for batch_idx, batch in enumerate(train_loader):
                s = None
                p = None
                x, ts_batch, m = batch[0], batch[1], batch[2]  # x, ts_batch, m
                cc = torch.ones_like(x)
                if static_features_cols:  
                    s = batch[3]
                    if self.cost_columns is not None:
                        cc = batch[4]
                else:
                    if self.cost_columns is not None:
                        cc = batch[3]
                if predict_state_cols is not None:
                    p = batch[-1]
                x, ts_batch, m, cc = x.to(device), ts_batch.to(device), m.to(device), cc.to(device)
                x_batch = x
                if self.x_min is not None: x = x.clamp(min=self.x_min)
                if self.x_max is not None: x = x.clamp(max=self.x_max)
                if s is not None: s = s.to(device)
                if p is not None: p = p.to(device)

                if rebuild:
                    m_train   = self._make_random_mask(x,ts_batch,m,device,max_drop=max_drop)
                else:
                    m_train = m.clone()
                    m_train[:,-1,:]=0
                m_train_ts = m_train.any(dim=2, keepdim=True).float()

                x_masked = x * m_train
                optimizer.zero_grad(set_to_none=True)
                # always use test=False during training to enable diffusion noise
                out = self.forward(
                    x_masked, timestamps=ts_batch,
                    static_feats=s, return_x_hat=True, mask=m_train, mask_ts=m_train_ts, test=False,
                    only_gru=only_gru
                )
                (
                    state,
                    _,
                    state_tmax,
                    x_hat,
                    tmax,
                    noise,
                    noise_hat,
                    vae_x,
                    vae_mu,
                    vae_logvar,
                    vae_tmax,
                    vae_tmax_mu,
                    vae_tmax_logvar,
                    vae_logvar_obs,
                    vae_tmax_logvar_obs,
                ) = out
                

                if print_x_stats and batch_idx == 0:
                    _print_x_stats(
                        x,
                        feature_cols,
                        raw=x_batch,
                        x_min=self.x_min,
                        x_max=self.x_max,
                    )
                if device.type == "cuda":
                    autocast_ctx = torch.amp.autocast(device_type="cuda")
                else:
                    from contextlib import nullcontext
                    autocast_ctx = nullcontext()
                with autocast_ctx:

                    loss, L1, L2, L3, L4, L5, L6 = self._compute_loss(
                        x, 
                        x_hat if x_hat is not None else None,
                        tmax,
                        state,
                        state_tmax,
                        ts_batch,
                        m, 
                        m_train,
                        cc,
                        p,
                        delta_pred_window,
                        noise, 
                        noise_hat,
                        lambda1,
                        weight_tmax,
                        vae_mu,
                        vae_logvar,
                        vae_x if vae_x is not None else None,
                        vae_tmax,
                        vae_tmax_mu,
                        vae_tmax_logvar,
                        vae_logvar_obs if vae_logvar_obs is not None else None,
                        vae_tmax_logvar_obs,
                        kl_scale=self._kl_scale(ep, kl_start, kl_end, kl_warmup_epochs),
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
                stepped = (new_scale >= prev_scale)
                if stepped:
                    did_step_once = True
                    scheduler.step()     # agora é seguro (optimizer.step() ocorreu neste batch)
                for i, item in enumerate([L1, L2, L3, L4, L5, L6]):
                    total_train[i][0] += item[0]; total_train[i][1] += item[1]

            train_L1 = total_train[0][0] / max(total_train[0][1], 1.0)
            train_L2 = total_train[1][0] / max(total_train[1][1], 1.0)
            train_L3 = total_train[2][0] / max(total_train[2][1], 1.0)
            train_L4 = total_train[3][0] / max(total_train[3][1], 1.0)
            train_L5 = total_train[4][0] / max(total_train[4][1], 1.0)
            train_L6 = total_train[5][0] / max(total_train[5][1], 1.0)

            if validate and val_loader is not None:
                val_metrics = self.test_model(val_loader, y_seq=y_win[val_idx], 
                                              all_groups=all_groups, only_gru=only_gru,
                                              reconstruction_test=reconstruction_test,
                                              status_pred_window=delta_pred_window
                                              )
                val_parts = [
                    f"Epoch {ep}/{epochs} | "
                    f"Train(sampled) L1:{train_L1:.6f} L2:{train_L2:.6f} L3:{train_L3:.6f}  L4:{train_L4:.6f} L5:{train_L5:.6f} L6:{train_L6:.6f} | "
                ]
                if self.lam[0] > 0:
                    val_parts.append(
                        f"Val macro:{val_metrics['macro_mse']:.6f} ± {val_metrics['macro_se']:.6f} | "
                        f"Val micro:{val_metrics['micro_mse']:.6f} ± {val_metrics['micro_se']:.6f} "
                    )
                if self.lam[1] > 0:
                    val_parts.append(
                        f"Val macro (noise):{val_metrics['macro_mse_n']:.6f} ± {val_metrics['macro_se_n']:.6f} | "
                        f"Val micro (noise):{val_metrics['micro_mse_n']:.6f} ± {val_metrics['micro_se_n']:.6f} "
                    )
                if self.lam[2] > 0:
                    val_parts.append(
                        f"Val macro (state change):{val_metrics['macro_mse_s']:.6f} ± {val_metrics['macro_se_s']:.6f} | "
                        f"Val micro (state change):{val_metrics['micro_mse_s']:.6f} ± {val_metrics['micro_se_s']:.6f} "
                    )
                if self.lam[4] > 0:
                    val_parts.append(
                        f"Val macro (VAE):{val_metrics['macro_mse_v']:.6f} ± {val_metrics['macro_se_v']:.6f} | "
                        f"Val micro (VAE):{val_metrics['micro_mse_v']:.6f} ± {val_metrics['micro_se_v']:.6f} | "
                        f"ELBO:{val_metrics['micro_elbo_v']:.6f} NLL:{val_metrics['micro_nll_v']:.6f} cov90:{val_metrics['micro_cov_v']:.3f} width90:{val_metrics['micro_width_v']:.6f} "
                )
                if len(self.lam) > 5 and self.lam[5] > 0:
                    val_parts.append(
                        f"Val macro (VAE tmax):{val_metrics['macro_mse_vt']:.6f} ± {val_metrics['macro_se_vt']:.6f} | "
                        f"Val micro (VAE tmax):{val_metrics['micro_mse_vt']:.6f} ± {val_metrics['micro_se_vt']:.6f} | "
                        f"ELBO:{val_metrics['micro_elbo_vt']:.6f} NLL:{val_metrics['micro_nll_vt']:.6f} cov90:{val_metrics['micro_cov_vt']:.3f} width90:{val_metrics['micro_width_vt']:.6f} "
                    )
                print("".join(val_parts))
            else:
                print(
                    f"Epoch {ep}/{epochs} | "
                    f"Train(sampled) L1:{train_L1:.6f} L2:{train_L2:.6f} L3:{train_L3:.6f} L4:{train_L4:.6f} L5:{train_L5:.6f} L6:{train_L6:.6f} | "
                )

            # teste fixo e ES
            test_metrics = self.test_model(
                test_loader, y_seq=y_win[test_idx], 
                all_groups=all_groups, only_gru=only_gru, reconstruction_test=reconstruction_test,
                status_pred_window=delta_pred_window
                )
            # incluir losses de treino L1..L6 no dicionário retornado por época
            epoch_time = time.time() - epoch_start
            epoch_metrics = dict(test_metrics)
            epoch_metrics.update({
                "train_L1": float(train_L1),
                "train_L2": float(train_L2),
                "train_L3": float(train_L3),
                "train_L4": float(train_L4),
                "train_L5": float(train_L5),
                "train_L6": float(train_L6),
                "epoch_time": float(epoch_time),
            })
            yield epoch_metrics
            test_parts = []
            if self.lam[0] > 0:
                test_parts.append(
                    f"          >> Test macro:{test_metrics['macro_mse']:.6f} ± {test_metrics['macro_se']:.6f} | "
                    f"micro:{test_metrics['micro_mse']:.6f} ± {test_metrics['micro_se']:.6f}"
                )
            if self.lam[1] > 0:
                test_parts.append(
                    f"          >> Test (Noise) macro:{test_metrics['macro_mse_n']:.6f} ± {test_metrics['macro_se_n']:.6f} | "
                    f"micro:{test_metrics['micro_mse_n']:.6f} ± {test_metrics['micro_se_n']:.6f}"
                )
            if self.lam[2] > 0:
                test_parts.append(
                    f"          >> Test (State Change) macro:{test_metrics['macro_mse_s']:.6f} ± {test_metrics['macro_se_s']:.6f} | "
                    f"micro:{test_metrics['micro_mse_s']:.6f} ± {test_metrics['micro_se_s']:.6f}"
                )
            if self.lam[4] > 0:
                test_parts.append(
                    f"          >> Test (VAE) macro:{test_metrics['macro_mse_v']:.6f} ± {test_metrics['macro_se_v']:.6f} | "
                    f"micro:{test_metrics['micro_mse_v']:.6f} ± {test_metrics['micro_se_v']:.6f} | "
                    f"ELBO:{test_metrics['micro_elbo_v']:.6f} NLL:{test_metrics['micro_nll_v']:.6f} cov90:{test_metrics['micro_cov_v']:.3f} width90:{test_metrics['micro_width_v']:.6f}"
                )
            if len(self.lam) > 5 and self.lam[5] > 0:
                test_parts.append(
                    f"          >> Test (VAE tmax) macro:{test_metrics['macro_mse_vt']:.6f} ± {test_metrics['macro_se_vt']:.6f} | "
                    f"micro:{test_metrics['micro_mse_vt']:.6f} ± {test_metrics['micro_se_vt']:.6f} | "
                    f"ELBO:{test_metrics['micro_elbo_vt']:.6f} NLL:{test_metrics['micro_nll_vt']:.6f} cov90:{test_metrics['micro_cov_vt']:.3f} width90:{test_metrics['micro_width_vt']:.6f}"
                )
            if test_parts:
                print("\n".join(test_parts))

            if early_stopping:
                score = (test_metrics["micro_mse"]*self.lam[0] + test_metrics["micro_mse_n"]*self.lam[1] + \
                         test_metrics["micro_mse_s"]*self.lam[2] + test_metrics["micro_elbo_v"]*self.lam[4] + \
                         (test_metrics.get("micro_elbo_vt", 0.0) * self.lam[5] if len(self.lam)>5 else 0.0) + \
                2 * (test_metrics["micro_se"]*self.lam[0] + test_metrics["micro_se_n"]*self.lam[1] \
                     + test_metrics["micro_se_s"]*self.lam[2]))
                improved = score < best_score
                if improved:
                    self.save("tsdiffusion.pt"); best_score = score
                    best_epoch = ep; wait = patience
                else:
                    wait -= 1
                    if wait <= 0:
                        print(f"Early stopping at epoch {ep}/{epochs} (best test score: {best_score:.6f} @ epoch {best_epoch})")
                        break

        # --- Resultado final no teste fixo
        final_metrics = self.test_model(
            test_loader, y_seq=y_win[test_idx], all_groups=all_groups, only_gru=only_gru,
            reconstruction_test=reconstruction_test,status_pred_window=delta_pred_window
            )
        parts = ["TEST RESULTS | "]
        if self.lam[0] > 0:
            parts.append(f"macro: {final_metrics['macro_mse']:.6f} ± {final_metrics['macro_se']:.6f} | "
                         f"micro: {final_metrics['micro_mse']:.6f} ± {final_metrics['micro_se']:.6f} ")
        if self.lam[1] > 0:
            parts.append(f"macro (noise): {final_metrics['macro_mse_n']:.6f} ± {final_metrics['macro_se_n']:.6f} | "
                         f"micro (noise): {final_metrics['micro_mse_n']:.6f} ± {final_metrics['micro_se_n']:.6f} ")
        if self.lam[2] > 0:
            parts.append(f"macro (state change): {final_metrics['macro_mse_s']:.6f} ± {final_metrics['macro_se_s']:.6f} | "
                         f"micro (state change): {final_metrics['micro_mse_s']:.6f} ± {final_metrics['micro_se_s']:.6f} ")
        if self.lam[4] > 0:
            parts.append(f"macro (VAE): {final_metrics['macro_mse_v']:.6f} ± {final_metrics['macro_se_v']:.6f} | "
                         f"micro (VAE): {final_metrics['micro_mse_v']:.6f} ± {final_metrics['micro_se_v']:.6f} | "
                         f"ELBO:{final_metrics['micro_elbo_v']:.6f} NLL:{final_metrics['micro_nll_v']:.6f} cov90:{final_metrics['micro_cov_v']:.3f} width90:{final_metrics['micro_width_v']:.6f} ")
        if len(self.lam) > 5 and self.lam[5] > 0:
            parts.append(f"macro (VAE tmax): {final_metrics['macro_mse_vt']:.6f} ± {final_metrics['macro_se_vt']:.6f} | "
                         f"micro (VAE tmax): {final_metrics['micro_mse_vt']:.6f} ± {final_metrics['micro_se_vt']:.6f} | "
                         f"ELBO:{final_metrics['micro_elbo_vt']:.6f} NLL:{final_metrics['micro_nll_vt']:.6f} cov90:{final_metrics['micro_cov_vt']:.3f} width90:{final_metrics['micro_width_vt']:.6f} ")
        print("".join(parts))
        pg = test_metrics["per_group_mse"]
        pg_sew = test_metrics["per_group_se_w"]
        pg_cnt = test_metrics["per_group_counts"]
        print("          >> per_group (weighted SE):",
        {g: f"{pg[g]:.6f} ± {pg_sew[g]:.6f} (n={pg_cnt[g]})" for g in sorted(pg.keys())}
        )
        pg_n = test_metrics["per_group_mse_n"]
        pg_sew_n = test_metrics["per_group_se_w_n"]
        print("          >> per_group (weighted SE):",
        {g: f"{pg_n[g]:.6f} ± {pg_sew_n[g]:.6f} (n={pg_cnt[g]})" for g in sorted(pg.keys())}
        )
        pg_s = test_metrics["per_group_mse_s"]
        pg_sew_s = test_metrics["per_group_se_w_s"]
        print("          >> per_group (weighted SE):",
        {g: f"{pg_s[g]:.6f} ± {pg_sew_s[g]:.6f} (n={pg_cnt[g]})" for g in sorted(pg.keys())}
        )
        pg_v = test_metrics["per_group_mse_v"]
        pg_sew_v = test_metrics["per_group_se_w_v"]
        print("          >> per_group (weighted SE) VAE:",
        {g: f"{pg_v[g]:.6f} ± {pg_sew_v[g]:.6f} (n={pg_cnt[g]})" for g in sorted(pg.keys())}
        )
        if self.lam[4] > 0:
            pg_elbo_v = test_metrics["per_group_elbo_v"]
            print("          >> per_group ELBO VAE:",
            {g: f"{pg_elbo_v[g]:.6f}" for g in sorted(pg_elbo_v.keys())}
            )
        if len(self.lam) > 5 and self.lam[5] > 0:
            pg_elbo_vt = test_metrics["per_group_elbo_vt"]
            print("          >> per_group ELBO VAE tmax:",
            {g: f"{pg_elbo_vt[g]:.6f}" for g in sorted(pg_elbo_vt.keys())}
            )
        yield None

    def test_model_preforward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None
    ) -> torch.Tensor:
        z = self.denoise(
            state=self.encoder(torch.cat([x, mask], dim=-1)),
            timestamps=timestamps,
            static_feats=static_feats,
            device=x.device,
            steps=self.denoise_steps if hasattr(self, 'denoise_steps') else 100
            )
        return self.decoder(z)

    def test_model(
            self, 
            loader: DataLoader, 
            y_seq, 
            all_groups=None, 
            only_gru=False,
            reconstruction_test: bool = True,
            status_pred_window: int = 600
            ):
        if reconstruction_test:
            res = [self._test_model(
                loader, 
                y_seq, 
                all_groups, 
                only_gru,
                reconstruction_test,
                status_pred_window          
            ) for i in range(13)]
            res_index = [
                (res[i]["micro_mse"]*self.lam[0] + res[i]["micro_mse_n"]*self.lam[1] + \
                 res[i]["micro_mse_s"]*self.lam[2] + res[i]["micro_elbo_v"]*self.lam[4] + \
                 (res[i].get("micro_elbo_vt", 0.0) * self.lam[5] if len(self.lam)>5 else 0.0) + \
                2 * (res[i]["micro_se"]*self.lam[0] + res[i]["micro_se_n"]*self.lam[1] \
                     + res[i]["micro_se_s"] * self.lam[2] + res[i]["micro_se_v"]*self.lam[4] \
                     + (res[i].get("micro_se_vt", 0.0) * self.lam[5] if len(self.lam)>5 else 0.0)) 
                ) for
                i in range(len(res))
            ]
            r = np.median(res_index)
            idx = res_index.index(r)
            return res[idx]
        else:
            return self._test_model(
                loader, 
                y_seq, 
                all_groups, 
                only_gru,
                reconstruction_test,
                status_pred_window    
            )           


    def _test_model(
            self, 
            loader: DataLoader, 
            y_seq, 
            all_groups=None, 
            only_gru=False,
            reconstruction_test: bool = True,
            status_pred_window: int = 600
            ):
        """
        Avalia reconstrução por janela e retorna:
        - micro_mse, micro_se            (ponderado por nobs, todas as janelas)
        - macro_mse, macro_se            (média das MÉDIAS por grupo; SE entre grupos, não-ponderado)
        - per_group_mse                  (média ponderada por nobs dentro do grupo)
        - per_group_se_w                 (SE ponderado por nobs dentro do grupo; usa n_eff)
        - per_group_se_unw               (SE não-ponderado dentro do grupo; diagnóstico)
        - per_group_counts, per_group_sum_nobs
        Fórmulas de SE:
        - Não-ponderado: SE = std_amostral / sqrt(n)
        - Ponderado:     SE = sqrt( s2_w / n_eff ), onde
                            s2_w = Σ α_i (x_i - μ)^2  e  n_eff = 1 / Σ α_i^2, α_i = w_i / Σ w_i
        """
        import math
        device = next(self.parameters()).device
        self.eval()

        y_seq = np.asarray(y_seq, dtype=int)
        pos = 0

        # acumuladores por grupo
        G_W      = {}   # sum w = sum nobs
        G_SSE    = {}   # sum sse = sum w * mse
        G_WM2    = {}   # sum w * mse^2
        G_MSE    = {}   # lista de mse por janela (p/ SE não-ponderado)
        G_CNT    = {}   # nº janelas
        G_W_N      = {}   # sum w = sum nobs
        G_SSE_N    = {}   # sum sse = sum w * mse
        G_WM2_N    = {}   # sum w * mse^2
        G_MSE_N    = {}   # lista de mse por janela (p/ SE não-ponderado)
        G_W_S      = {}   # sum w = sum nobs
        G_SSE_S    = {}   # sum sse = sum w * mse
        G_WM2_S    = {}   # sum w * mse^2
        G_MSE_S    = {}   # lista de mse por janela (p/ SE não-ponderado)
        G_W_V      = {}
        G_SSE_V    = {}
        G_WM2_V    = {}
        G_MSE_V    = {}
        G_W_VT     = {}
        G_SSE_VT   = {}
        G_WM2_VT   = {}
        G_MSE_VT   = {}
        G_NLL_V    = {}
        G_NLL_VT   = {}
        G_COV_V    = {}
        G_TOTC_V   = {}
        G_WWIDTH_V = {}
        G_COV_VT   = {}
        G_TOTC_VT  = {}
        G_WWIDTH_VT= {}
        G_ELBO_V_NUM  = {}
        G_ELBO_V_DEN  = {}
        G_ELBO_VT_NUM = {}
        G_ELBO_VT_DEN = {}

        # globais (micro)
        T_W, T_SSE, T_WM2 = 0.0, 0.0, 0.0
        T_W_N, T_SSE_N, T_WM2_N = 0.0, 0.0, 0.0
        T_W_S, T_SSE_S, T_WM2_S = 0.0, 0.0, 0.0
        T_W_V, T_SSE_V, T_WM2_V = 0.0, 0.0, 0.0
        T_W_VT, T_SSE_VT, T_WM2_VT = 0.0, 0.0, 0.0
        T_NLL_V, T_W_NLL_V = 0.0, 0.0
        T_NLL_VT, T_W_NLL_VT = 0.0, 0.0
        T_COV_V, T_COV_DEN_V = 0.0, 0.0
        T_WIDTH_V = 0.0
        T_COV_VT, T_COV_DEN_VT = 0.0, 0.0
        T_WIDTH_VT = 0.0
        T_ELBO_V_NUM, T_ELBO_V_DEN = 0.0, 0.0
        T_ELBO_VT_NUM, T_ELBO_VT_DEN = 0.0, 0.0


        with torch.no_grad():
            for batch in loader:
                z90 = 1.6448536269514722  # quantil ~90% para intervalo simétrico
                s = None
                p = None
                x, ts_batch, m = batch[0], batch[1], batch[2]  # x, ts_batch, m
                cc = torch.ones_like(x)
                if self.static_dim>0:  
                    s = batch[3]
                    if self.cost_columns is not None:
                        cc = batch[4]
                else:
                    if self.cost_columns is not None:
                        cc = batch[3]
                if self.status_dim>0:
                    p = batch[-1]
                x, ts_batch, m, cc = x.to(device), ts_batch.to(device), m.to(device), cc.to(device)
                x_raw = x
                if self.x_min is not None: x = x.clamp(min=self.x_min)
                if self.x_max is not None: x = x.clamp(max=self.x_max)
                if s is not None: s = s.to(device)
                if p is not None: 
                    p = p.to(device)
                else:
                    p = ts_batch.unsqueeze(-1)

                B = x.shape[0]
                yb = y_seq[pos:pos+B]
                if len(yb) != B:
                    raise ValueError(f"test_model: desalinhado (batch={B}, labels={len(yb)} a partir de pos={pos}).")
                pos += B
                if reconstruction_test:
                    m_train = self._make_random_mask(x,ts_batch,m,device,max_drop=max_drop)
                    mask_ts = m_train.any(dim=2, keepdim=True).float()
                else:
                    m_train = m.clone(); m_train[:, -1, :] = 0.0; mask_ts = m_train.any(dim=2, keepdim=True).float()
                x_masked = x * m_train
                out = self.forward(
                    x_masked, timestamps=ts_batch, static_feats=s,
                    return_x_hat=True, mask=m_train, mask_ts=mask_ts, test=False, only_gru=only_gru
                )
                (
                    _,
                    _,
                    _,
                    x_hat,
                    tmax_hat,
                    noise,
                    noise_hat,
                    vae_x,
                    vae_mu,
                    vae_logvar,
                    vae_tmax,
                    vae_tmax_mu,
                    vae_tmax_logvar,
                    vae_logvar_obs,
                    vae_tmax_logvar_obs,
                ) = out
                x_forward = x
                x_target = x
                
                offset_state_pred = (p - ts_batch.unsqueeze(-1)).clamp(min=0,max=status_pred_window) / status_pred_window
                if tmax_hat is None:
                    offset_tmax = offset_state_pred
                else:
                    offset_tmax = (tmax_hat).clamp(min=0,max=1) 
                mask_tmax = m.new_zeros(m.size(0),m.size(1),1)
                mask_tmax[:,-1,:] = 1.0
                changing_state = ((offset_state_pred<1)*(offset_state_pred>0)).float()
                err = mask_tmax * ((offset_tmax - offset_state_pred) * changing_state)   # (B,T,S)
                #err_change = err

                sse_s_bt  = (err**2).sum(dim=(1, 2))           # (B,)
                nobs_s_bt = (changing_state * mask_tmax).sum(dim=(1, 2))+1e-8               # (B,)
                mse_s_bt  = (sse_s_bt / nobs_s_bt).detach().cpu().numpy()
                sse_s_bt  = sse_s_bt.detach().cpu().numpy()
                nobs_s_bt = nobs_s_bt.detach().cpu().numpy()
                

                if noise_hat is None:
                    noise_hat = torch.zeros_like(m_train)
                    noise = torch.zeros_like(m_train)
                sse_n_bt  = ((noise_hat-noise)**2).sum(dim=(1, 2))           # (B,)
                nobs_n_bt = (torch.ones_like(noise)*mask_ts).sum(dim=(1, 2)).clamp(min=1.0)               # (B,)
                mse_n_bt  = (sse_n_bt / nobs_n_bt).detach().cpu().numpy()
                sse_n_bt  = sse_n_bt.detach().cpu().numpy()
                nobs_n_bt = nobs_n_bt.detach().cpu().numpy()

                if x_hat is None: x_hat = x_forward
                mask_err = m * (1.0 - m_train)
                sse_bt  = ((x_target - x_hat)**2 * mask_err * cc).sum(dim=(1, 2))               # (B,)
                nobs_bt = (mask_err * cc).sum(dim=(1, 2)).clamp(min=1.0)                   # (B,)
                mse_bt  = (sse_bt / nobs_bt).detach().cpu().numpy()
                sse_bt  = sse_bt.detach().cpu().numpy()
                nobs_bt = nobs_bt.detach().cpu().numpy()

                if vae_x is None:
                    vae_x = x_target

                sse_v_bt  = ((x_target - vae_x)**2 * mask_err * cc).sum(dim=(1, 2))
                nobs_v_bt = (mask_err * cc).sum(dim=(1, 2)).clamp(min=1.0)
                mse_v_bt  = (sse_v_bt / nobs_v_bt).detach().cpu().numpy()
                sse_v_bt  = sse_v_bt.detach().cpu().numpy()
                nobs_v_bt = nobs_v_bt.detach().cpu().numpy()
                mu_v = vae_x
                logvar_v = vae_logvar_obs if vae_logvar_obs is not None else torch.zeros_like(mu_v)
                logvar_v = (logvar_v + math.log(self.sigma_temp)).clamp(min=-5.0, max=5.0)
                sigma_v = torch.exp(0.5 * logvar_v)
                # KL por amostra (soma sobre dims)
                if vae_mu is not None and vae_logvar is not None:
                    kl_v_per_b = -0.5 * (1 + vae_logvar - vae_mu.pow(2) - vae_logvar.exp()).sum(dim=(1, 2))
                else:
                    kl_v_per_b = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                nll_v = 0.5 * (
                    math.log(2 * math.pi) + logvar_v + ((x - mu_v) ** 2) / torch.exp(logvar_v)
                )
                nll_v = (nll_v * mask_err * cc).sum(dim=(1, 2))
                nll_v_bt = nll_v.detach().cpu().numpy()
                kl_v_bt = kl_v_per_b.detach().cpu().numpy()
                covered_v = (
                    (x >= (mu_v - z90 * sigma_v))
                    * (x <= (mu_v + z90 * sigma_v))
                    * (mask_err > 0)
                ).float()
                cov_v_bt = covered_v.sum(dim=(1, 2)).detach().cpu().numpy()
                cov_den_v_bt = (mask_err > 0).float().sum(dim=(1, 2)).clamp(min=1.0).detach().cpu().numpy()
                width_v_num = (2 * z90 * sigma_v * (mask_err > 0).float()).sum(dim=(1, 2)).detach().cpu().numpy()
                width_v_bt = width_v_num / cov_den_v_bt

                if vae_tmax is None:
                    offset_vae_tmax = offset_state_pred
                else:
                    offset_vae_tmax = vae_tmax.clamp(min=0,max=1) 
                err_vt = mask_tmax * ((offset_vae_tmax - offset_state_pred) * changing_state)    # (B,T,S)
                sse_vt_bt = (err_vt ** 2).sum(dim=(1,2))
                nobs_vt_bt = (changing_state * mask_tmax).sum(dim=(1, 2))+1e-8 
                mse_vt_bt = (sse_vt_bt / nobs_vt_bt).detach().cpu().numpy()
                sse_vt_bt = sse_vt_bt.detach().cpu().numpy()
                nobs_vt_bt = nobs_vt_bt.detach().cpu().numpy()
                mu_vt = offset_vae_tmax * changing_state * mask_tmax
                logvar_vt = vae_tmax_logvar_obs if vae_tmax_logvar_obs is not None else torch.zeros_like(mu_vt)
                logvar_vt = (logvar_vt + math.log(self.sigma_temp)).clamp(min=-5.0, max=5.0)
                sigma_vt = torch.exp(0.5 * logvar_vt) * changing_state * mask_tmax
                if vae_tmax_mu is not None and vae_tmax_logvar is not None:
                    kl_vt_per_b = -0.5 * (1 + vae_tmax_logvar - vae_tmax_mu.pow(2) - vae_tmax_logvar.exp()).sum(dim=(1, 2))
                else:
                    kl_vt_per_b = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                nll_vt = 0.5 * (
                    math.log(2 * math.pi) + logvar_vt + ((offset_state_pred - mu_vt) ** 2) / torch.exp(logvar_vt)
                )
                nll_vt = nll_vt.clamp_min(0.0)
                nll_vt = (nll_vt * changing_state * mask_tmax).sum(dim=(1, 2))
                nll_vt_bt = nll_vt.detach().cpu().numpy()
                kl_vt_bt = kl_vt_per_b.detach().cpu().numpy()
                covered_vt = (
                    (offset_state_pred >= (mu_vt - z90 * sigma_vt))
                    * (offset_state_pred  <= (mu_vt + z90 * sigma_vt)) * changing_state * mask_tmax
                    
                ).float()
                cov_vt_bt = covered_vt.sum(dim=(1, 2)).detach().cpu().numpy()
                cov_den_vt_bt = (changing_state * mask_tmax).sum(dim=(1, 2)).clamp(min=1e-8).detach().cpu().numpy()
                width_vt_num = (2 * z90 * sigma_vt * changing_state * mask_tmax).sum(dim=(1, 2)).detach().cpu().numpy()
                width_vt_bt = width_vt_num / cov_den_vt_bt

                # dimensões latentes (por amostra) para normalizar o KL
                if vae_logvar is not None:
                    z_latent_dim = float(vae_logvar.shape[1] * vae_logvar.shape[2])
                else:
                    z_latent_dim = 1.0
                if vae_tmax_logvar is not None:
                    z_tmax_latent_dim = float(vae_tmax_logvar.shape[1] * vae_tmax_logvar.shape[2])
                else:
                    z_tmax_latent_dim = 1.0

                for b in range(B):
                    g   = int(yb[b])
                    w   = float(nobs_bt[b])
                    mse = float(mse_bt[b])
                    sse = float(sse_bt[b])
                    w_v   = float(nobs_v_bt[b])
                    mse_v = float(mse_v_bt[b])
                    sse_v = float(sse_v_bt[b])
                    nll_vb = float(nll_v_bt[b])
                    kl_vb = float(kl_v_bt[b])
                    # NLL média por ponto observado e KL médio por dimensão latente
                    nll_mean_vb = nll_vb / max(w_v, 1e-8)
                    kl_mean_vb = kl_vb / max(z_latent_dim, 1.0)
                    elbo_vb = nll_mean_vb + kl_mean_vb
                    cov_vb = float(cov_v_bt[b])
                    cov_den_vb = float(cov_den_v_bt[b])
                    width_vb = float(width_v_bt[b])
                    w_n   = float(nobs_n_bt[b])
                    mse_n = float(mse_n_bt[b])
                    sse_n = float(sse_n_bt[b])
                    w_s   = float(nobs_s_bt[b])
                    mse_s = float(mse_s_bt[b])
                    sse_s = float(sse_s_bt[b])


                    G_W[g]   = G_W.get(g, 0.0)   + w
                    G_SSE[g] = G_SSE.get(g, 0.0) + sse
                    G_WM2[g] = G_WM2.get(g, 0.0) + (w * mse * mse)
                    G_MSE.setdefault(g, []).append(mse)
                    G_W_V[g] = G_W_V.get(g, 0.0) + w_v
                    G_SSE_V[g] = G_SSE_V.get(g, 0.0) + sse_v
                    G_WM2_V[g] = G_WM2_V.get(g, 0.0) + (w_v * mse_v * mse_v)
                    G_MSE_V.setdefault(g, []).append(mse_v)
                    w_vt   = float(nobs_vt_bt[b])
                    mse_vt = float(mse_vt_bt[b])
                    sse_vt = float(sse_vt_bt[b])
                    nll_vtb = float(nll_vt_bt[b])
                    kl_vtb = float(kl_vt_bt[b])
                    nll_mean_vtb = nll_vtb / max(w_vt, 1e-8)
                    kl_mean_vtb = kl_vtb / max(z_tmax_latent_dim, 1.0)
                    elbo_vtb = nll_mean_vtb + kl_mean_vtb
                    cov_vtb = float(cov_vt_bt[b])
                    cov_den_vtb = float(cov_den_vt_bt[b])
                    width_vtb = float(width_vt_bt[b])
                    G_W_VT[g] = G_W_VT.get(g, 0.0) + w_vt
                    G_SSE_VT[g] = G_SSE_VT.get(g, 0.0) + sse_vt
                    G_WM2_VT[g] = G_WM2_VT.get(g, 0.0) + (w_vt * mse_vt * mse_vt)
                    G_MSE_VT.setdefault(g, []).append(mse_vt)
                    G_NLL_V[g] = G_NLL_V.get(g, 0.0) + nll_vb
                    G_TOTC_V[g] = G_TOTC_V.get(g, 0.0) + cov_den_vb
                    G_COV_V[g] = G_COV_V.get(g, 0.0) + cov_vb
                    G_WWIDTH_V[g] = G_WWIDTH_V.get(g, 0.0) + width_vb * cov_den_vb
                    G_ELBO_V_NUM[g] = G_ELBO_V_NUM.get(g, 0.0) + elbo_vb * w_v
                    G_ELBO_V_DEN[g] = G_ELBO_V_DEN.get(g, 0.0) + w_v
                    G_NLL_VT[g] = G_NLL_VT.get(g, 0.0) + nll_vtb
                    G_TOTC_VT[g] = G_TOTC_VT.get(g, 0.0) + cov_den_vtb
                    G_COV_VT[g] = G_COV_VT.get(g, 0.0) + cov_vtb
                    G_WWIDTH_VT[g] = G_WWIDTH_VT.get(g, 0.0) + width_vtb * cov_den_vtb
                    G_ELBO_VT_NUM[g] = G_ELBO_VT_NUM.get(g, 0.0) + elbo_vtb * w_vt
                    G_ELBO_VT_DEN[g] = G_ELBO_VT_DEN.get(g, 0.0) + w_vt
                    G_W_N[g]   = G_W_N.get(g, 0.0)   + w_n
                    G_SSE_N[g] = G_SSE_N.get(g, 0.0) + sse_n
                    G_WM2_N[g] = G_WM2_N.get(g, 0.0) + (w_n * mse_n * mse_n)
                    G_MSE_N.setdefault(g, []).append(mse_n)
                    G_W_S[g]   = G_W_S.get(g, 0.0)   + w_s
                    G_SSE_S[g] = G_SSE_S.get(g, 0.0) + sse_s
                    G_WM2_S[g] = G_WM2_S.get(g, 0.0) + (w_s * mse_s * mse_s)
                    G_MSE_S.setdefault(g, []).append(mse_s)
                    G_CNT[g] = G_CNT.get(g, 0) + 1

                    T_W   += w
                    T_SSE += sse
                    T_WM2 += (w * mse * mse)
                    T_W_V += w_v
                    T_SSE_V += sse_v
                    T_WM2_V += (w_v * mse_v * mse_v)
                    T_W_N   += w_n
                    T_SSE_N += sse_n
                    T_WM2_N += (w_n * mse_n * mse_n)
                    T_W_S   += w_s
                    T_SSE_S += sse_s
                    T_WM2_S += (w_s * mse_s * mse_s)
                    T_W_VT += w_vt
                    T_SSE_VT += sse_vt
                    T_WM2_VT += (w_vt * mse_vt * mse_vt)
                    T_NLL_V += nll_vb
                    T_W_NLL_V += w_v
                    T_COV_V += cov_vb
                    T_COV_DEN_V += cov_den_vb
                    T_WIDTH_V += width_vb * cov_den_vb
                    T_ELBO_V_NUM += elbo_vb * w_v
                    T_ELBO_V_DEN += w_v
                    T_NLL_VT += nll_vtb
                    T_W_NLL_VT += w_vt
                    T_COV_VT += cov_vtb
                    T_COV_DEN_VT += cov_den_vtb
                    T_WIDTH_VT += width_vtb * cov_den_vtb
                    T_ELBO_VT_NUM += elbo_vtb * w_vt
                    T_ELBO_VT_DEN += w_vt


        # grupos a reportar
        if all_groups is None:
            groups = sorted(G_W.keys())
        else:
            groups = sorted(np.unique(list(all_groups)).tolist())

        if not groups:
            return {
                "macro_mse": float("nan"), "macro_se": float("nan"),
                "micro_mse": float("nan"), "micro_se": float("nan"),
                "per_group_mse": {}, "per_group_se_w": {}, "per_group_se_unw": {},
                "per_group_counts": {}, "per_group_sum_nobs": {},
                "macro_mse_n": float("nan"), "macro_se_n": float("nan"),
                "micro_mse_n": float("nan"), "micro_se_n": float("nan"),
                "per_group_mse_n": {}, "per_group_se_w_n": {}, "per_group_se_unw_n": {},
                "per_group_sum_nobs_n": {},
                "macro_mse_s": float("nan"), "macro_se_s": float("nan"),
                "micro_mse_s": float("nan"), "micro_se_s": float("nan"),
                "per_group_mse_s": {}, "per_group_se_w_s": {}, "per_group_se_unw_s": {},
                "per_group_sum_nobs_s": {},
                "macro_mse_v": float("nan"), "macro_se_v": float("nan"),
                "micro_mse_v": float("nan"), "micro_se_v": float("nan"),
                "per_group_mse_v": {}, "per_group_se_w_v": {}, "per_group_se_unw_v": {},
                "per_group_sum_nobs_v": {},
                "macro_mse_vt": float("nan"), "macro_se_vt": float("nan"),
                "micro_mse_vt": float("nan"), "micro_se_vt": float("nan"),
                "per_group_mse_vt": {}, "per_group_se_w_vt": {}, "per_group_se_unw_vt": {},
                "per_group_sum_nobs_vt": {},
                "micro_nll_v": float("nan"), "micro_nll_vt": float("nan"),
                "macro_nll_v": float("nan"), "macro_nll_vt": float("nan"),
                "micro_elbo_v": float("nan"), "micro_elbo_vt": float("nan"),
                "macro_elbo_v": float("nan"), "macro_elbo_vt": float("nan"),
                "per_group_elbo_v": {}, "per_group_elbo_vt": {},
                "micro_cov_v": float("nan"), "micro_cov_vt": float("nan"),
                "macro_cov_v": float("nan"), "macro_cov_vt": float("nan"),
                "micro_width_v": float("nan"), "micro_width_vt": float("nan"),
                "macro_width_v": float("nan"), "macro_width_vt": float("nan")
            }

        # por grupo
        per_group_mse       = {}
        per_group_se_w      = {}
        per_group_se_unw    = {}
        per_group_counts    = {}
        per_group_sum_nobs  = {}
        per_group_mse_n       = {}
        per_group_se_w_n      = {}
        per_group_se_unw_n    = {}
        per_group_sum_nobs_n  = {}
        per_group_mse_s       = {}
        per_group_se_w_s      = {}
        per_group_se_unw_s    = {}
        per_group_sum_nobs_s  = {}
        per_group_mse_v       = {}
        per_group_se_w_v      = {}
        per_group_se_unw_v    = {}
        per_group_sum_nobs_v  = {}
        per_group_mse_vt       = {}
        per_group_se_w_vt      = {}
        per_group_se_unw_vt    = {}
        per_group_sum_nobs_vt  = {}
        per_group_nll_v        = {}
        per_group_nll_vt       = {}
        per_group_cov_v        = {}
        per_group_cov_vt       = {}
        per_group_width_v      = {}
        per_group_width_vt     = {}
        per_group_elbo_v       = {}
        per_group_elbo_vt      = {}
        per_group_nll_v        = {}
        per_group_nll_vt       = {}
        per_group_cov_v        = {}
        per_group_cov_vt       = {}
        per_group_width_v      = {}
        per_group_width_vt     = {}


        for g in groups:
            Wg = G_W.get(g, 0.0)
            Wg_v = G_W_V.get(g, 0.0)
            Wg_n = G_W_N.get(g, 0.0)
            Wg_s = G_W_S.get(g, 0.0)
            Wg_vt = G_W_VT.get(g, 0.0)
            per_group_sum_nobs[g] = float(Wg)
            per_group_sum_nobs_v[g] = float(Wg_v)
            per_group_sum_nobs_n[g] = float(Wg_n)
            per_group_sum_nobs_s[g] = float(Wg_s)
            per_group_sum_nobs_vt[g] = float(Wg_vt)
            cnt = G_CNT.get(g, 0)
            per_group_counts[g] = int(cnt)

            if Wg > 0.0:
                mu_g = G_SSE[g] / Wg                           # média ponderada por nobs
                per_group_mse[g] = float(mu_g)

                # SE não-ponderado (amostral) entre janelas
                mses = np.asarray(G_MSE.get(g, []), dtype=float)
                if mses.size >= 2:
                    std_unw = float(np.std(mses, ddof=1))
                    per_group_se_unw[g] = std_unw / math.sqrt(mses.size)
                elif mses.size == 1:
                    per_group_se_unw[g] = float("nan")
                else:
                    per_group_se_unw[g] = float("nan")

                # SE ponderado por nobs (usa n_eff)
                # s2_w = E_w[(X - mu)^2] = (Σ w x^2)/Wg - mu_g^2
                s2_w = max(G_WM2[g] / Wg - mu_g * mu_g, 0.0)
                # n_eff = 1 / Σ α_i^2, com α_i = w_i / Wg
                # para computar Σ α_i^2, precisamos das α_i por janela do grupo:
                # reusa mses + w por grupo (não armazenamos w_i individuais por grupo; então recompute α_i via segunda passada)
                # -> atalho: acumule Σ w_i^2 enquanto itera (opção mais eficiente).
                # Como não acumulamos, aproximamos n_eff por contagem não-ponderada quando não há forte desbalanceamento:
                # Melhor: compute n_eff aproximado por (Wg^2) / Σ w_i^2 – para isso, reconstruímos Σ w_i^2 do G_WM2 e s2_w:
                # G_WM2 = Σ w_i x_i^2. Não temos Σ w_i^2 diretamente; então usamos aproximação conservadora n_eff = cnt.
                # Se quiser exato, guarde Σ w_i^2 durante a passada no loader.
                if cnt >= 2:
                    n_eff = cnt  # aproximação segura; se quiser exato, armazene sum_w2 por grupo
                    per_group_se_w[g] = float(math.sqrt(s2_w / n_eff))
                else:
                    per_group_se_w[g] = float("nan")
            else:
                per_group_mse[g]    = float("nan")
                per_group_se_unw[g] = float("nan")
                per_group_se_w[g]   = float("nan")

            if Wg_n > 0.0:
                mu_g_n = G_SSE_N[g] / Wg_n    
                per_group_mse_n[g] = float(mu_g_n)

                mses_n = np.asarray(G_MSE_N.get(g, []), dtype=float)

                if mses_n.size >= 2:
                    std_unw_n = float(np.std(mses_n, ddof=1))
                    per_group_se_unw_n[g] = std_unw_n / math.sqrt(mses_n.size)
                elif mses_n.size == 1:
                    per_group_se_unw_n[g] = float("nan")
                else:
                    per_group_se_unw_n[g] = float("nan")

                s2_w_n = max(G_WM2_N[g] / Wg_n - mu_g_n * mu_g_n, 0.0)

                if cnt >= 2:
                    n_eff = cnt  # aproximação segura; se quiser exato, armazene sum_w2 por grupo
                    per_group_se_w_n[g] = float(math.sqrt(s2_w_n / n_eff))
                else:
                    per_group_se_w_n[g] = float("nan")
            else:
                per_group_mse_n[g]    = float("nan")
                per_group_se_unw_n[g] = float("nan")
                per_group_se_w_n[g]   = float("nan")

            if Wg_s > 0.0:
                mu_g_s = G_SSE_S[g] / Wg_s    
                per_group_mse_s[g] = float(mu_g_s)

                mses_s = np.asarray(G_MSE_S.get(g, []), dtype=float)

                if mses_s.size >= 2:
                    std_unw_s = float(np.std(mses_s, ddof=1))
                    per_group_se_unw_s[g] = std_unw_s / math.sqrt(mses_s.size)
                elif mses_s.size == 1:
                    per_group_se_unw_s[g] = float("nan")
                else:
                    per_group_se_unw_s[g] = float("nan")

                s2_w_s = max(G_WM2_S[g] / Wg_s - mu_g_s * mu_g_s, 0.0)

                if cnt >= 2:
                    n_eff = cnt  # aproximação segura; se quiser exato, armazene sum_w2 por grupo
                    per_group_se_w_s[g] = float(math.sqrt(s2_w_s / n_eff))
                else:
                    per_group_se_w_s[g] = float("nan")
            else:
                per_group_mse_s[g]    = float("nan")
                per_group_se_unw_s[g] = float("nan")
                per_group_se_w_s[g]   = float("nan")

            if Wg_v > 0.0:
                mu_g_v = G_SSE_V[g] / Wg_v
                per_group_mse_v[g] = float(mu_g_v)

                mses_v = np.asarray(G_MSE_V.get(g, []), dtype=float)
                if mses_v.size >= 2:
                    std_unw_v = float(np.std(mses_v, ddof=1))
                    per_group_se_unw_v[g] = std_unw_v / math.sqrt(mses_v.size)
                elif mses_v.size == 1:
                    per_group_se_unw_v[g] = float("nan")
                else:
                    per_group_se_unw_v[g] = float("nan")

                s2_w_v = max(G_WM2_V[g] / Wg_v - mu_g_v * mu_g_v, 0.0)

                if cnt >= 2:
                    n_eff = cnt
                    per_group_se_w_v[g] = float(math.sqrt(s2_w_v / n_eff))
                else:
                    per_group_se_w_v[g] = float("nan")
                per_group_nll_v[g] = float(G_NLL_V.get(g, 0.0) / max(Wg_v, 1e-8))
                per_group_cov_v[g] = float(G_COV_V.get(g, 0.0) / max(G_TOTC_V.get(g, 0.0), 1.0))
                per_group_width_v[g] = float(G_WWIDTH_V.get(g, 0.0) / max(G_TOTC_V.get(g, 0.0), 1.0))
                per_group_elbo_v[g] = float(G_ELBO_V_NUM.get(g, 0.0) / max(G_ELBO_V_DEN.get(g, 0.0), 1e-8))
            else:
                per_group_mse_v[g]    = float("nan")
                per_group_se_unw_v[g] = float("nan")
                per_group_se_w_v[g]   = float("nan")
                per_group_nll_v[g]    = float("nan")
                per_group_cov_v[g]    = float("nan")
                per_group_width_v[g]  = float("nan")
                per_group_elbo_v[g]   = float("nan")

            if Wg_vt > 0.0:
                mu_g_vt = G_SSE_VT[g] / Wg_vt
                per_group_mse_vt[g] = float(mu_g_vt)

                mses_vt = np.asarray(G_MSE_VT.get(g, []), dtype=float)
                if mses_vt.size >= 2:
                    std_unw_vt = float(np.std(mses_vt, ddof=1))
                    per_group_se_unw_vt[g] = std_unw_vt / math.sqrt(mses_vt.size)
                elif mses_vt.size == 1:
                    per_group_se_unw_vt[g] = float("nan")
                else:
                    per_group_se_unw_vt[g] = float("nan")

                s2_w_vt = max(G_WM2_VT[g] / Wg_vt - mu_g_vt * mu_g_vt, 0.0)

                if cnt >= 2:
                    n_eff = cnt
                    per_group_se_w_vt[g] = float(math.sqrt(s2_w_vt / n_eff))
                else:
                    per_group_se_w_vt[g] = float("nan")
                per_group_nll_vt[g] = float(G_NLL_VT.get(g, 0.0) / max(Wg_vt, 1e-8))
                per_group_cov_vt[g] = float(G_COV_VT.get(g, 0.0) / max(G_TOTC_VT.get(g, 0.0), 1.0))
                per_group_width_vt[g] = float(G_WWIDTH_VT.get(g, 0.0) / max(G_TOTC_VT.get(g, 0.0), 1.0))
                per_group_elbo_vt[g] = float(G_ELBO_VT_NUM.get(g, 0.0) / max(G_ELBO_VT_DEN.get(g, 0.0), 1e-8))
            else:
                per_group_mse_vt[g]    = float("nan")
                per_group_se_unw_vt[g] = float("nan")
                per_group_se_w_vt[g]   = float("nan")
                per_group_nll_vt[g]    = float("nan")
                per_group_cov_vt[g]    = float("nan")
                per_group_width_vt[g]  = float("nan")
                per_group_elbo_vt[g]   = float("nan")


        # micro (ponderado por nobs) – SE ponderado
        if T_W > 0.0:
            micro_mse = T_SSE / T_W
            micro_mse_n = T_SSE_N / T_W_N
            micro_mse_s = T_SSE_S / T_W_S
            micro_mse_v = T_SSE_V / T_W_V if T_W_V > 0 else float("nan")
            micro_mse_vt = T_SSE_VT / T_W_VT if T_W_VT > 0 else float("nan")
            # var ponderada populacional
            s2_micro = max(T_WM2 / T_W - micro_mse * micro_mse, 0.0)
            s2_micro_n = max(T_WM2_N / T_W_N - micro_mse_n * micro_mse_n, 0.0)
            s2_micro_s = max(T_WM2_S / T_W_S - micro_mse_s * micro_mse_s, 0.0)
            s2_micro_v = max(T_WM2_V / T_W_V - micro_mse_v * micro_mse_v, 0.0) if T_W_V > 0 else float("nan")
            s2_micro_vt = max(T_WM2_VT / T_W_VT - micro_mse_vt * micro_mse_vt, 0.0) if T_W_VT > 0 else float("nan")
            # n_eff global (aprox): use número de janelas (contagem total) como proxy
            # Para n_eff exato, acumule Σ w_i^2 globalmente. Se puder, acrescente 'sum_w2' no loop.
            total_cnt = int(sum(per_group_counts.values()))
            micro_se = float(math.sqrt(s2_micro / max(total_cnt, 1)))
            micro_se_n = float(math.sqrt(s2_micro_n / max(total_cnt, 1)))
            micro_se_s = float(math.sqrt(s2_micro_s / max(total_cnt, 1)))
            micro_se_v = float(math.sqrt(s2_micro_v / max(total_cnt, 1))) if T_W_V > 0 else float("nan")
            micro_se_vt = float(math.sqrt(s2_micro_vt / max(total_cnt, 1))) if T_W_VT > 0 else float("nan")
            micro_nll_v = float(T_NLL_V / max(T_W_NLL_V, 1e-8)) if T_W_NLL_V > 0 else float("nan")
            micro_nll_vt = float(T_NLL_VT / max(T_W_NLL_VT, 1e-8)) if T_W_NLL_VT > 0 else float("nan")
            micro_elbo_v = float(T_ELBO_V_NUM / max(T_ELBO_V_DEN, 1e-8)) if T_ELBO_V_DEN > 0 else float("nan")
            micro_elbo_vt = float(T_ELBO_VT_NUM / max(T_ELBO_VT_DEN, 1e-8)) if T_ELBO_VT_DEN > 0 else float("nan")
            micro_cov_v = float(T_COV_V / max(T_COV_DEN_V, 1.0)) if T_COV_DEN_V > 0 else float("nan")
            micro_cov_vt = float(T_COV_VT / max(T_COV_DEN_VT, 1.0)) if T_COV_DEN_VT > 0 else float("nan")
            micro_width_v = float(T_WIDTH_V / max(T_COV_DEN_V, 1.0)) if T_COV_DEN_V > 0 else float("nan")
            micro_width_vt = float(T_WIDTH_VT / max(T_COV_DEN_VT, 1.0)) if T_COV_DEN_VT > 0 else float("nan")
        else:
            micro_mse = float("nan"); micro_se = float("nan")
            micro_mse_n = float("nan"); micro_se_n = float("nan")
            micro_mse_s = float("nan"); micro_se_s = float("nan")
            micro_mse_v = float("nan"); micro_se_v = float("nan")
            micro_mse_vt = float("nan"); micro_se_vt = float("nan")
            micro_nll_v = float("nan"); micro_nll_vt = float("nan")
            micro_cov_v = float("nan"); micro_cov_vt = float("nan")
            micro_width_v = float("nan"); micro_width_vt = float("nan")
            micro_elbo_v = float("nan"); micro_elbo_vt = float("nan")

        # macro: média das MÉDIAS por grupo (não-ponderado) e SE entre grupos
        mu_gs = [per_group_mse[g] for g in groups if np.isfinite(per_group_mse[g])]
        mu_gs_n = [per_group_mse_n[g] for g in groups if np.isfinite(per_group_mse_n[g])]
        mu_gs_s = [per_group_mse_s[g] for g in groups if np.isfinite(per_group_mse_s[g])]
        mu_gs_v = [per_group_mse_v[g] for g in groups if np.isfinite(per_group_mse_v[g])]
        mu_gs_vt = [per_group_mse_vt[g] for g in groups if np.isfinite(per_group_mse_vt[g])]
        mu_gs_nll_v = [per_group_nll_v[g] for g in groups if np.isfinite(per_group_nll_v[g])] if 'per_group_nll_v' in locals() else []
        mu_gs_nll_vt = [per_group_nll_vt[g] for g in groups if np.isfinite(per_group_nll_vt[g])] if 'per_group_nll_vt' in locals() else []
        mu_gs_cov_v = [per_group_cov_v[g] for g in groups if np.isfinite(per_group_cov_v[g])] if 'per_group_cov_v' in locals() else []
        mu_gs_cov_vt = [per_group_cov_vt[g] for g in groups if np.isfinite(per_group_cov_vt[g])] if 'per_group_cov_vt' in locals() else []
        mu_gs_width_v = [per_group_width_v[g] for g in groups if np.isfinite(per_group_width_v[g])] if 'per_group_width_v' in locals() else []
        mu_gs_width_vt = [per_group_width_vt[g] for g in groups if np.isfinite(per_group_width_vt[g])] if 'per_group_width_vt' in locals() else []
        mu_gs_elbo_v = [per_group_elbo_v[g] for g in groups if np.isfinite(per_group_elbo_v[g])] if 'per_group_elbo_v' in locals() else []
        mu_gs_elbo_vt = [per_group_elbo_vt[g] for g in groups if np.isfinite(per_group_elbo_vt[g])] if 'per_group_elbo_vt' in locals() else []
        G_eff = len(mu_gs)
        G_eff_n = len(mu_gs_n)
        G_eff_s = len(mu_gs_s)
        G_eff_v = len(mu_gs_v)
        G_eff_vt = len(mu_gs_vt)
        G_eff_nll_v = len(mu_gs_nll_v)
        G_eff_nll_vt = len(mu_gs_nll_vt)
        G_eff_cov_v = len(mu_gs_cov_v)
        G_eff_cov_vt = len(mu_gs_cov_vt)
        G_eff_width_v = len(mu_gs_width_v)
        G_eff_width_vt = len(mu_gs_width_vt)
        G_eff_elbo_v = len(mu_gs_elbo_v)
        G_eff_elbo_vt = len(mu_gs_elbo_vt)
        if G_eff >= 1:
            macro_mse = float(np.mean(mu_gs))
            if G_eff >= 2:
                std_between = float(np.std(mu_gs, ddof=1))
                macro_se = std_between / math.sqrt(G_eff)
            else:
                macro_se = float("nan")
        else:
            macro_mse = float("nan"); macro_se = float("nan")

        if G_eff_n >= 1:
            macro_mse_n = float(np.mean(mu_gs_n))
            if G_eff_n >= 2:
                std_between_n = float(np.std(mu_gs_n, ddof=1))
                macro_se_n = std_between_n / math.sqrt(G_eff_n)
            else:
                macro_se_n = float("nan")
        else:
            macro_mse_n = float("nan"); macro_se_n = float("nan")

        if G_eff_s >= 1:
            macro_mse_s = float(np.mean(mu_gs_s))
            if G_eff_s >= 2:
                std_between_s = float(np.std(mu_gs_s, ddof=1))
                macro_se_s = std_between_s / math.sqrt(G_eff_s)
            else:
                macro_se_s = float("nan")
        else:
            macro_mse_s = float("nan"); macro_se_s = float("nan")
        if G_eff_v >= 1:
            macro_mse_v = float(np.mean(mu_gs_v))
            if G_eff_v >= 2:
                std_between_v = float(np.std(mu_gs_v, ddof=1))
                macro_se_v = std_between_v / math.sqrt(G_eff_v)
            else:
                macro_se_v = float("nan")
        else:
            macro_mse_v = float("nan"); macro_se_v = float("nan")
        if G_eff_vt >= 1:
            macro_mse_vt = float(np.mean(mu_gs_vt))
            if G_eff_vt >= 2:
                std_between_vt = float(np.std(mu_gs_vt, ddof=1))
                macro_se_vt = std_between_vt / math.sqrt(G_eff_vt)
            else:
                macro_se_vt = float("nan")
        else:
            macro_mse_vt = float("nan"); macro_se_vt = float("nan")

        macro_nll_v = float(np.mean(mu_gs_nll_v)) if G_eff_nll_v >= 1 else float("nan")
        macro_nll_vt = float(np.mean(mu_gs_nll_vt)) if G_eff_nll_vt >= 1 else float("nan")
        macro_cov_v = float(np.mean(mu_gs_cov_v)) if G_eff_cov_v >= 1 else float("nan")
        macro_cov_vt = float(np.mean(mu_gs_cov_vt)) if G_eff_cov_vt >= 1 else float("nan")
        macro_width_v = float(np.mean(mu_gs_width_v)) if G_eff_width_v >= 1 else float("nan")
        macro_width_vt = float(np.mean(mu_gs_width_vt)) if G_eff_width_vt >= 1 else float("nan")
        macro_elbo_v = float(np.mean(mu_gs_elbo_v)) if G_eff_elbo_v >= 1 else float("nan")
        macro_elbo_vt = float(np.mean(mu_gs_elbo_vt)) if G_eff_elbo_vt >= 1 else float("nan")


        return {
            "macro_mse": macro_mse, "macro_se": macro_se,
            "micro_mse": micro_mse, "micro_se": micro_se,
            "per_group_mse": per_group_mse,
            "per_group_se_w": per_group_se_w,
            "per_group_se_unw": per_group_se_unw,
            "per_group_counts": per_group_counts,
            "per_group_sum_nobs": per_group_sum_nobs,
            "macro_mse_n": macro_mse_n, "macro_se_n": macro_se_n,
            "micro_mse_n": micro_mse_n, "micro_se_n": micro_se_n,
            "per_group_mse_n": per_group_mse_n,
            "per_group_se_w_n": per_group_se_w_n,
            "per_group_se_unw_n": per_group_se_unw_n,
            "per_group_sum_nobs_n": per_group_sum_nobs_n,
            "macro_mse_s": macro_mse_s, "macro_se_s": macro_se_s,
            "micro_mse_s": micro_mse_s, "micro_se_s": micro_se_s,
            "per_group_mse_s": per_group_mse_s,
            "per_group_se_w_s": per_group_se_w_s,
            "per_group_se_unw_s": per_group_se_unw_s,
            "per_group_sum_nobs_s": per_group_sum_nobs_s,
            "macro_mse_v": macro_mse_v, "macro_se_v": macro_se_v,
            "micro_mse_v": micro_mse_v, "micro_se_v": micro_se_v,
            "per_group_mse_v": per_group_mse_v,
            "per_group_se_w_v": per_group_se_w_v,
            "per_group_se_unw_v": per_group_se_unw_v,
            "per_group_sum_nobs_v": per_group_sum_nobs_v,
            "macro_mse_vt": macro_mse_vt, "macro_se_vt": macro_se_vt,
            "micro_mse_vt": micro_mse_vt, "micro_se_vt": micro_se_vt,
            "per_group_mse_vt": per_group_mse_vt,
            "per_group_se_w_vt": per_group_se_w_vt,
            "per_group_se_unw_vt": per_group_se_unw_vt,
            "per_group_sum_nobs_vt": per_group_sum_nobs_vt,
            "micro_nll_v": micro_nll_v, "micro_nll_vt": micro_nll_vt,
            "macro_nll_v": macro_nll_v, "macro_nll_vt": macro_nll_vt,
            "per_group_nll_v": per_group_nll_v,
            "per_group_nll_vt": per_group_nll_vt,
            "micro_elbo_v": micro_elbo_v, "micro_elbo_vt": micro_elbo_vt,
            "macro_elbo_v": macro_elbo_v, "macro_elbo_vt": macro_elbo_vt,
            "per_group_elbo_v": per_group_elbo_v,
            "per_group_elbo_vt": per_group_elbo_vt,
            "micro_cov_v": micro_cov_v, "micro_cov_vt": micro_cov_vt,
            "macro_cov_v": macro_cov_v, "macro_cov_vt": macro_cov_vt,
            "per_group_cov_v": per_group_cov_v,
            "per_group_cov_vt": per_group_cov_vt,
            "micro_width_v": micro_width_v, "micro_width_vt": micro_width_vt,
            "macro_width_v": macro_width_v, "macro_width_vt": macro_width_vt,
            "per_group_width_v": per_group_width_v,
            "per_group_width_vt": per_group_width_vt,

        }
    


    def denoise_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        timestamp_col: str,
        static_features_cols: list[str] | None = None,
        window_size: int | None = None,
        window_step: int = 1,
        steps: int | None = None,
        replace_only_missing: bool = True,
        device: torch.device | None = None,
    ) -> pd.DataFrame:
        """
        Retorna um novo DataFrame com as colunas feature_cols denoised.
        - Se window_size for None ou >= len(df), processa de uma vez.
        - Com janelas sobrepostas, agrega por média.
        - Por padrão só substitui NaN (replace_only_missing=True).
        """
        self.eval()
        device = device or next(self.parameters()).device
        static_features_cols = static_features_cols or []

        # Mantém índice original; caso ordene por tempo, voltaremos ao índice depois
        orig_index = df.index
        needs_sort = (timestamp_col != "index")
        if needs_sort:
            df_sorted = df.sort_values(timestamp_col).reset_index(drop=False)
            idx_col_name = df_sorted.columns[0]  # coluna do índice original após reset_index
        else:
            df_sorted = df.copy()

        # Monta dataset exatamente como no treino
        ds = self._make_dataset(
            df_sorted,
            timestamp_col=timestamp_col,
            window_size=window_size,
            feature_cols=feature_cols,
            static_features_cols=static_features_cols,
            window_step=window_step,
        )
        tensors = ds.tensors
        if len(tensors) == 4:
            seqs, ts_seqs, mask_seqs, stat_seqs = tensors
        else:
            seqs, ts_seqs, mask_seqs = tensors
            stat_seqs = None

        seqs = seqs.to(device)
        ts_seqs = ts_seqs.to(device)
        mask_seqs = mask_seqs.to(device)
        if stat_seqs is not None:
            stat_seqs = stat_seqs.to(device)

        with torch.no_grad():
            h0 = self.encoder(torch.cat([seqs, mask_seqs], dim=-1))
            if self.lam[1] == 0:
                # sem termo de ruído (L2), não há estrutura de denoise treinada:
                # usa apenas o reconstrutor/imputation de um passo (L1/L5).
                x_r = self.impute(
                    state=h0,
                    device=device,
                    timestamps=ts_seqs,
                    static_feats=stat_seqs,
                    mask=mask_seqs,
                    x0=seqs,
                    return_x_hat=True,
                )
                x_r = x_r.detach().cpu().numpy()
                x_hat = None
            else:
                n_steps = steps if steps is not None else getattr(self, "denoise_steps", self.num_steps)
                x_r, z = self.denoise(
                    state=h0,
                    timestamps=ts_seqs,
                    static_feats=stat_seqs,
                    device=device,
                    steps=n_steps,
                    x0=seqs,                     # <- importante
                    mask=mask_seqs,              # <- importante
                    enforce_data_consistency=False
                )
                x_hat = self.decoder(z).detach().cpu().numpy()
                x_r = x_r.detach().cpu().numpy()

        n = len(df_sorted)
        C = len(feature_cols)
        out = np.zeros((n, C), dtype=np.float32)
        out_r = np.zeros((n, C), dtype=np.float32)
        cnt = np.zeros((n, C), dtype=np.float32)
        cnt_r = np.zeros((n, C), dtype=np.float32)

        # matriz original e máscara de missing (True = faltante)
        orig_vals = df_sorted[feature_cols].to_numpy()
        miss = ~np.isfinite(orig_vals)

        if window_size is None or window_size >= n:
            pred_r = x_r[0]
            if x_hat is not None:
                pred = x_hat[0]
                if replace_only_missing:
                    out = np.where(miss, pred, orig_vals)
                    cnt = np.where(miss, 1.0, 0.0).astype(np.float32)
                else:
                    out = pred
                    cnt[:] = 1.0
            # reconstrução (sempre disponível)
            if replace_only_missing:
                out_r = np.where(miss, pred_r, orig_vals)
                cnt_r = np.where(miss, 1.0, 0.0).astype(np.float32)
            else:
                out_r = pred_r
                cnt_r[:] = 1.0
        else:
            starts = np.arange(0, n - window_size + 1, window_step, dtype=int)
            for k, s in enumerate(starts):
                e = min(s + window_size, n)
                pred_r = x_r[k, :e - s, :]
                if x_hat is not None:
                    pred = x_hat[k, :e - s, :]  # (L_k, C)
                    if replace_only_missing:
                        sel = miss[s:e, :]                  # só substitui onde falta
                        # escreve pred onde falta; mantém original onde não falta
                        blended = np.where(sel, pred, orig_vals[s:e, :])
                        out[s:e, :] += blended
                        cnt[s:e, :] += sel.astype(np.float32)
                    else:
                        out[s:e, :] += pred
                        cnt[s:e, :] += 1.0
                # sempre acumula reconstrução
                if replace_only_missing:
                    sel_r = miss[s:e, :]
                    blended_r = np.where(sel_r, pred_r, orig_vals[s:e, :])
                    out_r[s:e, :] += blended_r
                    cnt_r[s:e, :] += sel_r.astype(np.float32)
                else:
                    out_r[s:e, :] += pred_r
                    cnt_r[s:e, :] += 1.0

            # posições não cobertas por nenhuma janela ou nunca substituídas
            if x_hat is not None:
                no_write = (cnt == 0.0)
                out[no_write] = orig_vals[no_write]
                # média nas posições com múltiplas escritas
                written = (cnt > 0.0)
                out[written] = out[written] / cnt[written]

            no_write_r = (cnt_r == 0.0)
            out_r[no_write_r] = orig_vals[no_write_r]
            written_r = (cnt_r > 0.0)
            out_r[written_r] = out_r[written_r] / cnt_r[written_r]

        # monta DataFrame de saída
        result_r = df_sorted.copy()
        result_r[feature_cols] = out_r
        if x_hat is not None:
            result = df_sorted.copy()
            result[feature_cols] = out
        else:
            result = None
        # restaura ordem/índice original caso tenha ordenado por tempo
        if needs_sort:
            result_r = result_r.set_index(idx_col_name).loc[orig_index]
            result_r.index = orig_index  # garante exatamente o mesmo Index
            if result is not None:
                result = result.set_index(idx_col_name).loc[orig_index]
                result.index = orig_index  # garante exatamente o mesmo Index

        return result_r, result
    
    def generate_samples(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        timestamp_col: str,
        slice: slice,
        size: int,
        static_features_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Gera amostras passo a passo (autoregressivo):
        - Cada novo passo gerado é alimentado de volta como parte do contexto,
          mantendo o mesmo tamanho de janela definido por `slice`.
        - Todas as colunas em feature_cols são robust‑escaladas internamente
          e as amostras são devolvidas na escala robusta.
        """
        self.eval()
        static_features_cols = static_features_cols or []

        feature_cols = list(feature_cols)  # mantém duplicatas para casar com in_channels
        if len(feature_cols) != self.in_channels:
            raise ValueError(
                f"generate_samples: len(feature_cols)={len(feature_cols)} difere do modelo (in_channels={self.in_channels}). "
                "Use o mesmo conjunto de features usado no treino."
            )
        missing_features = [c for c in set(feature_cols) if c not in df.columns]
        if missing_features:
            raise ValueError(
                f"generate_samples: feature_cols ausentes no DataFrame: {missing_features}"
            )

        # RobustScaler apenas nas feature_cols (escala interna do modelo)
        df_scaled = df.copy()
        scaler = RobustScaler()
        df_scaled[feature_cols] = scaler.fit_transform(
            df_scaled[feature_cols].to_numpy(dtype=float)
        )

        # contexto inicial na escala robusta
        df_ctx = df_scaled.iloc[slice].copy()
        if df_ctx.empty:
            raise ValueError("generate_samples: slice vazio; selecione ao menos uma linha de contexto.")

        # passo temporal (índice ou timestamp)
        if timestamp_col == "index":
            idx_full = df_scaled.index
            if len(idx_full) >= 2:
                try:
                    step_idx = idx_full[1] - idx_full[0]
                except Exception:
                    step_idx = 1
            else:
                step_idx = 1
        else:
            ts_series = df_ctx[timestamp_col]
            if len(ts_series) >= 2:
                dt = ts_series.iloc[1] - ts_series.iloc[0]
            else:
                diffs = ts_series.diff().dropna()
                dt = diffs.median() if not diffs.empty else 1

        generated_indices: list = []
        generated_rows: list[np.ndarray] = []

        for k in range(size):
            # índice / timestamp do próximo passo
            if timestamp_col == "index":
                next_idx = df_ctx.index[-1] + step_idx
                new_row = pd.DataFrame(index=[next_idx], columns=df_scaled.columns, dtype=float)
            else:
                next_ts = df_ctx[timestamp_col].iloc[-1] + dt
                new_row = pd.DataFrame(columns=df_scaled.columns, index=[len(df_scaled) + k], dtype=float)
                new_row[timestamp_col] = next_ts

            # copiar features estáticas da última linha do contexto
            if static_features_cols:
                last_static = df_ctx.iloc[-1][static_features_cols]
                for col in static_features_cols:
                    new_row[col] = last_static[col]

            # colunas dinâmicas: todas as feature_cols = NaN; demais mantêm último valor
            last_row = df_ctx.iloc[-1]
            for col in df_scaled.columns:
                if col in feature_cols:
                    new_row[col] = np.nan
                elif col not in static_features_cols and col != timestamp_col:
                    new_row[col] = last_row[col]

            # contexto + 1 futuro
            df_ext = pd.concat([df_ctx, new_row], axis=0)

            # imputação apenas onde há NaN, usando reconstrutor (result_r)
            result_r, _ = self.denoise_dataframe(
                df_ext,
                feature_cols=feature_cols,
                timestamp_col=timestamp_col,
                static_features_cols=static_features_cols,
                window_size=None,
                window_step=1,
                steps=None,
                replace_only_missing=True,
            )

            pred_row = result_r.iloc[-1]

            # armazena predição (ainda na escala robusta)
            if timestamp_col == "index":
                generated_indices.append(pred_row.name)
            else:
                generated_indices.append(pred_row[timestamp_col])
            generated_rows.append(pred_row[feature_cols].to_numpy(dtype=float))

            # atualiza contexto: descarta primeira linha e adiciona a linha preenchida
            new_row_filled = new_row.copy()
            pred_vals = pred_row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
            new_row_filled[feature_cols] = pred_vals
            df_ctx = pd.concat([df_ctx.iloc[1:], new_row_filled], axis=0)

        data = np.vstack(generated_rows) if generated_rows else np.empty((0, len(feature_cols)))
        if timestamp_col == "index":
            out_df = pd.DataFrame(data, index=generated_indices, columns=feature_cols)
        else:
            out_df = pd.DataFrame(data, columns=feature_cols)
            out_df.insert(0, timestamp_col, generated_indices)

        return out_df

    def extract_percentile_pdf_tmax(
            self,
            df: pd.DataFrame,
            feature_cols: list[str],
            timestamp_col: str,
            predict_state_cols: list[str],
            window_size: int,
            window_step: int,
            static_features_cols: list[str] | None = None,
            percentile: float = 0.5,
    ):
        """
        Calcula o percentil teórico de tmax (normal aproximada) por janela e devolve um novo
        DataFrame com o valor inferido em cada timestamp da janela. Valores fora do
        intervalo (0,1) são zerados.
        """
        if self.status_dim <= 0 or (self.lam[2] <= 0 and self.lam[5] <= 0):
            raise RuntimeError("O modelo não possui cabeça de tmax/variância para gerar PDFs.")
        if predict_state_cols is None or len(predict_state_cols) == 0:
            raise ValueError("predict_state_cols é obrigatório para calcular a janela de previsão.")

        static_features_cols = static_features_cols or []
        device = next(self.parameters()).device
        self.eval()

        needs_sort = (timestamp_col != "index")
        if needs_sort:
            df_sorted = df.sort_values(timestamp_col).reset_index(drop=False)
            idx_col_name = df_sorted.columns[0]  # índice original preservado
        else:
            df_sorted = df.copy()
            idx_col_name = df_sorted.index.name
        n = len(df_sorted)
        if window_size is None or window_size >= n:
            starts = np.array([0], dtype=int)
            end_positions = np.array([n - 1], dtype=int)
        else:
            starts = np.arange(0, n - window_size + 1, window_step, dtype=int)
            end_positions = starts + window_size - 1

        ds = self._make_dataset(
            df_sorted,
            timestamp_col=timestamp_col,
            window_size=window_size,
            feature_cols=feature_cols,
            static_features_cols=static_features_cols,
            window_step=window_step,
            predict_state_cols=predict_state_cols,
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False, pin_memory=True)

        percentile = float(np.clip(percentile, 1e-4, 1 - 1e-4))
        values = np.zeros((n, self.status_dim), dtype=np.float32)
        counts = np.zeros((n, self.status_dim), dtype=np.float32)
        pos = 0
        ts_values = df_sorted.index if timestamp_col == "index" else df_sorted[timestamp_col]

        with torch.no_grad():
            for batch in loader:
                x, ts_batch, m = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                idx = 3
                s = None
                if self.static_dim > 0:
                    s = batch[idx].to(device)
                    idx += 1
                    if self.cost_columns is not None:
                        idx += 1  # pula custos; não usados aqui
                else:
                    if self.cost_columns is not None:
                        idx += 1  # pula custos; não usados aqui

                mask_ts = m.any(dim=2, keepdim=True).float()
                out = self.forward(
                    x * m,
                    timestamps=ts_batch,
                    static_feats=s,
                    return_x_hat=False,
                    mask=m,
                    mask_ts=mask_ts,
                    test=True,
                )
                if len(out) == 16:
                    (
                        x,
                        state,
                        _,
                        state_tmax,
                        _,
                        tmax_hat,
                        _,
                        _,
                        _,
                        _,
                        _,
                        vae_tmax,
                        vae_tmax_mu,
                        vae_tmax_logvar,
                        vae_logvar_obs,
                        vae_tmax_logvar_obs,
                    ) = out
                else:
                    (
                        state,
                        _,
                        state_tmax,
                        _,
                        tmax_hat,
                        _,
                        _,
                        _,
                        _,
                        _,
                        vae_tmax,
                        vae_tmax_mu,
                        vae_tmax_logvar,
                        vae_logvar_obs,
                        vae_tmax_logvar_obs,
                    ) = out

                # Média/variância em fração da janela (0-1) – usa VAE se disponível (média determinística em vae_mu)
                if self.lam[5] > 0 and vae_tmax_mu is not None:
                    mean_frac = self.vae_tmax_decoder(vae_tmax_mu)
                    logvar_obs = self.vae_tmax_sigma_head(vae_tmax_mu)
                    logvar_obs = (logvar_obs + math.log(self.sigma_temp)).clamp(min=-5.0, max=5.0)
                    var_frac = torch.exp(logvar_obs)
                elif self.status_dim > 0 and self.log_likelihood and hasattr(self, "lambda_tmax_head") and state_tmax is not None:
                    lam_t_tmax = F.softplus(self.lambda_tmax_head(state_tmax)).clamp(min=1 / (2 * math.pi), max=2 * math.pi)
                    var_frac = (1.0 / lam_t_tmax).clamp(min=1e-8)
                    mean_frac = (tmax_hat if tmax_hat is not None else torch.zeros_like(lam_t_tmax)).clamp(0.0, 1.0)
                else:
                    mean_frac = (tmax_hat if tmax_hat is not None else torch.zeros_like(m[..., :1])).clamp(0.0, 1.0)
                    var_frac = torch.full_like(mean_frac, 0.05)

                # usa apenas o último passo da janela (onde tmax é avaliado)
                mean_last = mean_frac[:, -1, :]
                var_last = var_frac[:, -1, :]
                std_last = torch.sqrt(var_last + 1e-8)

                # quantil da normal (não truncada); se cair fora de (0,1), zera
                q = torch.tensor(percentile, device=device, dtype=mean_last.dtype)
                z_q = math.sqrt(2.0) * torch.special.erfinv(2 * q - 1)
                quantile_frac = mean_last + std_last * z_q
                quantile_frac = torch.where((quantile_frac > 0) & (quantile_frac < 1), quantile_frac, torch.zeros_like(quantile_frac))
                quantile_np = quantile_frac.cpu().numpy()

                bsz = x.size(0)
                for b in range(bsz):
                    anchor_idx = int(min(end_positions[pos + b], n - 1))
                    s_idx_start = int(starts[pos + b])
                    for s_idx in range(quantile_np.shape[1]):
                        val = float(quantile_np[b, s_idx])
                        values[s_idx_start : anchor_idx + 1, s_idx] += val
                        counts[s_idx_start : anchor_idx + 1, s_idx] += 1.0
                pos += bsz

        # agrega e escreve no DataFrame
        result = df_sorted.copy()
        state_cols = (
            predict_state_cols
            if isinstance(predict_state_cols, (list, tuple))
            else [predict_state_cols]
        )
        if len(state_cols) != self.status_dim:
            state_cols = [f"tmax_state_{i}" for i in range(self.status_dim)]

        for i, col in enumerate(state_cols):
            col_name = f"{col}_pct{int(percentile * 100)}"
            c = counts[:, i]
            v = values[:, i]
            filled = np.where(c > 0, v / c, 0.0)
            result[col_name] = filled

        if needs_sort:
            result = result.set_index(idx_col_name).loc[df.index]
            result.index = df.index
        else:
            result.index = df.index

        return result

    def generate_pdf_tmax(
            self,
            df: pd.DataFrame,
            feature_cols: list[str],
            timestamp_col: str,
            predict_state_cols: list[str],
            slice: slice,
            static_features_cols: list[str] | None = None,
            status_pred_window: int = 300,
            grid_points: int = 200,
    ) -> pd.DataFrame:
        """
        Extrai a PDF teórica (normal aproximada) de tmax para o ponto definido pelo slice
        fornecido (mesmo estilo de generate_samples). Retorna um DataFrame com μ/σ (em segundos)
        e a curva de densidade em uma grade de tempo.
        """
        if self.status_dim <= 0 or (self.lam[2] <= 0 and self.lam[5] <= 0):
            raise RuntimeError("O modelo não possui cabeça de tmax/variância para gerar PDFs.")
        if predict_state_cols is None or len(predict_state_cols) == 0:
            raise ValueError("predict_state_cols é obrigatório para calcular a janela de previsão.")
        if grid_points < 2:
            raise ValueError("grid_points deve ser >= 2 para construir a PDF.")

        static_features_cols = static_features_cols or []
        device = next(self.parameters()).device
        self.eval()

        needs_sort = (timestamp_col != "index")
        if needs_sort:
            df_sorted = df.sort_values(timestamp_col).reset_index(drop=False)
        else:
            df_sorted = df.copy()

        df_window = df_sorted.iloc[slice].copy()
        if df_window.empty:
            raise ValueError("slice vazio; selecione ao menos uma linha para avaliar a PDF.")

        # dataset de uma única janela (toda a fatia)
        ds = self._make_dataset(
            df_window,
            timestamp_col=timestamp_col,
            window_size=None,
            feature_cols=feature_cols,
            static_features_cols=static_features_cols,
            window_step=1,
            predict_state_cols=predict_state_cols,
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False, pin_memory=True)

        grid_seconds = torch.linspace(0.0, float(status_pred_window), steps=grid_points, device=device)
        grid_fraction = grid_seconds / max(float(status_pred_window), 1e-6)
        rows = []

        with torch.no_grad():
            for batch in loader:
                x, ts_batch, m = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                idx = 3
                s = None
                if self.static_dim > 0:
                    s = batch[idx].to(device)
                    idx += 1
                    if self.cost_columns is not None:
                        idx += 1  # pula custos; não usados aqui
                else:
                    if self.cost_columns is not None:
                        idx += 1  # pula custos; não usados aqui

                mask_ts = m.any(dim=2, keepdim=True).float()
                out = self.forward(
                    x * m,
                    timestamps=ts_batch,
                    static_feats=s,
                    return_x_hat=False,
                    mask=m,
                    mask_ts=mask_ts,
                    test=True,
                )
                if len(out) == 16:
                    (
                        x,
                        state,
                        _,
                        state_tmax,
                        _,
                        tmax_hat,
                        _,
                        _,
                        _,
                        _,
                        _,
                        vae_tmax,
                        vae_tmax_mu,
                        vae_tmax_logvar,
                        vae_logvar_obs,
                        vae_tmax_logvar_obs,
                    ) = out
                else:
                    (
                        state,
                        _,
                        state_tmax,
                        _,
                        tmax_hat,
                        _,
                        _,
                        _,
                        _,
                        _,
                        vae_tmax,
                        vae_tmax_mu,
                        vae_tmax_logvar,
                        vae_logvar_obs,
                        vae_tmax_logvar_obs,
                    ) = out

                # parâmetros da normal em fração da janela (0-1)
                if self.lam[5] > 0 and vae_tmax_mu is not None:
                    mean_frac = self.vae_tmax_decoder(vae_tmax_mu)
                    logvar_obs = self.vae_tmax_sigma_head(vae_tmax_mu)
                    logvar_obs = (logvar_obs + math.log(self.sigma_temp)).clamp(min=-5.0, max=5.0)
                    var_frac = torch.exp(logvar_obs)
                elif self.status_dim > 0 and self.log_likelihood and hasattr(self, "lambda_tmax_head") and state_tmax is not None:
                    lam_t_tmax = F.softplus(self.lambda_tmax_head(state_tmax)).clamp(min=1 / (2 * math.pi), max=2 * math.pi)
                    var_frac = (1.0 / lam_t_tmax).clamp(min=1e-8)
                    mean_frac = tmax_hat if tmax_hat is not None else torch.zeros_like(lam_t_tmax)
                else:
                    mean_frac = tmax_hat if tmax_hat is not None else torch.zeros_like(m[..., :1])
                    var_frac = torch.full_like(mean_frac, 0.05)

                mean_last = mean_frac[:, -1, :]  # (1,S)
                var_last = var_frac[:, -1, :]
                std_last = torch.sqrt(var_last + 1e-8)

                mean_sec = (mean_last * status_pred_window).cpu().numpy()
                std_sec = (std_last * status_pred_window).cpu().numpy()

                mean_b = mean_last.unsqueeze(1)
                std_b = std_last.unsqueeze(1).clamp(min=1e-6)
                pdf_norm = (1.0 / (std_b * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((grid_fraction.view(1, -1, 1) - mean_b) / std_b) ** 2)
                pdf_sec = (pdf_norm / max(float(status_pred_window), 1e-6)).cpu().numpy()

                ts_anchor = (
                    df_window.index[-1] if timestamp_col == "index" else df_window[timestamp_col].iloc[-1]
                )
                for s_idx in range(mean_sec.shape[1]):
                    rows.append(
                        {
                            timestamp_col: ts_anchor,
                            "state_dim": int(s_idx),
                            "mu_seconds": float(mean_sec[0, s_idx]),
                            "sigma_seconds": float(std_sec[0, s_idx]),
                            "t_grid_seconds": grid_seconds.cpu().numpy().tolist(),
                            "pdf": pdf_sec[0, :, s_idx].tolist(),
                        }
                    )

        return pd.DataFrame(rows)
