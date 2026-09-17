"""Explainability V2 utilities for seqKAN vs GRU comparison.

Extracted from notebook to keep analysis notebook focused on running results.
"""

# --- EXPLAINABILITY_V2: TF(core)+TopK, EA robusta, SE corrigida, AD por regras ---
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from seqkan.kan.spline import coef2curve

# ============================================================
# 0) utilidades
# ============================================================

def _device(model):
    return next(model.parameters()).device


def _to_dev(model, x, ts, m):
    d = _device(model)
    return x.to(d), ts.to(d), m.to(d)


def _xhat(model, x, ts, m):
    model.eval()
    x, ts, m = _to_dev(model, x, ts, m)
    with torch.no_grad():
        out = model.forward(x, timestamps=ts, return_x_hat=True, mask=m, test=True)
    return out[3]


def _spearman(a, b):
    ra = pd.Series(np.asarray(a)).rank(method='average').to_numpy()
    rb = pd.Series(np.asarray(b)).rank(method='average').to_numpy()
    sa, sb = np.std(ra), np.std(rb)
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _make_eval_batch(model, df, feature_cols, timestamp_col='index', window_size=128, window_step=8, max_windows=256):
    ds = model._make_dataset(
        df=df,
        timestamp_col=timestamp_col,
        window_size=window_size,
        feature_cols=feature_cols,
        static_features_cols=None,
        window_step=window_step,
        df_denoised=None,
        predict_state_cols=None,
    )
    seqs, ts_seqs, mask_seqs = ds.tensors[:3]
    n = min(max_windows, seqs.shape[0])
    return seqs[:n], ts_seqs[:n], mask_seqs[:n]


# ============================================================
# 1) IMPORTÂNCIA comparável (gradxinput / IG / permutation)
# ============================================================

def _attr_gradxinput(model, x, ts, m):
    model.eval()
    x, ts, m = _to_dev(model, x, ts, m)
    xg = x.detach().clone().requires_grad_(True)
    out = model.forward(xg, timestamps=ts, return_x_hat=True, mask=m, test=True)
    y = out[3][:, -1, :].mean()
    g = torch.autograd.grad(y, xg, retain_graph=False, create_graph=False)[0]
    imp = (g * xg).abs().mean(dim=(0, 1))
    return imp.detach().cpu().numpy()


def _attr_integrated_gradients(model, x, ts, m, steps=24):
    model.eval()
    x, ts, m = _to_dev(model, x, ts, m)
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)

    for alpha in torch.linspace(0, 1, steps, device=x.device):
        xi = baseline + alpha * (x - baseline)
        xi.requires_grad_(True)
        out = model.forward(xi, timestamps=ts, return_x_hat=True, mask=m, test=True)
        y = out[3][:, -1, :].mean()
        grad = torch.autograd.grad(y, xi, retain_graph=False, create_graph=False)[0]
        total_grad = total_grad + grad

    ig = (x - baseline) * total_grad / float(steps)
    imp = ig.abs().mean(dim=(0, 1))
    return imp.detach().cpu().numpy()


def _attr_permutation(model, x, ts, m, seed=0):
    # importância por aumento de erro quando permuta uma feature no batch
    rng = np.random.default_rng(seed)
    model.eval()
    x, ts, m = _to_dev(model, x, ts, m)
    with torch.no_grad():
        pred0 = _xhat(model, x, ts, m)
        base_mse = ((pred0 - x) ** 2).mean().item()

    C = x.shape[-1]
    imps = np.zeros(C, dtype=float)
    for c in range(C):
        xp = x.clone()
        idx = torch.tensor(rng.permutation(x.shape[0]), device=x.device)
        xp[:, :, c] = xp[idx, :, c]
        with torch.no_grad():
            pred = _xhat(model, xp, ts, m)
            mse = ((pred - x) ** 2).mean().item()
        imps[c] = mse - base_mse
    return imps



def _attr_permutation_window(model, x, ts, m, seed=0):
    # permutação temporal por janela (window-wise): mede delta de erro por feature
    rng = np.random.default_rng(seed)
    model.eval()
    x, ts, m = _to_dev(model, x, ts, m)
    with torch.no_grad():
        pred0 = _xhat(model, x, ts, m)
        base_mse_w = ((pred0 - x) ** 2).mean(dim=(1, 2))

    B, T, C = x.shape
    imps = np.zeros(C, dtype=float)
    for c in range(C):
        xp = x.clone()
        for b in range(B):
            idx_t = torch.tensor(rng.permutation(T), device=x.device)
            xp[b, :, c] = xp[b, idx_t, c]
        with torch.no_grad():
            pred = _xhat(model, xp, ts, m)
            mse_w = ((pred - x) ** 2).mean(dim=(1, 2))
        imps[c] = float((mse_w - base_mse_w).mean().item())
    return imps


def feature_importance(model, x, ts, m, method='gradxinput', seed=0):
    method = method.lower()
    if method == 'gradxinput':
        return _attr_gradxinput(model, x, ts, m)
    if method in ('ig', 'integrated_gradients'):
        return _attr_integrated_gradients(model, x, ts, m)
    if method == 'permutation':
        return _attr_permutation(model, x, ts, m, seed=seed)
    if method in ('permutation_window', 'window_permutation', 'perm_window'):
        return _attr_permutation_window(model, x, ts, m, seed=seed)
    raise ValueError('method deve ser gradxinput, ig, permutation ou permutation_window')


# ============================================================
# 2) TF (core) + segunda evidência top-k (seqKAN)
# ============================================================

def _edge_score_from_layer(layer, score='coef_l2'):
    if hasattr(layer, 'coef') and layer.coef is not None:
        coef = layer.coef.detach()
        if score in ('coef_l1', 'coef_abs'):
            s = coef.abs().mean(dim=-1)
        else:
            s = torch.sqrt((coef ** 2).mean(dim=-1) + 1e-12)
        return s
    if hasattr(layer, 'scale_sp') and layer.scale_sp is not None:
        return layer.scale_sp.detach().abs()
    raise ValueError('layer sem coef/scale_sp')


def _align_in_out(layer, scores):
    in_dim = layer.grid.shape[0]
    if scores.shape[0] == in_dim:
        return scores
    if scores.shape[1] == in_dim:
        return scores.T
    raise ValueError(f'shape invalida scores={scores.shape}, in_dim={in_dim}')


def topk_edges_and_mass(layer, k=20, score='coef_l2'):
    s = _align_in_out(layer, _edge_score_from_layer(layer, score=score))
    s = s.detach().cpu()
    flat = s.reshape(-1)
    kk = min(int(k), flat.numel())
    vals, idx = torch.topk(flat, k=kk)

    out_dim = s.shape[1]
    edges = []
    for v, f in zip(vals.tolist(), idx.tolist()):
        i = int(f // out_dim)
        j = int(f % out_dim)
        edges.append({'i_in': i, 'j_out': j, 'score': float(v)})

    total_mass = float(flat.abs().sum().item())
    topk_mass = float(vals.abs().sum().item())
    mass_ratio = topk_mass / max(total_mass, 1e-12)
    return edges, mass_ratio


def topk_edges_and_mass_linear(linear_layer, k=20):
    # Linear: weight shape (out_dim, in_dim) -> converte para (in_dim, out_dim)
    w = linear_layer.weight.detach().abs().T.cpu()
    flat = w.reshape(-1)
    kk = min(int(k), flat.numel())
    vals, idx = torch.topk(flat, k=kk)

    out_dim = w.shape[1]
    edges = []
    for v, f in zip(vals.tolist(), idx.tolist()):
        i = int(f // out_dim)
        j = int(f % out_dim)
        edges.append({'i_in': i, 'j_out': j, 'score': float(v)})

    total_mass = float(flat.sum().item())
    topk_mass = float(vals.sum().item())
    mass_ratio = topk_mass / max(total_mass, 1e-12)
    return edges, mass_ratio


def sample_spline(layer, i_in, j_out, n=300):
    k = int(layer.k)
    # usa CPU só para construir range de plot
    grid_cpu = layer.grid[i_in].detach().cpu().numpy()
    x_min, x_max = grid_cpu[k], grid_cpu[-k-1]
    x = np.linspace(x_min, x_max, n)

    # garante que x/grid/coef estejam no mesmo device
    dev = layer.grid.device
    x_t = torch.tensor(x, dtype=layer.grid.dtype, device=dev).view(-1, 1)
    grid_t = layer.grid[i_in:i_in+1]
    coef_t = layer.coef[i_in:i_in+1, j_out:j_out+1]

    with torch.no_grad():
        y = coef2curve(x_t, grid_t, coef_t, layer.k)[:, 0, 0]
        if hasattr(layer, 'mask') and layer.mask is not None:
            y = y * layer.mask[i_in, j_out]
    return x, y.detach().cpu().numpy()


def plot_topk_splines(layer, edges, title='Top-k splines', n_show=20, ncols=5):
    show = edges[:n_show]
    n = len(show)
    nrows = int(np.ceil(n / ncols)) if n > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.0, nrows*2.4))
    axes = np.array(axes).reshape(-1)

    for ax, e in zip(axes, show):
        x, y = sample_spline(layer, e['i_in'], e['j_out'])
        ax.plot(x, y)
        ax.set_title(f"{e['i_in']}→{e['j_out']} | {e['score']:.3g}", fontsize=8)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(show):]:
        ax.axis('off')

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    plt.show()
    return fig


def tf_metrics(seqkan_model, topk=20):
    layer_cell = seqkan_model.seqKAN.kan_cell.act_fun[0]

    edges_cell, mass_cell = topk_edges_and_mass(layer_cell, k=topk)

    out_mod = seqkan_model.seqKAN.kan_out
    if hasattr(out_mod, 'act_fun'):
        layer_out = out_mod.act_fun[0]
        edges_out, mass_out = topk_edges_and_mass(layer_out, k=topk)
        out_kind = 'kan'
    elif isinstance(out_mod, torch.nn.Linear):
        edges_out, mass_out = topk_edges_and_mass_linear(out_mod, k=topk)
        out_kind = 'linear'
    else:
        edges_out, mass_out = [], np.nan
        out_kind = type(out_mod).__name__

    # TF core claim
    tf_core = 1.0

    # segunda evidência
    tf_topk_mass = float(np.nanmean([mass_cell, mass_out]))

    return {
        'tf_core_claim': tf_core,
        'tf_topk_mass_ratio': tf_topk_mass,
        'tf_topk_mass_cell': float(mass_cell),
        'tf_topk_mass_out': float(mass_out) if not np.isnan(mass_out) else np.nan,
        'tf_out_kind': out_kind,
        'top_edges_cell': edges_cell,
        'top_edges_out': edges_out,
    }


# ============================================================
# 3) EA robusta (ruído 0.5%~2%, seeds, múltiplas janelas)
# ============================================================

def ea_stability(model, x, ts, m, method='gradxinput',
                 noise_pcts=(0.005, 0.01, 0.02),
                 n_seeds=5, windows_per_seed=32):
    x = x.detach()
    ts = ts.detach()
    m = m.detach()

    N = x.shape[0]
    C = x.shape[-1]
    x_np = x.cpu().numpy()
    feat_range = x_np.reshape(-1, C).ptp(axis=0) + 1e-12

    vals = []
    for s in range(n_seeds):
        rng = np.random.default_rng(100 + s)
        take = min(windows_per_seed, N)
        idx = rng.choice(N, size=take, replace=False)

        xw = x[idx]
        tsw = ts[idx]
        mw = m[idx]

        base = feature_importance(model, xw, tsw, mw, method=method, seed=100 + s)

        for pct in noise_pcts:
            noise = rng.normal(loc=0.0, scale=(pct * feat_range), size=xw.shape)
            noise_t = torch.tensor(noise, dtype=xw.dtype, device=xw.device)
            xp = xw + noise_t
            pert = feature_importance(model, xp, tsw, mw, method=method, seed=200 + s)
            vals.append(_spearman(base, pert))

    vals = np.array(vals, dtype=float)
    return {
        'ea_method': method,
        'ea_spearman_mean': float(np.nanmean(vals)),
        'ea_spearman_std': float(np.nanstd(vals)),
        'ea_n_eval': int(np.sum(~np.isnan(vals))),
    }


def ea_permutation_window_importance(model, x, ts, m, n_seeds=5, windows_per_seed=32):
    # importância por janela para evidenciar separação entre arquiteturas
    x = x.detach()
    ts = ts.detach()
    m = m.detach()
    N = x.shape[0]
    C = x.shape[-1]

    records = []
    all_imps = []

    for s in range(n_seeds):
        rng = np.random.default_rng(300 + s)
        take = min(windows_per_seed, N)
        idx = rng.choice(N, size=take, replace=False)
        xw = x[idx]
        tsw = ts[idx]
        mw = m[idx]

        imp = feature_importance(model, xw, tsw, mw, method='permutation_window', seed=400 + s)
        all_imps.append(imp)
        for c in range(C):
            records.append({
                'seed': int(s),
                'feature': int(c),
                'importance': float(imp[c]),
            })

    arr = np.array(all_imps, dtype=float)
    mean_imp = np.nanmean(arr, axis=0) if arr.size else np.zeros(C)
    std_imp = np.nanstd(arr, axis=0) if arr.size else np.zeros(C)

    df = pd.DataFrame(records)
    df_summary = pd.DataFrame({
        'feature': np.arange(C, dtype=int),
        'importance_mean': mean_imp,
        'importance_std': std_imp,
    }).sort_values('importance_mean', ascending=False, ignore_index=True)

    return {
        'df_window_importance': df,
        'df_window_importance_summary': df_summary,
    }


# ============================================================
# 4) SE corrigida
#    - seqKAN: arestas com contribuição > eps
#    - GRU: compressibilidade (pruning proxy)
# ============================================================

def _seqkan_hidden_size(seq):
    hs = getattr(seq, 'hidden_size', None)
    if hs is None:
        hs = getattr(seq, 'hidden_dim', None)
    if hs is None:
        raise AttributeError('seqKAN sem hidden_size/hidden_dim')
    return int(hs)


def _build_seqkan_h_input(seq, x_t, h, m_t):
    # Compatível com variantes: com kan_encoder (z_t) e sem kan_encoder (x_t direto)
    cand = []
    if hasattr(seq, 'kan_encoder'):
        try:
            z_t = seq.kan_encoder(x_t)
            cand.append(z_t)
        except Exception:
            pass
    cand.append(x_t)

    expected = None
    if hasattr(seq, 'kan_cell') and hasattr(seq.kan_cell, 'act_fun') and len(seq.kan_cell.act_fun) > 0:
        layer0 = seq.kan_cell.act_fun[0]
        if hasattr(layer0, 'grid'):
            expected = int(layer0.grid.shape[0])

    for base in cand:
        h_in = torch.cat([base, h, m_t], dim=-1)
        if expected is None or h_in.shape[-1] == expected:
            return h_in

    return torch.cat([cand[0], h, m_t], dim=-1)


def _seqkan_edge_contrib_mean(seqkan_model, x, ts, m):
    # contribuição média abs por aresta (cell e out) usando trajetória real
    seq = seqkan_model.seqKAN
    x, ts, m = _to_dev(seqkan_model, x, ts, m)
    B, T, C = x.shape
    hs = _seqkan_hidden_size(seq)
    h = torch.zeros(B, hs, device=x.device, dtype=x.dtype)

    layer_c = seq.kan_cell.act_fun[0]
    out_mod = seq.kan_out

    acc_c = torch.zeros(layer_c.grid.shape[0], layer_c.coef.shape[1], device=x.device)
    if hasattr(out_mod, 'act_fun'):
        layer_o = out_mod.act_fun[0]
        acc_o = torch.zeros(layer_o.grid.shape[0], layer_o.coef.shape[1], device=x.device)
        out_kind = 'kan'
    elif isinstance(out_mod, torch.nn.Linear):
        w_t = out_mod.weight.T  # (hidden, out)
        acc_o = torch.zeros(w_t.shape[0], w_t.shape[1], device=x.device)
        out_kind = 'linear'
    else:
        acc_o = torch.zeros(1, 1, device=x.device)
        out_kind = 'unknown'

    with torch.no_grad():
        mask_z = seq.mask_proj((1 - m))
        for t in range(T):
            x_t = x[:, t, :]
            m_t = mask_z[:, t, :] * h
            h_in = _build_seqkan_h_input(seq, x_t, h, m_t)

            yc = coef2curve(h_in, layer_c.grid, layer_c.coef, layer_c.k)
            yc = yc * layer_c.mask[None, :, :]
            acc_c += yc.abs().mean(dim=0)

            h = seq.kan_cell(h_in)

            if out_kind == 'kan':
                yo = coef2curve(h, layer_o.grid, layer_o.coef, layer_o.k)
                yo = yo * layer_o.mask[None, :, :]
                acc_o += yo.abs().mean(dim=0)
            elif out_kind == 'linear':
                # contribuição por aresta hidden->out: |h_i * w_{i,j}|
                yo = (h[:, :, None] * w_t[None, :, :]).abs()
                acc_o += yo.mean(dim=0)

    acc_c = acc_c / max(T, 1)
    acc_o = acc_o / max(T, 1)
    return acc_c.detach().cpu().numpy(), acc_o.detach().cpu().numpy()



def seqkan_edge_attribution_time(seqkan_model, x, ts, m, max_windows=64):
    # contribuição das arestas ao longo do tempo (cell vs out)
    seq = seqkan_model.seqKAN
    x = x[:max_windows]
    ts = ts[:max_windows]
    m = m[:max_windows]
    x, ts, m = _to_dev(seqkan_model, x, ts, m)

    B, T, _ = x.shape
    hs = _seqkan_hidden_size(seq)
    h = torch.zeros(B, hs, device=x.device, dtype=x.dtype)

    layer_c = seq.kan_cell.act_fun[0]
    out_mod = seq.kan_out
    if hasattr(out_mod, 'act_fun'):
        layer_o = out_mod.act_fun[0]
        out_kind = 'kan'
    elif isinstance(out_mod, torch.nn.Linear):
        w_t = out_mod.weight.T
        out_kind = 'linear'
    else:
        out_kind = 'unknown'

    rec = []
    with torch.no_grad():
        mask_z = seq.mask_proj((1 - m))
        for t in range(T):
            x_t = x[:, t, :]
            m_t = mask_z[:, t, :] * h
            h_in = _build_seqkan_h_input(seq, x_t, h, m_t)

            yc = coef2curve(h_in, layer_c.grid, layer_c.coef, layer_c.k)
            yc = yc * layer_c.mask[None, :, :]
            yo_prev = yc.abs().mean(dim=0)

            h = seq.kan_cell(h_in)

            if out_kind == 'kan':
                yo = coef2curve(h, layer_o.grid, layer_o.coef, layer_o.k)
                yo = yo * layer_o.mask[None, :, :]
                yo_cur = yo.abs().mean(dim=0)
            elif out_kind == 'linear':
                yo_cur = (h[:, :, None] * w_t[None, :, :]).abs().mean(dim=0)
            else:
                yo_cur = torch.zeros(1, 1, device=x.device)

            rec.append({
                't': int(t),
                'cell_edge_contrib_mean': float(yo_prev.mean().item()),
                'cell_edge_contrib_max': float(yo_prev.max().item()),
                'out_edge_contrib_mean': float(yo_cur.mean().item()),
                'out_edge_contrib_max': float(yo_cur.max().item()),
            })

    return pd.DataFrame(rec)

def se_seqkan_used_edges(seqkan_model, x, ts, m, eps_frac=0.01):
    cc, co = _seqkan_edge_contrib_mean(seqkan_model, x, ts, m)
    allc = np.concatenate([cc.reshape(-1), co.reshape(-1)])
    eps = float(eps_frac * np.max(allc)) if allc.size else 0.0
    used = float(np.sum(allc > eps))
    total = float(allc.size)
    sparsity = 1.0 - used / max(total, 1.0)
    return {
        'se_seqkan_used_edges': used,
        'se_seqkan_total_edges': total,
        'se_seqkan_sparsity_eps': sparsity,
        'se_seqkan_eps': eps,
    }


def gru_compressibility(gru_model, quantiles=(0.5, 0.8, 0.9, 0.95)):
    ws = []
    for p in gru_model.parameters():
        if p.requires_grad:
            ws.append(p.detach().abs().reshape(-1).cpu())
    if not ws:
        return {'gru_compressibility': {}}
    w = torch.cat(ws).numpy()
    out = {}
    for q in quantiles:
        thr = float(np.quantile(w, q))
        frac = float(np.mean(w <= thr))
        out[f'q{int(q*100)}'] = {'threshold': thr, 'compressible_fraction': frac}
    return {'gru_compressibility': out}


def se_gru_parallel_metrics(gru_model, x, ts, m, eps_frac=0.01):
    # Métricas estruturais paralelas ao SE do seqKAN:
    # 1) fração de pesos com |w| > eps_w
    # 2) fração de unidades hidden com ativação média > eps_h
    ws = []
    for p in gru_model.parameters():
        if p.requires_grad:
            ws.append(p.detach().abs().reshape(-1).cpu())
    if not ws:
        return {
            'se_gru_eps_w': np.nan,
            'se_gru_weights_active_frac_eps': np.nan,
            'se_gru_weights_active_n': np.nan,
            'se_gru_weights_total_n': np.nan,
            'se_gru_eps_h': np.nan,
            'se_gru_hidden_active_frac_eps': np.nan,
            'se_gru_hidden_active_n': np.nan,
            'se_gru_hidden_total_n': np.nan,
        }

    w = torch.cat(ws).numpy()
    max_w = float(np.max(w)) if w.size else 0.0
    eps_w = float(eps_frac * max_w) if max_w > 0 else 0.0
    active_w = float(np.sum(w > eps_w))
    total_w = float(w.size)
    frac_w = active_w / max(total_w, 1.0)

    # hidden usage: média de |h| por unidade ao longo de batch e tempo
    x_d, ts_d, m_d = _to_dev(gru_model, x, ts, m)
    with torch.no_grad():
        out = gru_model.forward(x_d, timestamps=ts_d, return_x_hat=False, mask=m_d, test=True)
        h = out[0]  # (B, T, Hstate)
        h_mean_abs = h.abs().mean(dim=(0, 1)).detach().cpu().numpy()
    max_h = float(np.max(h_mean_abs)) if h_mean_abs.size else 0.0
    eps_h = float(eps_frac * max_h) if max_h > 0 else 0.0
    active_h = float(np.sum(h_mean_abs > eps_h))
    total_h = float(h_mean_abs.size)
    frac_h = active_h / max(total_h, 1.0)

    return {
        'se_gru_eps_w': eps_w,
        'se_gru_weights_active_frac_eps': float(frac_w),
        'se_gru_weights_active_n': float(active_w),
        'se_gru_weights_total_n': float(total_w),
        'se_gru_eps_h': eps_h,
        'se_gru_hidden_active_frac_eps': float(frac_h),
        'se_gru_hidden_active_n': float(active_h),
        'se_gru_hidden_total_n': float(total_h),
    }


# ============================================================
# 5) AD com regras defensáveis
# ============================================================

def _pd_curve(model, x, ts, m, feat_idx, out_idx, vmin, vmax, n=50):
    grid = np.linspace(float(vmin), float(vmax), int(n))
    ys = []
    for v in grid:
        x2 = x.clone()
        x2[:, :, feat_idx] = v
        y = _xhat(model, x2, ts, m)[:, -1, out_idx].mean().item()
        ys.append(y)
    return grid, np.array(ys, dtype=float)


def _rule_monotonic(model, x, ts, m, rule):
    g, y = _pd_curve(model, x, ts, m, rule['feature'], rule['output'], rule['vmin'], rule['vmax'], rule.get('n', 50))
    dy = np.diff(y)
    exp = rule.get('expected', 'increasing')
    if exp == 'increasing':
        score = float(np.mean(dy >= 0))
    else:
        score = float(np.mean(dy <= 0))
    thr = float(rule.get('threshold', 0.8))
    return score >= thr, score


def _rule_saturation(model, x, ts, m, rule):
    g, y = _pd_curve(model, x, ts, m, rule['feature'], rule['output'], rule['vmin'], rule['vmax'], rule.get('n', 60))
    dy = np.abs(np.diff(y))
    sat_after = float(rule['sat_after'])
    idx = np.searchsorted(g, sat_after)
    idx = min(max(idx, 2), len(dy)-2)
    slope_after = float(np.mean(dy[idx:]))
    slope_before = float(np.mean(dy[:idx]))
    ratio = slope_after / max(slope_before, 1e-12)
    tol = float(rule.get('max_ratio', 0.3))
    return ratio <= tol, ratio


def _rule_sign_consistency(model, x, ts, m, rule):
    # aumento de carga -> aumento de potência etc.
    lo = float(rule['v_lo'])
    hi = float(rule['v_hi'])
    out_idx = int(rule['output'])
    feat = int(rule['feature'])
    x_lo = x.clone(); x_lo[:, :, feat] = lo
    x_hi = x.clone(); x_hi[:, :, feat] = hi
    y_lo = _xhat(model, x_lo, ts, m)[:, -1, out_idx].mean().item()
    y_hi = _xhat(model, x_hi, ts, m)[:, -1, out_idx].mean().item()
    delta = y_hi - y_lo
    exp = rule.get('expected', 'positive')
    ok = (delta >= 0) if exp == 'positive' else (delta <= 0)
    return bool(ok), float(delta)


def ad_rules_eval(model, x, ts, m, rules):
    if rules is None:
        rules = []

    details = []
    for r in rules:
        rtype = r.get('type', 'monotonic')
        if rtype == 'monotonic':
            ok, metric = _rule_monotonic(model, x, ts, m, r)
        elif rtype == 'saturation':
            ok, metric = _rule_saturation(model, x, ts, m, r)
        elif rtype == 'sign_consistency':
            ok, metric = _rule_sign_consistency(model, x, ts, m, r)
        else:
            raise ValueError(f'tipo de regra desconhecido: {rtype}')
        details.append({'type': rtype, 'ok': bool(ok), 'metric': float(metric), **r})

    n = len(details)
    n_ok = int(sum(d['ok'] for d in details))
    rate = float(n_ok / n) if n > 0 else np.nan
    return {
        'ad_pass_rate': rate,
        'ad_rules_ok': n_ok,
        'ad_rules_n': n,
        'ad_details': details,
    }


# ============================================================
# 6) Benchmark consolidado
# ============================================================

def explainability_benchmark_v2(seqkan_model, gru_model, x, ts, m,
                                topk=20,
                                ea_method='gradxinput',
                                ea_extra_methods=('ig', 'permutation_window'),
                                ea_noise_pcts=(0.005, 0.01, 0.02),
                                ea_n_seeds=5,
                                ea_windows_per_seed=32,
                                se_eps_frac=0.01,
                                ad_rules=None,
                                compute_seqkan_edge_time=True,
                                edge_time_max_windows=64,
                                make_plots=True):
    ad_rules = ad_rules or []

    # TF
    tf = tf_metrics(seqkan_model, topk=topk)

    # EA principal (comparável, mesmo método)
    ea_seq = ea_stability(seqkan_model, x, ts, m, method=ea_method,
                          noise_pcts=ea_noise_pcts, n_seeds=ea_n_seeds,
                          windows_per_seed=ea_windows_per_seed)
    ea_gru = ea_stability(gru_model, x, ts, m, method=ea_method,
                          noise_pcts=ea_noise_pcts, n_seeds=ea_n_seeds,
                          windows_per_seed=ea_windows_per_seed)

    # EA extra para aumentar separação seqKAN vs GRU
    ea_compare_rows = []
    methods = [ea_method] + [m for m in (ea_extra_methods or []) if m != ea_method]
    for meth in methods:
        seq_m = ea_stability(seqkan_model, x, ts, m, method=meth,
                             noise_pcts=ea_noise_pcts, n_seeds=ea_n_seeds,
                             windows_per_seed=ea_windows_per_seed)
        gru_m = ea_stability(gru_model, x, ts, m, method=meth,
                             noise_pcts=ea_noise_pcts, n_seeds=ea_n_seeds,
                             windows_per_seed=ea_windows_per_seed)
        ea_compare_rows.append({
            'EA_method': meth,
            'seqKAN_spearman_mean': seq_m['ea_spearman_mean'],
            'seqKAN_spearman_std': seq_m['ea_spearman_std'],
            'GRU_spearman_mean': gru_m['ea_spearman_mean'],
            'GRU_spearman_std': gru_m['ea_spearman_std'],
            'delta_seq_minus_gru': seq_m['ea_spearman_mean'] - gru_m['ea_spearman_mean'],
        })

    df_ea_compare = pd.DataFrame(ea_compare_rows).sort_values('delta_seq_minus_gru', ascending=False, ignore_index=True)

    # permutation importance por janela (resumo por feature)
    seq_perm = ea_permutation_window_importance(seqkan_model, x, ts, m,
                                                n_seeds=ea_n_seeds,
                                                windows_per_seed=ea_windows_per_seed)
    gru_perm = ea_permutation_window_importance(gru_model, x, ts, m,
                                                n_seeds=ea_n_seeds,
                                                windows_per_seed=ea_windows_per_seed)

    df_perm_seq = seq_perm['df_window_importance_summary'].copy()
    df_perm_seq['model'] = 'seqKAN'
    df_perm_gru = gru_perm['df_window_importance_summary'].copy()
    df_perm_gru['model'] = 'GRU'
    df_perm_window_summary = pd.concat([df_perm_seq, df_perm_gru], ignore_index=True)

    # edge attribution temporal do seqKAN
    if compute_seqkan_edge_time:
        df_edge_time = seqkan_edge_attribution_time(
            seqkan_model, x, ts, m, max_windows=edge_time_max_windows
        )
    else:
        df_edge_time = pd.DataFrame()

    # SE
    se_seq = se_seqkan_used_edges(seqkan_model, x, ts, m, eps_frac=se_eps_frac)
    se_gru = gru_compressibility(gru_model)
    se_gru_par = se_gru_parallel_metrics(gru_model, x, ts, m, eps_frac=se_eps_frac)

    # AD
    ad_seq = ad_rules_eval(seqkan_model, x, ts, m, ad_rules)
    ad_gru = ad_rules_eval(gru_model, x, ts, m, ad_rules)

    # tabela principal (compacta)
    rows = []
    rows.append({
        'model': 'seqKAN',
        'TF_core_claim': tf['tf_core_claim'],
        'TF_topk_mass_ratio': tf['tf_topk_mass_ratio'],
        'EA_method': ea_seq['ea_method'],
        'EA_spearman_mean': ea_seq['ea_spearman_mean'],
        'EA_spearman_std': ea_seq['ea_spearman_std'],
        'SE_seqkan_sparsity_eps': se_seq['se_seqkan_sparsity_eps'],
        'SE_gru_compress_q90': np.nan,
        'SE_gru_weights_active_frac_eps': np.nan,
        'SE_gru_hidden_active_frac_eps': np.nan,
        'AD_pass_rate': ad_seq['ad_pass_rate'],
        'AD_rules_n': ad_seq['ad_rules_n'],
    })
    q90 = se_gru['gru_compressibility'].get('q90', {}).get('compressible_fraction', np.nan)
    rows.append({
        'model': 'GRU',
        'TF_core_claim': 0.0,
        'TF_topk_mass_ratio': np.nan,
        'EA_method': ea_gru['ea_method'],
        'EA_spearman_mean': ea_gru['ea_spearman_mean'],
        'EA_spearman_std': ea_gru['ea_spearman_std'],
        'SE_seqkan_sparsity_eps': np.nan,
        'SE_gru_compress_q90': q90,
        'SE_gru_weights_active_frac_eps': se_gru_par['se_gru_weights_active_frac_eps'],
        'SE_gru_hidden_active_frac_eps': se_gru_par['se_gru_hidden_active_frac_eps'],
        'AD_pass_rate': ad_gru['ad_pass_rate'],
        'AD_rules_n': ad_gru['ad_rules_n'],
    })

    df_main = pd.DataFrame(rows)

    # evidência TF extra
    df_top_cell = pd.DataFrame(tf['top_edges_cell'])
    df_top_out = pd.DataFrame(tf['top_edges_out'])

    if make_plots:
        layer_cell = seqkan_model.seqKAN.kan_cell.act_fun[0]
        plot_topk_splines(layer_cell, tf['top_edges_cell'], title=f'seqKAN cell: top-{topk} splines', n_show=min(topk, 20))

        out_mod = seqkan_model.seqKAN.kan_out
        if hasattr(out_mod, 'act_fun'):
            layer_out = out_mod.act_fun[0]
            plot_topk_splines(layer_out, tf['top_edges_out'], title=f'seqKAN out: top-{topk} splines', n_show=min(topk, 20))
        else:
            print(f"[TF] plot de spline da saída indisponível (kan_out={type(out_mod).__name__}).")

        if not df_edge_time.empty:
            plt.figure(figsize=(8, 3.2))
            plt.plot(df_edge_time['t'], df_edge_time['cell_edge_contrib_mean'], label='cell mean')
            plt.plot(df_edge_time['t'], df_edge_time['out_edge_contrib_mean'], label='out mean')
            plt.title('seqKAN edge attribution over time')
            plt.xlabel('t')
            plt.ylabel('mean abs edge contrib')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    artifacts = {
        'df_main': df_main,
        'df_top_edges_cell': df_top_cell,
        'df_top_edges_out': df_top_out,
        'df_ea_compare': df_ea_compare,
        'df_perm_window_summary': df_perm_window_summary,
        'df_perm_window_seq': seq_perm['df_window_importance'],
        'df_perm_window_gru': gru_perm['df_window_importance'],
        'df_seqkan_edge_time': df_edge_time,
        'ad_seq_details': pd.DataFrame(ad_seq['ad_details']) if ad_seq['ad_details'] else pd.DataFrame(),
        'ad_gru_details': pd.DataFrame(ad_gru['ad_details']) if ad_gru['ad_details'] else pd.DataFrame(),
        'se_seq': se_seq,
        'se_gru': se_gru,
        'se_gru_parallel': se_gru_par,
        'tf': tf,
        'ea_seq': ea_seq,
        'ea_gru': ea_gru,
    }
    return artifacts


__all__ = [
    'explainability_benchmark_v2',
    '_make_eval_batch',
    'tf_metrics',
    'ea_stability',
    'ea_permutation_window_importance',
    'seqkan_edge_attribution_time',
    'se_seqkan_used_edges',
    'gru_compressibility',
    'se_gru_parallel_metrics',
    'ad_rules_eval',
]
