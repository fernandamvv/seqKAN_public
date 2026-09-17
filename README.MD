# industrial_ts – Toolkit Moderno para Séries Temporais Industriais - Versão CPE727 2025 03

Toolkit voltado a séries temporais contínuas com grandes lacunas de amostragem. O pacote concentra encoders Jump-ODE, baselines RNN (GRU/LSTM) e um modelo de difusão latente completo (`tsdiffusion.TSDiffusion`, apelidado aqui de **IndustrialTS**). Toda a documentação abaixo foca nos módulos ativos (`dataloader.py`, `ode_jump.py`, `ode_jump_encoder.py`, `tsdiffusion.py`, `gru.py`, `lstm.py`).

## Dependências e instalação
- Python 3.10+
- PyTorch (GPU opcional)
- torchdiffeq
- pandas, numpy, scipy, scikit-learn
- cognite-sdk (para ingestão direta)
- matplotlib / seaborn / plotly (visualização opcional)

Instale na raiz do projeto: `pip install -r requirements.txt`. Defina `COGNITE_CLIENT_SECRET` para habilitar ingestão direta do Cognite Data Fusion.

## Dataloaders Disponíveis
### Cognite (Open Industrial Data)
`industrial_ts.dataloader.DataLoader` baixa séries temporais do asset desejado e já expõe rotinas de enriquecimento.
```python
from industrial_ts.dataloader import DataLoader

dloader = DataLoader(asset_name="23-KA-9101", granularity="2s")
dloader.add_segments(segments=6, window=30, step=10)  # segmentação HMM multivariada
dloader.add_time_to_change_state_timestamp()
dloader.add_next_anomaly_timestamp("PH (CBM) 1st Stage ActPress Ratio", value=2.0, boundary="above")
df_cognite = dloader.df
```

### 3W DataLake
O repositório mantém um loader específico para o DataLake 3W (`from loader import Loader3W`). Ele fornece preprocessing, estatísticas (`stats.pkl`) e integração direta com os pipelines `train_model`/`test_model` presentes em `ode_jump.py` e `tsdiffusion.py`.
```python
from loader import Loader3W

loader3w = Loader3W()
loader3w.load_stats("stats.pkl")
datasets = list(loader3w.preprocess(include_status_pred=True, status_pred_window=900))
```

## Núcleo Contínuo-Discreto
### `industrial_ts.ode_jump`
- **`DenoiseGate`**: mistura sinal original e reconstruído para manter consistência em dados parcialmente observados.
- **`JumpODE`**: integra sinais entre eventos pelo método RK4 e aplica *jumps* via GRUCell.
- **`ODEJump`**: empacota encoder linear, `JumpODE`, decoder e cabeças auxiliares. Contém utilitários como `train_model`, `_make_dataset`, `test_model` e `impute` para pipelines supervisionados sem difusão.

### `industrial_ts.ode_jump_encoder`
- **`CrossAttention`, `TransformerGate`, `AttnMemory`**: blocos de atenção causal usados para refinar o estado contínuo.
- **`JumpODEEncoder`**: versão auto-regressiva com atenção multi-head, FiLM temporal (`TimeHybridEncoding`) e máscara causal.
- **`ODEJumpEncoder`**: herda `ODEJump` e reutiliza `JumpODEEncoder` para construir modelos condicionais (ex.: ruído da difusão).

## IndustrialTS (`industrial_ts.tsdiffusion.TSDiffusion`)
Importe usando o alias abaixo para reforçar o posicionamento:
```python
from industrial_ts.tsdiffusion import TSDiffusion as IndustrialTS
```

### Construtor (`__init__`)
```python
IndustrialTS(
    in_channels: int,
    hidden_dim: int = 256,
    static_dim: int = 0,
    status_dim: int = 0,
    lam: list[float] = [0.9, 0.0, 0.0, 0.1, 0.0, 0.0],
    n_heads: int = 4,
    n_layers: int = 4,
    num_steps: int = 1000,
    cost_columns: list[str] | None = None,
    n_heads_g: int = 4,
    n_layers_g: int = 4,
    log_likelihood: bool = True,
    sigma_temp: float = 0.7,
)
```
- `in_channels/static_dim/status_dim` devem casar exatamente com as listas de colunas usadas no dataset.
- `lam` ativa/desativa cabeças de perda (L1..L6). Se `lam[4]` ou `lam[5]` > 0 o VAE correspondente é instanciado.
- `num_steps` define o número de passos do DDPM latente; ajuste junto de `sigma_temp`.
- `cost_columns` habilita pesos por coluna durante o treino (perda L1/L5).
- `n_heads/n_layers` controlam o Transformer interno do Jump-ODE; `n_heads_g/n_layers_g` controlam os encoders auxiliares (ex.: ruído e horizonte).

### Colunas esperadas & pesos de perda
- `feature_cols`: colunas contínuas normalizadas (usualmente robust-scaler) -> `in_channels`.
- `static_features_cols` (opcional): atributos fixos por série -> `static_dim`.
- `predict_state_cols`: colunas que armazenam horizontes futuros/estados -> `status_dim`.
- `timestamp_col`: datetime ou índice monotônico. Internamente convertido para segundos normalizados por `TS_SPAN`.
- `lam = [L1, L2, L3, L4, L5, L6]` controla perdas: reconstrução, ruído da difusão, horizonte/estado, presença, ELBO recon, ELBO horizonte.

### IndustrialTS API (todos os métodos públicos)
#### `forward(...)`
Executa o passo direto da difusão (treino ou inferência). Normalmente chamado dentro do loop de treino ou avaliação.
```python
model = IndustrialTS(in_channels=11, hidden_dim=256, status_dim=3, lam=[0.9,0.1,0.1,0.1,0.0,0.0])
x_masked = batch_x * mask_train
mask_ts = mask_train.any(dim=2, keepdim=True).float()
state, state_noise, state_tmax, x_hat, tmax_hat, *_ = model.forward(
    x_masked,
    timestamps=batch_ts,
    static_feats=batch_static,
    return_x_hat=True,
    mask=mask_train,
    mask_ts=mask_ts,
    test=False,
)
```

#### `impute(state, device, timestamps, static_feats, mask, x0, return_x_hat=True)`
Reconstrói os sinais observados a partir de um estado latente já codificado (usado quando `lam[1]==0`, isto é, sem passo de difusão completo).
```python
latent = model.encoder(torch.cat([batch_x, batch_mask], dim=-1))
x_recon = model.impute(
    state=latent,
    device=batch_x.device,
    timestamps=batch_ts,
    static_feats=batch_static,
    mask=batch_mask,
    x0=batch_x,
    return_x_hat=True,
)
```

#### `denoise(state, timestamps, static_feats, device, steps, x0, mask, enforce_data_consistency=True)`
Executa o DDPM reverso no espaço latente, devolvendo o sinal clamped (substitui pontos mascarados pelo original) e o estado final.
```python
x_filled, latent_final = model.denoise(
    state=latent,
    timestamps=batch_ts,
    static_feats=batch_static,
    device=batch_x.device,
    steps=200,
    x0=batch_x,
    mask=batch_mask,
)
```

#### `train_cognite(...)`
Pipeline de treino supervisionado com splits por janela e *oversampling* de estados. É um gerador: retorna métricas por época e finaliza com `None` após o teste final.
```python
trainer = model.train_cognite(
    df=df_cognite,
    feature_cols=feature_cols,
    static_features_cols=static_cols,
    timestamp_col="timestamp",
    states_col="states",
    predict_state_cols=["state-pred-0", "state-pred-1"],
    status_pred_window=900,
    batch_size=32,
    epochs=20,
)
for metrics in trainer:
    if metrics is None:
        break
    print(metrics["micro_mse"], metrics.get("micro_mse_s"))
```
Parâmetros úteis:
- `window_size/window_step`: tamanho e passo das janelas geradas por `_make_dataset`.
- `states_col` e `predict_state_cols`: colunas usadas para rotular grupos (estado) e horizonte.
- `only_gru=True` mantém apenas os ramos GRU dos encoders, útil para ablações.
- `lambda1`, `weight_tmax`, `kl_start/kl_end/kl_warmup_epochs`: hiperparâmetros finos das perdas compostas.
- `optimizer_name`/`optimizer_params` e `warmup_steps/min_lr_factor` controlam o otimizador + scheduler.

#### `test_model_preforward(x, timestamps, static_feats=None, mask=None)`
Atalho que codifica, denoisa e decodifica um lote de tensores já pré-processados.
```python
with torch.no_grad():
    recon = model.test_model_preforward(
        x=batch_x,
        timestamps=batch_ts,
        static_feats=batch_static,
        mask=batch_mask,
    )
```

#### `test_model(loader, y_seq, all_groups=None, only_gru=False, reconstruction_test=True, status_pred_window=600)`
Calcula métricas macro/micro, ELBOs e intervalos de confiança. Aceita qualquer `DataLoader` construído via `_make_dataset`.
```python
test_dl = DataLoader(test_subset, batch_size=64, shuffle=False)
metrics = model.test_model(
    loader=test_dl,
    y_seq=state_labels_test,
    all_groups=np.unique(state_labels_train),
    status_pred_window=900,
)
print("macro mse", metrics["macro_mse"], "macro mse noise", metrics["macro_mse_n"])
```

#### `denoise_dataframe(df, feature_cols, timestamp_col, static_features_cols=None, window_size=None, window_step=1, steps=None, replace_only_missing=True)`
Opera diretamente em um DataFrame, devolvendo duas versões: reconstrução e (quando `lam[1]>0`) imputação completa.
```python
rec_df, imp_df = model.denoise_dataframe(
    df=df_cognite,
    feature_cols=feature_cols,
    timestamp_col="timestamp",
    static_features_cols=static_cols,
    window_size=600,
    window_step=50,
)
```

#### `generate_samples(df, feature_cols, timestamp_col, slice, size, static_features_cols=None)`
Gera passos futuros de forma autoregressiva, mantendo a janela definida pelo `slice` de contexto.
```python
samples = model.generate_samples(
    df=df_cognite,
    feature_cols=feature_cols,
    timestamp_col="timestamp",
    slice=slice(-600, None),
    size=50,
    static_features_cols=static_cols,
)
```
Use este método sempre que quiser “continuar” a série; internamente ele chama `denoise_dataframe` a cada passo para garantir consistência com as observações passadas. Caso esteja procurando por “generate_series”, este é o método equivalente no código atual.

#### `extract_percentile_pdf_tmax(df, feature_cols, timestamp_col, predict_state_cols, window_size, window_step, static_features_cols=None, percentile=0.5)`
Calcula o quantil desejado (em fração da janela) para o tempo restante até mudança de estado, escrevendo novas colunas no DataFrame.
```python
tmax_pct90 = model.extract_percentile_pdf_tmax(
    df=df_cognite,
    feature_cols=feature_cols,
    timestamp_col="timestamp",
    predict_state_cols=["state-pred-0", "state-pred-1"],
    window_size=600,
    window_step=60,
    percentile=0.9,
)
```

#### `generate_pdf_tmax(df, feature_cols, timestamp_col, predict_state_cols, slice, static_features_cols=None, status_pred_window=300, grid_points=200)`
Extrai μ/σ (em segundos) e a PDF completa do tempo até a próxima transição para a janela especificada.
```python
pdf_df = model.generate_pdf_tmax(
    df=df_cognite,
    feature_cols=feature_cols,
    timestamp_col="timestamp",
    predict_state_cols=["state-pred-0"],
    slice=slice(-600, None),
    static_features_cols=static_cols,
    status_pred_window=900,
    grid_points=256,
)
```
Ambas as funções acima compartilham a mesma cabeça de tmax: `extract_percentile_pdf_tmax` injeta colunas diretamente no DataFrame, enquanto `generate_pdf_tmax` devolve a densidade completa (grid e PDF). Use uma ou outra dependendo se precisa de percentis específicos ou da curva inteira.

## Backbones alternativos
### `industrial_ts.lstm`
- **`LSTM`**: célula com opções bidirecionais, fusão por `concat`, `gate` ou `gru` e suporte a *variational dropout*.
- **`TS_LSTM`**: baseline supervisionado que herda `ODEJump` e usa a pilha LSTM para reconstrução.
- **`LSTMEncoder`**: versão encoder-only utilizada nos modelos difusivos.
- **`TSDF_LSTM`**: envolve `IndustrialTS`, substituindo `encoder_ode_x/tmax` por `LSTMEncoder` e mantendo cabeças de VAE/difusão. Ideal para séries densas.
```python
from industrial_ts.lstm import TSDF_LSTM
model_lstm = TSDF_LSTM(
    in_channels=11,
    hidden_dim=352,
    status_dim=3,
    bi_lstm=True,
    bi_method='gate',
    lam=[0.9,0.1,0.1,0.1,0.0,0.0],
)
```

### `industrial_ts.gru`
- **`GRU`**: versão com variações bidirecionais similares às do LSTM.
- **`TS_GRU`**: baseline supervisionado herdeiro de `ODEJump`.
- **`GRUEncoder`**: encoder especializado usado nos modelos de difusão.
- **`TSDF_GRU`**: wrapper equivalente ao LSTM, porém com GRU Cells; permite ativar/desativar LayerNorm e dropout variacional.
```python
from industrial_ts.gru import TSDF_GRU
model_gru = TSDF_GRU(
    in_channels=11,
    hidden_dim=352,
    status_dim=3,
    bi_gru=True,
    lam=[0.9,0.1,0.1,0.1,0.0,0.0],
)
```

## Observações finais
- Sempre alinhe `feature_cols` e `static_features_cols` com o que foi usado no treino (incluindo ordem e duplicatas).
- Para ativações VAE (`lam[4]`/`lam[5]`), mantenha `sigma_temp` compatível com a escala dos dados.
- Tanto `Loader3W` quanto `DataLoader` podem ser usados para gerar DataFrames de treino. O primeiro já traz splits e estatísticas dos ativos 3W; o segundo integra diretamente com o Open Industrial Data via OAuth.
- GPUs são recomendadas para `num_steps >= 1000` ou janelas maiores que 600 timestamps, mas o código funciona em CPU para depuração.
