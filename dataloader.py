from datetime import datetime
import os

import numpy as np
import pandas as pd
from cognite.client import ClientConfig, CogniteClient
from cognite.client.credentials import OAuthClientCredentials
from sklearn.preprocessing import RobustScaler

from .segmenter import UnsupervisedStateSegmenter

# --- CREDENCIAIS ---
CLIENT_ID = "1b90ede3-271e-401b-81a0-a4d52bea3273"
CLIENT_SECRET = os.environ.get('COGNITE_CLIENT_SECRET')
# ----------------------------------

# Configurações do projeto publicdata
TENANT_ID = "48d5043c-cf70-4c49-881c-c638f5796997"
COGNITE_PROJECT = "publicdata"
BASE_URL = "https://api.cognitedata.com"

# Configuração das credenciais
creds = OAuthClientCredentials(
    token_url=f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token",
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    scopes=[f"{BASE_URL}/.default"],
)

# Instanciação do client Cognite
cdf_config = ClientConfig(client_name="eda-compressor-analysis", project=COGNITE_PROJECT, credentials=creds, base_url=BASE_URL)
client = CogniteClient(cdf_config)

class DataLoader():
    def __init__(self, start=datetime(2019, 9, 1), end=datetime(2019, 12, 1), granularity="2s", asset_name="23-KA-9101"):
        compressor_asset_list = client.assets.search(name=asset_name, limit=1)

        compressor_asset = compressor_asset_list[0]

        ts_list = client.time_series.list(asset_ids=[compressor_asset.id], limit=None)

        df_ts_info = ts_list.to_pandas()
        # Criar dicionário para mapeamento
        id_to_description = {ts.external_id: ts.description for ts in ts_list if ts.description}
        external_ids_interesse = df_ts_info['external_id'].tolist()

        print(f"\nBuscando dados para as {len(external_ids_interesse)} séries temporais encontradas.")

        # Define o período de tempo para a análise usando objetos datetime.
        start_time = start # Tempo disponível no dataset (3 meses)
        end_time = end

        # Recupera os dados com uma granularidade de 5 minutos
        df_data = client.time_series.data.retrieve_dataframe(
            external_id=external_ids_interesse,
            start=start_time,
            end=end_time, # Período de 3 meses
            granularity=granularity, # 5 minutos entre cada ponto de dados
            aggregates=["average"], # Média para granulidade
            ignore_unknown_ids=True, # Ignora IDs não encontrados
            include_unit=False,
            include_aggregate_name=False,
        )

        # Versões anteriores do SDK podiam incluir o agregado no nome da coluna.
        df_data.columns = [
            (col[0] if isinstance(col, tuple) else col).split('|')[0]
            for col in df_data.columns
        ]
        df_data_descriptive = df_data.rename(columns=id_to_description)
        self.df = df_data_descriptive.drop(columns=['PH (CBM) 1st Stage Poly Eff Dev']) # Coluna vazia

    def add_segments(self, segments, window, step, path=None, series=None):
        df = self.df.copy()
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        if series is not None:
            df = df[series]
        scaler = RobustScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        self.segmenter = UnsupervisedStateSegmenter(k_states=segments, window=window, step=step, device='cuda', min_slice_windows=window)
        if path is None:
            self.segmenter.fit(df)
        else:
            self.segmenter = self.segmenter.load(path)
        res = self.segmenter.predict(df)
        self.df = self.df[:len(res['states'])]
        self.df['states'] = res['states']
        self.pred=res

    def add_time_to_change_state_timestamp(self):
        df = self.df.copy()
        if df.empty or 'states' not in df.columns:
            self.df = df
            return
        states_series = df['states']
        unique_states = sorted(states_series.dropna().unique())
        if len(unique_states) == 0:
            self.df = df
            return

        # prepara colunas (mantidas vazias até preencher o próximo estado)
        state_cols = {state_id: f'state-{state_id}' for state_id in unique_states}
        for col in state_cols.values():
            df[col] = pd.NaT

        states_values = states_series.to_numpy()
        # pontos onde o estado muda (ignora primeira linha)
        change_positions = np.where(states_series.ne(states_series.shift()).to_numpy())[0]
        change_positions = change_positions[change_positions > 0]

        if len(change_positions) == 0:
            self.df = df
            return

        prev_pos = 0
        for pos in change_positions:
            next_state = states_values[pos]
            if pd.isna(next_state):
                prev_pos = pos
                continue
            col_name = state_cols.get(next_state)
            if col_name is None:
                prev_pos = pos
                continue
            change_time = df.index[pos]
            if prev_pos < pos:
                df.iloc[prev_pos:pos, df.columns.get_loc(col_name)] = change_time
            prev_pos = pos

        self.df = df

    def add_next_anomaly_timestamp(self, column: str, value: float, boundary: str):
        """
        Cria uma coluna 'anomaly' com o timestamp do PRÓXIMO evento em que
        a coluna `column` atinge/infringe o `value`, de acordo com `boundary`:
        - boundary = "above": evento quando column >= value
        - boundary = "below": evento quando column <= value

        Durante o próprio evento (períodos em que já está fora do limite),
        a coluna 'anomaly' fica como NaT. Quando volta para a zona "normal",
        passa a apontar para o próximo timestamp de evento.
        """

        df = self.df.copy()
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        scaler = RobustScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        if df.empty or column not in df.columns:
            self.df = df
            return

        series = df[column]
        b = boundary.lower()
        if b == "above":
            is_anom = series >= value
        elif b == "below":
            is_anom = series <= value
        else:
            raise ValueError("boundary deve ser 'above' ou 'below'")

        n = len(df)
        # inicializa coluna com NaT
        anomaly_col = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

        # se não há nenhum evento, só salva a coluna vazia
        if not is_anom.any():
            df["anomaly"] = anomaly_col
            self.df = df
            return

        mask = is_anom.to_numpy()

        # posições de INÍCIO de eventos (True depois de False)
        start_mask = mask & ~np.concatenate(([False], mask[:-1]))
        start_pos = np.flatnonzero(start_mask)

        if len(start_pos) == 0:
            df["anomaly"] = anomaly_col
            self.df = df
            return

        # para cada índice i, achar o primeiro start_pos > i
        idx = np.arange(n)
        next_start_idx_idx = np.searchsorted(start_pos, idx, side="right")
        valid = next_start_idx_idx < len(start_pos)

        index_values = df.index.to_numpy()
        next_ts = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        next_indices = start_pos[next_start_idx_idx[valid]]
        next_ts[valid] = index_values[next_indices]

        # durante o evento (mask == True) deve ser NaT
        next_ts[mask] = np.datetime64("NaT")

        df["anomaly"] = next_ts
        self.df = df
