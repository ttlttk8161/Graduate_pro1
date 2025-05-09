import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import rtdl
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ==============================================
# 0. 실행 환경에 따라 디바이스 결정
# ==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================
# 1. 데이터 로드 및 전처리
# ==============================================
file_path_csv = "data/data2.csv"  # 실제 파일 경로 설정

try:
    df = pd.read_csv(file_path_csv, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path_csv, encoding='euc-kr')

# 불필요한 컬럼 제거 (예: 'Name' 컬럼)
df = df.drop(columns=['Name'], errors='ignore')

# 2011~2023년 동안 존재하는 기업만 필터링
stock_min_year = df.groupby("Stock")["YEAR"].min()
stock_max_year = df.groupby("Stock")["YEAR"].max()

valid_stocks = stock_min_year[(stock_min_year == 2011) & (stock_max_year == 2023)].index
df_filtered = df[df["Stock"].isin(valid_stocks)]
year_counts = df_filtered.groupby("Stock")["YEAR"].count()
valid_stocks = year_counts[year_counts == 13].index
df_filtered = df_filtered[df_filtered["Stock"].isin(valid_stocks)]
df_filtered = df_filtered.sort_values(by=["Stock", "YEAR"])

# ==============================================
# 2. 연속형 & 이진 변수 분리, 정규화
# ==============================================
continuous_features = ["OWN", "FORN", "SIZE", "LEV", "CUR", "GRW", "ROA", "ROE", "CFO", "PPE", "AGE", "INVREC", "MB", "TQ"]
binary_features = ["BIG4", "LOSS"]

# Stock 정보를 범주형(정수형)으로 변환
df_filtered["Stock_ID"] = df_filtered["Stock"].astype('category').cat.codes




# 연속형 변수 MinMax 정규화
minmax_scaler = MinMaxScaler()
scaled_cont = minmax_scaler.fit_transform(
    df_filtered[continuous_features]
)

# ② logit(σ⁻¹) 변환 : [0,1] → ℝ
EPS = 1e-6                           # 수치 안정
scaled_cont = np.clip(scaled_cont, EPS, 1.0-EPS)
logit_cont  = np.log(scaled_cont / (1.0 - scaled_cont))

df_filtered[continuous_features] = logit_cont

# 이진 변수: 0/1 정수형
df_filtered[binary_features] = df_filtered[binary_features].astype(int)

# 전체 feature 목록
features = continuous_features + binary_features

# ==============================================
# 3. 기업 단위 시퀀스 데이터 생성 (각 기업 13년치)
# ==============================================
stocks = df_filtered["Stock"].unique()
grouped_cont = []
grouped_bin = []
grouped_year = []
grouped_stock = []

for stock in stocks:
    df_stock = df_filtered[df_filtered["Stock"] == stock].sort_values(by="YEAR")
    grouped_cont.append(df_stock[continuous_features].values)  # (13, 14)
    grouped_bin.append(df_stock[binary_features].values)         # (13, 2)
    grouped_year.append(df_stock["YEAR"].values)                 # (13,)
    grouped_stock.append(df_stock["Stock_ID"].iloc[0])            # scalar

X_cont_seq = np.stack(grouped_cont, axis=0)  # (num_stocks, 13, 14)
X_bin_seq = np.stack(grouped_bin, axis=0)     # (num_stocks, 13, 2)
year_seq = np.stack(grouped_year, axis=0)       # (num_stocks, 13)
stock_seq = np.array(grouped_stock)            # (num_stocks,)

# 타겟: 연속형 + 이진 (14+2=16)
target_seq = np.concatenate([X_cont_seq, X_bin_seq], axis=-1)  # (num_stocks, 13, 16)

# 텐서 변환
X_cont_tensor = torch.tensor(X_cont_seq, dtype=torch.float32)
X_bin_tensor = torch.tensor(X_bin_seq, dtype=torch.float32)
year_tensor_seq = torch.tensor(year_seq, dtype=torch.float32)
stock_tensor_seq = torch.tensor(stock_seq, dtype=torch.long)
target_tensor_seq = torch.tensor(target_seq, dtype=torch.float32)

# DataLoader 구성 (기업 단위 시퀀스)
dataset_seq = TensorDataset(X_cont_tensor, X_bin_tensor, year_tensor_seq, stock_tensor_seq, target_tensor_seq)
dataloader_seq = DataLoader(dataset_seq, batch_size=64, shuffle=True)

# ==============================================
# 4. sine-cosine 기반 연도 임베딩 함수 (sequence 지원)
# ==============================================
def get_sine_cosine_year_embedding(years, dim=13):
    """
    - years: (batch, num_years) 또는 (num_samples,) 형태의 실제 연도값 텐서
    - 출력: (..., dim) 형태의 연도 임베딩
    """
    # 만약 입력이 2차원 이상이면 마지막 차원에 대해 unsqueeze
    if len(years.shape) == 1:
        years = years.unsqueeze(-1)
    else:
        years = years.unsqueeze(-1)  # (batch, num_years, 1)
    half_dim = dim // 2
    freqs = torch.exp(
        torch.arange(0, half_dim, dtype=torch.float32) * (-np.log(10000.0) / half_dim)
    ).to(years.device)
    sinusoidal_input = years * freqs  # (..., half_dim)
    sin_embed = torch.sin(sinusoidal_input)
    cos_embed = torch.cos(sinusoidal_input)
    year_embedding = torch.cat([sin_embed, cos_embed], dim=-1)
    if year_embedding.shape[-1] < dim:
        pad_size = dim - year_embedding.shape[-1]
        padding = torch.zeros(year_embedding.shape[:-1] + (pad_size,), device=year_embedding.device)
        year_embedding = torch.cat([year_embedding, padding], dim=-1)
    return year_embedding

# ==============================================
# 5. Positional Encoding (Sinusoidal)
# ==============================================
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

# ==============================================
# 6. CompanySequenceModel: FT-Transformer + tst (Sequence 모델)
# ==============================================
class CompanySequenceModel(nn.Module):
    def __init__(self, cont_input_dim, bin_input_dim, year_dim, stock_dim=32, ft_out_dim=16, num_years=13):
        """
        - cont_input_dim: 연속형 변수 개수 (예: 14)
        - bin_input_dim: 이진 변수 개수 (예: 2)
        - year_dim: 연도 임베딩 차원 (예: 13)
        - stock_dim: Stock 임베딩 차원 (예: 32)
        - ft_out_dim: FT-Transformer 최종 출력 차원 (예: 16)
        - num_years: 시퀀스 길이 (예: 13년)
        """
        super(CompanySequenceModel, self).__init__()
        self.num_years = num_years
        
        # 임베딩 레이어 구성
        self.cont_embedding = nn.Linear(cont_input_dim, 32)
        self.bin_embedding = nn.Linear(bin_input_dim, 16)
        num_stock_embeddings = df_filtered["Stock_ID"].nunique()
        self.stock_embedding = nn.Embedding(num_embeddings=num_stock_embeddings, embedding_dim=stock_dim)
        
        # 모든 임베딩 결합 후 차원 조정을 위한 레이어:
        # total_input_dim = 32 (연속형) + 16 (이진) + 13 (연도) + 32 (stock) = 93
        total_input_dim = 32 + 16 + year_dim + stock_dim
        self.embedding = nn.Linear(total_input_dim, 128)
        self.bn = nn.BatchNorm1d(128)
        
        # rtdl 라이브러리의 FT-Transformer (행별 적용)
        self.ft_transformer = rtdl.FTTransformer.make_default(
            n_num_features=128,
            cat_cardinalities=None,
            d_out=ft_out_dim
        )
        
        # ----- TST (Time Series Transformer) 부분 시작 -----
        # 1. 1D Convolution: (batch, num_years, ft_out_dim) → (batch, num_years, ft_out_dim)
        #    → 시계열의 로컬 패턴(단기 의존성)을 캡처
        self.conv1d = nn.Conv1d(in_channels=ft_out_dim, out_channels=ft_out_dim, kernel_size=3, padding=1)
        
        # 2. Positional Encoding: 시퀀스 순서 정보를 추가 (기존 구현 재사용)
        self.pos_encoder = PositionalEncoding(ft_out_dim, max_len=num_years)
        
        # 3. TST Encoder: Transformer Encoder 블록을 2겹 쌓아 TST 역할 수행
        encoder_layer = nn.TransformerEncoderLayer(d_model=ft_out_dim, nhead=2, dropout=0.1, batch_first=True)
        self.tst_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # ----- TST 부분 끝 -----
    
    def forward(self, x_cont, x_bin, year_values, stock_id):
        batch, num_years, _ = x_cont.shape
        
        # 연도 임베딩: (batch, num_years, 13)
        year_embed = get_sine_cosine_year_embedding(year_values, dim=13)
        
        # 연속형, 이진 변수 임베딩 (각각 (batch*num_years, 임베딩 차원))
        cont_emb = self.cont_embedding(x_cont.view(-1, x_cont.shape[-1]))
        bin_emb = self.bin_embedding(x_bin.view(-1, x_bin.shape[-1]))
        
        # Stock 임베딩: (batch, stock_dim) → (batch, num_years, stock_dim)
        stock_emb = self.stock_embedding(stock_id)
        stock_emb = stock_emb.unsqueeze(1).repeat(1, num_years, 1)
        
        # 모든 임베딩을 결합: 최종 shape → (batch*num_years, total_input_dim)
        x_all = torch.cat([
            cont_emb,
            bin_emb,
            year_embed.view(-1, year_embed.shape[-1]),
            stock_emb.view(-1, stock_emb.shape[-1])
        ], dim=-1)
        
        x_all = self.embedding(x_all)  # (batch*num_years, 128)
        x_all = self.bn(x_all)
        
        # FT-Transformer 적용: (batch*num_years, ft_out_dim)
        ft_out = self.ft_transformer(x_num=x_all, x_cat=None)
        ft_out = ft_out.view(batch, num_years, -1)  # (batch, num_years, ft_out_dim)
        
        # ----- TST 파트 시작 -----
        # 1. 1D Convolution: 입력은 (batch, num_years, ft_out_dim)
        #    → Conv1d는 (batch, ft_out_dim, num_years) 형태를 요구하므로 전치
        conv_in = ft_out.transpose(1, 2)       # (batch, ft_out_dim, num_years)
        conv_out = self.conv1d(conv_in)          # (batch, ft_out_dim, num_years)
        conv_out = conv_out.transpose(1, 2)      # (batch, num_years, ft_out_dim)
        
        # 2. Positional Encoding 추가
        tst_input = self.pos_encoder(conv_out)   # (batch, num_years, ft_out_dim)
        
        # 3. TST Encoder 적용: (batch, num_years, ft_out_dim)
        tst_output = self.tst_encoder(tst_input)
        # ----- TST 파트 끝 -----
        
        return tst_output



# ==============================================
# 7. CompanySequenceModel 학습 (FT-Transformer + Tst)
# ==============================================
cont_input_dim = len(continuous_features)  # 14
bin_input_dim = len(binary_features)         # 2
year_dim = 13
stock_dim = 32 #바꿈
ft_out_dim = 16
num_years = 13
epochs = 200

# ── 클래스 불균형 보정용 pos_weight 계산 ───────────────
p_big4 = (df_filtered['BIG4'] == 1).mean()
p_loss = (df_filtered['LOSS'] == 1).mean()

pos_w  = torch.tensor([                 # 클래스 불균형 보정
    ((1-p_big4)/p_big4)**0.5,                 # √비율 완화
    ((1-p_loss)/p_loss)**0.5
], device=device)

bce_bin   = nn.BCEWithLogitsLoss(pos_weight=pos_w)   # ←★
mse_cont  = nn.MSELoss()                             # ←★
λ_bin_enc = 10.0                                     # ←★ (10-30 탐색)

company_model = CompanySequenceModel(cont_input_dim, bin_input_dim, year_dim, stock_dim, ft_out_dim, num_years).to(device)
optimizer_company = optim.Adam(company_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(epochs):
    for batch_cont, batch_bin, batch_year, batch_stock, batch_target in dataloader_seq:
        batch_cont  = batch_cont.to(device)
        batch_bin   = batch_bin.to(device)
        batch_year  = batch_year.to(device)
        batch_stock = batch_stock.to(device)
        batch_target= batch_target.to(device)        # (B,13,16)

        optimizer_company.zero_grad()
        pred = company_model(batch_cont, batch_bin, batch_year, batch_stock)

        # ① 타깃 분리
        pred_cont, pred_bin = pred[:, :, :14], pred[:, :, 14:]
        tgt_cont , tgt_bin  = batch_target[:, :, :14], batch_target[:, :, 14:]

        # ② 손실 계산
        loss_cont = mse_cont(pred_cont, tgt_cont)
        loss_bin  = bce_bin(pred_bin, tgt_bin)
        loss      = loss_cont + λ_bin_enc * loss_bin

        loss.backward()
        optimizer_company.step()

    if epoch % 10 == 0:
        print(f"[CompanySequenceModel] Epoch {epoch:03d} | "
              f"loss_cont={loss_cont.item():.4f}  "
              f"loss_bin={loss_bin.item():.4f}")

# 학습 완료 후, 각 기업(13년치)의 FT 출력(복원된 context vector)을 diffusion 모델 학습용으로 수집
company_model.eval()
all_outputs = []
all_year = []
all_stock = []
with torch.no_grad():
    for batch_cont, batch_bin, batch_year, batch_stock, _ in dataloader_seq:
        batch_cont = batch_cont.to(device)
        batch_bin = batch_bin.to(device)
        batch_year = batch_year.to(device)
        batch_stock = batch_stock.to(device)
        out = company_model(batch_cont, batch_bin, batch_year, batch_stock)  # (batch, 13, 16)
        all_outputs.append(out.cpu())
        all_year.append(batch_year.cpu())
        # 각 기업의 stock_id를 13년치로 확장
        batch_stock_expanded = batch_stock.unsqueeze(1).repeat(1, num_years)
        all_stock.append(batch_stock_expanded.cpu())

output_tensor_seq = torch.cat(all_outputs, dim=0)    # (num_companies, 13, 16)
year_tensor_seq = torch.cat(all_year, dim=0)           # (num_companies, 13)
stock_tensor_seq_expanded = torch.cat(all_stock, dim=0) # (num_companies, 13)

# 평탄화: 각 연도별 데이터를 diffusion 모델 학습용으로 (총 샘플수 = num_companies * 13)
output_tensor_flat = output_tensor_seq.view(-1, ft_out_dim)           # (N, 16)
year_tensor_flat = year_tensor_seq.view(-1)  # (num_companies*13,)
year_embed_flat = get_sine_cosine_year_embedding(year_tensor_flat, dim=13)                      # (N,)
stock_tensor_flat = stock_tensor_seq_expanded.view(-1)                  # (N,)



# ===============================================================
# 8'. Transformer-Denoiser 기반 Diffusion (시계열 컨텍스트 활용)
#     - 한 기업의 13년 시퀀스(13×16)를 한 샘플로 학습
# ===============================================================

# ----------------- (1) Diffusion 하이퍼파라미터 -----------------
T_diff      = 10
beta_start  = 1e-4
beta_end    = 2e-2
betas       = torch.linspace(beta_start, beta_end, T_diff, device=device)      # (T,)
alphas      = 1.0 - betas
alpha_bars  = torch.cumprod(alphas, dim=0)                                     # (T,)

# ----------------- (2) 학습 / 평가용 Dataset -----------------
# ❶ 시퀀스 전체(13×16)를 한 샘플로 사용하므로 평탄화는 **제거**
stock_scalar_seq = torch.tensor(stock_seq, dtype=torch.long)
bin_label_tensor_seq = torch.tensor(X_bin_seq, dtype=torch.float32) 
diff_dataset = TensorDataset(
    output_tensor_seq,          # (N_company, 13, 16)  — x₀
    year_tensor_seq,            # (N_company, 13)      — 실제 연도
    stock_scalar_seq,
    bin_label_tensor_seq # (N_company,)         — stock id
)
diff_dataloader = DataLoader(diff_dataset, batch_size=64, shuffle=True)



# ----------------- (3) Sinusoidal 시간-스텝 임베딩 -----------------
class TimeEmbedding(nn.Module):
    """ 1-D scalar t → (d_model) sinusoidal 임베딩 """
    def __init__(self, d_model: int):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2) / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t : (B, 1)  정규화되지 않은 정수 [1 … T]
        sinusoid = t * self.inv_freq            # (B, d_model/2)
        return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)  # (B, d_model)

# ----------------- (4) Transformer-Denoiser -----------------
class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        feat_dim=16,         # 14 cont + 2 bin
        d_model=64,
        nhead=4, num_layers=4,
        stock_emb_dim=32,
        year_pos_dim=13,
    ):
        super().__init__()

        # ────────── 공통 세팅은 그대로 ──────────
        self.year_proj = nn.Linear(year_pos_dim, d_model)

        n_stock = df_filtered["Stock_ID"].nunique()
        self.stock_emb = nn.Embedding(n_stock, stock_emb_dim)

        self.in_proj = nn.Linear(feat_dim + stock_emb_dim, d_model)

        self.t_embed = nn.Sequential(
            TimeEmbedding(d_model), nn.Linear(d_model, d_model), nn.SiLU()
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=13)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)

        # ── (★) 출력 헤드 두 개로 분리 ────────────────────
        self.out_cont = nn.Linear(d_model, 14)   # 연속형 14
        self.out_bin  = nn.Linear(d_model,  2)   # 이진 2 (로짓)
        # ────────────────────────────────────────────────

    def forward(self, x_t, years, stock_id, t_norm):
        B, S, _ = x_t.shape

        year_embed = get_sine_cosine_year_embedding(
            years.reshape(-1), dim=self.year_proj.in_features
        ).view(B, S, -1)
        year_embed = self.year_proj(year_embed)

        stock_emb = self.stock_emb(stock_id).unsqueeze(1).repeat(1, S, 1)

        h = self.in_proj(torch.cat([x_t, stock_emb], dim=-1))
        h = self.pos_enc(h) + year_embed + self.t_embed(t_norm).unsqueeze(1)

        h = self.encoder(h)

        # ── (★) 두 개를 따로 반환 ──
        return self.out_cont(h), self.out_bin(h)    # (B,S,14), (B,S,2)




# forward diffusion — 시퀀스 전체에 동일 t 적용
def q_sample(x0_seq, t_int):
    """
    x0_seq : (B,S,16)
    t_int  : (B,)  1-based
    """
    t_int = t_int.long()  
    sqrt_ab  = torch.sqrt(alpha_bars[t_int-1]).view(-1,1,1)      # (B,1,1)
    sqrt_1m  = torch.sqrt(1-alpha_bars[t_int-1]).view(-1,1,1)
    noise    = torch.randn_like(x0_seq)
    return sqrt_ab*x0_seq + sqrt_1m*noise, noise


# ---------------------------------------------------------------
latent_dim = 0              # Transformer-Denoiser에는 latent FiLM이 필요없어 제거
d_model    = 64
denoiser   = TransformerDenoiser(feat_dim=16, d_model=d_model, year_pos_dim=13).to(device)
opt        = optim.AdamW(denoiser.parameters(), lr=1e-3)

def snr_weight(t_idx: torch.Tensor,
               alpha_bars: torch.Tensor,
               strategy: str = "karras",
               rho: float = 1.2) -> torch.Tensor:
    """
    • t_idx      : (B,) 1-based  정수
    • alpha_bars : (T,)  torch.Tensor   – 전역 상수
    • strategy   : {"karras", "simple"}
        - "karras"  :  w = (snr + 1) ** -rho   (Karras et al., 2022 권장)
        - "simple"  :  w = 1 / (snr + 1)      (Nichol&Dhariwal 후속 연구)
    • 반환        : (B,)   loss 스칼라에 곱할 weight
    """
    # SNR(t) = ᾱₜ / (1-ᾱₜ)
    ab = alpha_bars[t_idx - 1]                       # (B,)
    snr = ab / (1.0 - ab)

    if strategy == "karras":                         # 기본
        weight = (snr + 1.0).pow(-rho)
    elif strategy == "simple":
        weight = 1.0 / (snr + 1.0)
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return weight

# ──────────────────────────────────────────────────────────
criterion_c = nn.MSELoss(reduction='none')                   # 연속형 MSE
bce_fn      = nn.BCEWithLogitsLoss(pos_weight=pos_w,         # 이진형 BCE
                                   reduction='none')

λ_bin = 10.0   # 이진 손실 가중치 (10~40 사이에서 튜닝)


for ep in range(200):
    for x0, yrs, st, bin_true in diff_dataloader:
        x0, yrs, st = x0.to(device), yrs.to(device), st.to(device)
        B = x0.size(0)

        # ① forward diffusion  q(x_t | x₀)
        t_int, _ = torch.sort(torch.randint(1, T_diff + 1, (B,), device=device))
        x_t, _   = q_sample(x0, t_int)                 # (B,S,16)
        t_norm   = t_int.float().unsqueeze(1) / T_diff # (B,1)

        # ② 두 헤드 예측 ──────────────── (★ 변경)
        cont_hat, bin_hat = denoiser(x_t, yrs, st, t_norm)   # (B,S,14), (B,S,2)
        bin_hat = bin_hat.clamp(-15, 15)

        # ③ 타깃 분리
        cont_tgt = x0[:, :, :14] 
        bin_tgt  = bin_true.to(device)

        # ④ 손실 계산 ─────────────────── (★ 변경)
        mse = criterion_c(cont_hat, cont_tgt).mean(dim=(1,2))          # (B,)
        bce = bce_fn(bin_hat, bin_tgt).mean(dim=(1, 2))              # (B,)
        w   = snr_weight(t_int, alpha_bars, "karras", rho=1.2)       # (B,)
        loss = (w * mse + λ_bin * bce).mean() 

        # ⑤ 옵티마이저
        opt.zero_grad()
        loss.backward()
        opt.step()

    if ep % 10 == 0:
        print(f"[Denoiser] ep {ep:03d} | loss {loss.item():.5f}")


# ── 3. 역확산 샘플러  (x̂₀ → ε̂ 변환 후 DDPM μ) ────────────────
@torch.no_grad()
def p_sample_loop(model, years_vec, stock_id, seq_len=13):
    """
    • model     : TransformerDenoiser (연속·이진 두 헤드)
    • years_vec : 2011 … 2023  1-D 텐서 (seq_len,)
    • stock_id  : 정수 ID (원본 Stock_ID 범위)
    반환        : (seq_len, 16)   — 14 cont(logit), 2 bin(0/1)
    """
    # x_T  ~  N(0, I)
    x = torch.randn(1, seq_len, 16, device=device)
    years = years_vec.unsqueeze(0).to(device)         # (1,S)
    stock = torch.tensor([stock_id], device=device)   # (1,)

    for t in range(T_diff, 0, -1):
        t_norm = torch.full((1, 1), t / T_diff, device=device)

        # ① 두 헤드 출력
        cont_hat, bin_hat = model(x, years, stock, t_norm)  # (1,S,14)/(1,S,2)
        x0_hat = torch.cat([cont_hat, bin_hat], dim=-1)     # (1,S,16)

        # ② x̂₀ → ε̂  변환
        alpha_bar_t = alpha_bars[t - 1]
        eps_hat = (x - alpha_bar_t.sqrt() * x0_hat) / torch.sqrt(1 - alpha_bar_t)

        # ③ DDPM μ_θ(x_t)
        beta_t, alpha_t = betas[t - 1], alphas[t - 1]
        mean = (1 / alpha_t.sqrt()) * (x - beta_t * eps_hat / torch.sqrt(1 - alpha_bar_t))

        # ④ 샘플 / 마지막 스텝
        if t > 1:
            x = mean + beta_t.sqrt() * torch.randn_like(x)
        else:
            x = mean                                    # t==1 → x₀

    # ── (★) 이진 로짓 → 0/1  후 반환 ───────────────────────
    cont_final, bin_logit = x[:, :, :14], x[:, :, 14:]
    bin_prob = torch.sigmoid(bin_logit)
    bin_final = (bin_prob > 0.5).float()

    x0_final = torch.cat([cont_final, bin_final], dim=-1)   # (1,S,16)
    return x0_final.squeeze(0)                              # (S,16)



def inverse_transform(data_np: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    data_np : (N, 16) ndarray
        ├─ data_np[:,  :14]  : 연속형 14개 (logit 값)  
        └─ data_np[:, 14:16] : 이진형 2개  (이미 0/1 - 확정)

    Returns
    -------
    data_np : (N, 16) ndarray
        • 연속형 : 원 단위 스케일 (Min-Max 역변환)  
        • 이진형 : 0 / 1  (int)
    """
    # ── ① 연속형 : logit → sigmoid → inverse-MinMax ──────────
    data_np[:, :14] = 1.0 / (1.0 + np.exp(-data_np[:, :14]))           # σ(logit) ∈ (0,1)
    data_np[:, :14] = minmax_scaler.inverse_transform(data_np[:, :14]) # 원 스케일

    # ── ② 이진형 : 이미 0/1 로 확정 → 정수 캐스팅만 ────────────
    data_np[:, 14:16] = data_np[:, 14:16].astype(int)

    return data_np



# ===============================================================
# 9. 대량 기업 생성 & CSV 저장 -----------------------------------
def generate_synthetic_companies(model, num_companies=50000,
                                 seq_len=13, start_vid=10000):
    model.eval()
    n_stock_real = int(df_filtered["Stock_ID"].nunique())
    all_rows = []
    years_vec = torch.arange(2011, 2011+seq_len, dtype=torch.float32)

    for i in range(num_companies):
        virt_id   = start_vid + i
        stock_real= virt_id % n_stock_real
        x_gen     = p_sample_loop(model, years_vec, stock_real, seq_len)
        x_np      = inverse_transform(x_gen.cpu().numpy())
        rows      = np.hstack([
                      np.full((seq_len,1), virt_id),
                      years_vec.view(-1,1).numpy(),
                      x_np
                    ])
        all_rows.append(rows)
        if (i+1) % 1000 == 0:
            print(f"  • {i+1}/{num_companies} done")

    return np.vstack(all_rows)

# ── 1. 평가 모드 전환 ──────────────────────────────
denoiser.eval()

all_prob = []       # 확률을 쌓아둘 리스트

with torch.no_grad():
    for x0, yrs, st, _ in diff_dataloader: 
        # q(x_t | x0)  : 어떤 t 값이든 상관없지만 t=1(=가벼운 노이즈)로 해도 OK
        x_t, _ = q_sample(x0.to(device),
                          torch.ones(len(x0), device=device))  # t=1
        _, bin_hat = denoiser(
            x_t,
            yrs.to(device),
            st.to(device),
            torch.ones(len(x0), 1, device=device)              # t_norm = 1/T
        )
        # 시그모이드 → 확률
        prob = torch.sigmoid(bin_hat).cpu().numpy().ravel()    # 1-D
        all_prob.append(prob)

# ── 2. 히스토그램 출력 ─────────────────────────────
all_prob = np.concatenate(all_prob)          # (N_total * 2,)  두 이진 피처 합산
hist, bin_edges = np.histogram(all_prob, bins=10, range=(0.0, 1.0))

@torch.no_grad()
def positive_ratio(model, dataloader, threshold=0.5):
    """
    학습 종료 후 BIG4·LOSS 두 플래그에 대해
      • pred=모델 1 예측 비율
      • true=실제 1 비율
      • diff=차이           를 출력
    """
    model.eval()
    n_pred_pos = torch.zeros(2)   # BIG4, LOSS
    n_true_pos = torch.zeros(2)
    n_total    = 0

    for x0, yrs, st, bin_true in dataloader:
        x0, yrs, st = x0.to(device), yrs.to(device), st.to(device)
        B, S = x0.size(0), x0.size(1)

        # t=1 로 가벼운 노이즈 추가
        t_int  = torch.ones(B, device=device, dtype=torch.long)
        x_t, _ = q_sample(x0, t_int)
        t_norm = t_int.float().unsqueeze(1) / T_diff

        _, bin_logit = model(x_t, yrs, st, t_norm)        # (B,S,2)
        prob = torch.sigmoid(bin_logit)                   # 확률
        pred_pos = (prob > threshold).sum(dim=(0, 1))     # (2,)
        true_pos = (bin_true.to(device) > 0.5).sum(dim=(0,1))
        n_pred_pos += pred_pos.cpu()
        n_true_pos += true_pos.cpu()
        n_total    += B * S

    ratio_pred = (n_pred_pos / n_total).numpy()
    ratio_true = (n_true_pos / n_total).numpy()
    diff       = ratio_pred - ratio_true

    print(f"\n★ Positive-ratio check (thr={threshold})")
    for i, name in enumerate(["BIG4", "LOSS"]):
        print(f" {name:5s}  pred={ratio_pred[i]:.3%}  "
              f"true={ratio_true[i]:.3%}  diff={diff[i]:+.2%}")
    print()

# 2) BIG4 / LOSS 양성 예측-비율 vs. 실제-비율 출력
positive_ratio(denoiser, diff_dataloader, threshold=0.5)

print("Probabilities histogram (0~1):")
for i in range(10):
    print(f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}: {hist[i]}")

print("▶  가짜 기업 생성 시작 …")
fake_data = generate_synthetic_companies(denoiser, num_companies=50000)
# (1) 원본 순서로 먼저 DataFrame 생성  ----------------------------
raw_cols = ["Stock_ID", "YEAR"] + continuous_features + binary_features
df_fake  = pd.DataFrame(fake_data, columns=raw_cols)

# (2) 원하는 열 순서로 재배치  --------------------------------------
final_cols = ["OWN", "FORN", "BIG4","SIZE", "LEV", "CUR", "GRW", "ROA",
              "ROE","CFO", "PPE", "AGE", "INVREC", "MB", "TQ", "LOSS"]
df_fake = df_fake[["Stock_ID", "YEAR"] + final_cols] 

df_fake["Stock_ID"] = df_fake["Stock_ID"].astype(int)
df_fake["YEAR"]     = df_fake["YEAR"].astype(int)

for col in final_cols:
    df_fake[col] = df_fake[col].round(8)

df_fake.to_csv("generated_50000.csv", index=False)
print("✅ 50 000개 기업 × 13년 시계열 저장 ")
