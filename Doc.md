# Proposed Model: Company Data Generation 코드 설명

이 문서는 `Backend/Proposed_Model.ipynb` Jupyter Notebook 파일의 각 코드 셀과 라인에 대한 설명을 제공합니다.

## Configuration Parameters

```python
# 모델 및 학습 관련 하이퍼파라미터 설정
YEAR_DIM_PARAM = 13          # 연도 임베딩 차원
NUM_YEARS_PARAM = 13         # 처리할 연도 수 (예: 2011~2023)
FT_OUT_DIM_PARAM = 16        # FT-Transformer 모델의 출력 차원
STOCK_DIM_PARAM = 32         # 기업(Stock) ID 임베딩 차원
DENOISER_D_MODEL = 64        # Denoiser 모델의 내부 차원 (d_model)
EPOCHS_COMPANY_MODEL = 5     # CompanySequenceModel 학습 에포크 수 (테스트용으로 줄임, 원본 200)
EPOCHS_DENOISER_MODEL = 5    # Denoiser 모델 학습 에포크 수 (테스트용으로 줄임, 원본 200)
BATCH_SIZE = 64              # 학습 시 배치 크기
LEARNING_RATE_COMPANY = 0.001 # CompanySequenceModel 학습률
LEARNING_RATE_DENOISER = 1e-3 # Denoiser 모델 학습률
TENSORBOARD_LOG_DIR_COMPANY = 'runs/company_model_experiment' # Company 모델 TensorBoard 로그 저장 경로
TENSORBOARD_LOG_DIR_DENOISER = 'runs/denoiser_model_experiment' # Denoiser 모델 TensorBoard 로그 저장 경로
```

- 각 변수는 모델의 구조나 학습 과정을 제어하는 중요한 설정값입니다.
- 주석을 통해 각 파라미터의 의미를 설명하고 있습니다.

## Imports

```python
# 필요한 라이브러리 임포트
import pandas as pd               # 데이터 조작 및 분석 (DataFrame 등)
import numpy as np                # 수치 계산 (배열 등)
import torch                      # PyTorch 핵심 라이브러리 (텐서, 자동 미분 등)
import torch.nn as nn             # 신경망 모듈 (레이어, 손실 함수 등)
import torch.optim as optim       # 최적화 알고리즘 (Adam 등)
import rtdl                       # Tabular DL 라이브러리 (FT-Transformer 등)
from sklearn.preprocessing import MinMaxScaler # 데이터 정규화 (Min-Max 스케일링)
from torch.utils.data import DataLoader, TensorDataset # 데이터 로딩 및 배치 처리
import math                       # 수학 함수 (PositionalEncoding에서 사용)
from torch.utils.tensorboard.writer import SummaryWriter # TensorBoard 로깅
import time                       # 시간 관련 함수 (로그 디렉토리 이름 생성 등)
#from torchsort import soft_sort # CDF 손실 계산용 (현재 주석 처리됨)
```

- 모델 구현 및 학습에 필요한 다양한 파이썬 라이브러리를 가져옵니다.
- 각 라이브러리의 주된 용도를 주석으로 설명합니다.

## 0. 실행 환경에 따라 디바이스 결정 및 시각화 활성화

```python
# TensorBoard 확장 기능 로드 (Jupyter 환경)
%load_ext tensorboard
# TensorBoard 실행 (백그라운드) 및 로그 디렉토리 지정
%tensorboard --logdir runs
# 또는 특정 포트 및 모든 인터페이스 바인딩으로 실행 시:
# tensorboard --logdir runs --bind_all --port 6006

# 사용 가능한 디바이스 확인 (GPU 우선 사용, 없으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 선택된 디바이스 출력
print(f"Using device: {device}")
```

- `%load_ext tensorboard`: Jupyter Notebook 환경에서 TensorBoard 매직 명령어를 사용하기 위해 확장 기능을 로드합니다.
- `%tensorboard --logdir runs`: TensorBoard를 실행하고, 학습 로그가 저장될 `runs` 디렉토리를 지정합니다.
- `torch.device(...)`: CUDA(NVIDIA GPU)를 사용할 수 있으면 'cuda'를, 그렇지 않으면 'cpu'를 `device` 변수에 할당합니다. 모델과 텐서는 이 `device`로 이동되어 계산됩니다.
- `print(...)`: 현재 사용 중인 디바이스(CPU 또는 GPU)를 출력합니다.

## 1. 데이터 로드 및 전처리

```python
# CSV 파일 경로 설정
file_path_csv = "data/Table_Data.csv"  # 실제 파일 경로 설정

# CSV 파일 로드 시도 (UTF-8 인코딩 우선, 실패 시 EUC-KR 시도)
try:
    df = pd.read_csv(file_path_csv, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path_csv, encoding='euc-kr')

# 불필요한 'Name' 컬럼이 존재하면 제거 (오류 발생 시 무시)
if 'Name' in df.columns:
    df = df.drop(columns=['Name'], errors='ignore')

# 각 기업(Stock)별 데이터의 시작 연도와 종료 연도 계산
stock_min_year = df.groupby("Stock")["YEAR"].min()
stock_max_year = df.groupby("Stock")["YEAR"].max()

# 2011년부터 2023년까지 데이터가 존재하는 기업(Stock) 필터링
valid_stocks_initial = stock_min_year[(stock_min_year == 2011) & (stock_max_year == 2023)].index
# 필터링된 기업 데이터만 선택하여 새로운 DataFrame 생성 (.copy()로 경고 방지)
df_filtered = df[df["Stock"].isin(valid_stocks_initial)].copy()

# 필터링된 데이터가 비어 있는지 확인
if df_filtered.empty:
    print("No companies found that existed continuously from 2011 to 2023. Exiting.")
    # 데이터가 없으면 처리를 중단하거나 오류 발생시킬 수 있음
else:
    # 정확히 13년치(NUM_YEARS_PARAM) 데이터가 있는 기업만 최종 선택
    year_counts = df_filtered.groupby("Stock")["YEAR"].count()
    valid_stocks_final = year_counts[year_counts == NUM_YEARS_PARAM].index
    df_filtered = df_filtered[df_filtered["Stock"].isin(valid_stocks_final)].copy()
    # 기업(Stock)과 연도(YEAR) 기준으로 데이터 정렬
    df_filtered = df_filtered.sort_values(by=["Stock", "YEAR"])

    # 전처리 전 데이터 정보 출력
    print("Data before preprocessing:")
    print(f"Original df shape: {df.shape}") # 원본 DataFrame 크기
    print(f"Number of unique stocks before filtering: {df['Stock'].nunique()}") # 필터링 전 고유 기업 수
    display(df.head()) # 원본 데이터 상위 5개 행 출력 (Jupyter 환경)

    # 전처리 후 데이터 정보 출력
    print("\nData after preprocessing (filtering):")
    print(f"Filtered df shape: {df_filtered.shape}") # 필터링 후 DataFrame 크기
    print(f"Number of unique stocks after filtering: {df_filtered['Stock'].nunique()}") # 필터링 후 고유 기업 수
    display(df_filtered.head()) # 필터링된 데이터 상위 5개 행 출력
```

- `file_path_csv`: 원본 데이터 CSV 파일의 경로를 지정합니다.
- `pd.read_csv(...)`: Pandas를 사용하여 CSV 파일을 DataFrame으로 읽어옵니다. 인코딩 문제 발생 시 `euc-kr`로 다시 시도합니다.
- `df.drop(...)`: 'Name' 컬럼이 있다면 제거합니다.
- `df.groupby(...)`: 기업별로 그룹화하여 최소/최대 연도를 찾습니다.
- `valid_stocks_initial`: 2011년부터 2023년까지 데이터가 끊이지 않고 존재하는 기업 목록을 찾습니다.
- `df_filtered = df[...]`: 해당 기업들의 데이터만 `df_filtered`에 저장합니다.
- `if df_filtered.empty:`: 필터링 결과 데이터가 없는 경우 메시지를 출력합니다.
- `else:`: 데이터가 있는 경우, 추가 필터링을 진행합니다.
    - `year_counts = ...`: 기업별 연도 데이터 개수를 셉니다.
    - `valid_stocks_final = ...`: 정확히 `NUM_YEARS_PARAM`(13)개의 연도 데이터를 가진 기업만 최종 선택합니다.
    - `df_filtered = df_filtered[...]`: 최종 선택된 기업 데이터만 남깁니다.
    - `df_filtered.sort_values(...)`: 기업과 연도 순으로 정렬하여 시계열 순서를 맞춥니다.
- `print(...)` 및 `display(...)`: 전처리 전후의 데이터 크기, 고유 기업 수, 샘플 데이터를 출력하여 확인합니다.

## 2. 연속형 & 이진 변수 분리, 정규화

```python
# 연속형 변수와 이진 변수 목록 정의
continuous_features = ["OWN", "FORN", "SIZE", "LEV", "CUR", "GRW", "ROA", "ROE", "CFO", "PPE", "AGE", "INVREC", "MB", "TQ"]
binary_features = ["BIG4", "LOSS"]

# 필터링된 데이터가 비어 있지 않은 경우에만 실행
if not df_filtered.empty:
    # 기업(Stock) 코드를 범주형으로 변환하고, 각 범주에 고유 정수 ID 부여
    df_filtered.loc[:, "Stock_ID"] = df_filtered["Stock"].astype('category').cat.codes

    # 연속형 변수에 대해 Min-Max 정규화 수행 (0~1 사이 값으로 변환)
    minmax_scaler = MinMaxScaler() # MinMaxScaler 객체 생성
    scaled_cont = minmax_scaler.fit_transform( # 데이터에 맞춰 스케일러 학습 및 변환 동시 수행
        df_filtered[continuous_features] # 연속형 변수 데이터 선택
    )

    # Logit 변환 (σ⁻¹): 정규화된 값(0~1)을 실수 전체 범위(ℝ)로 변환
    EPS = 1e-6                           # 0 또는 1 값 방지를 위한 작은 값 (수치 안정성)
    scaled_cont = np.clip(scaled_cont, EPS, 1.0-EPS) # 값을 EPS와 1-EPS 사이로 제한
    logit_cont  = np.log(scaled_cont / (1.0 - scaled_cont)) # Logit 변환 공식 적용

    # 변환된 연속형 변수 값을 원래 DataFrame에 업데이트
    df_filtered.loc[:, continuous_features] = logit_cont

    # 이진 변수를 정수형(0 또는 1)으로 변환
    df_filtered.loc[:, binary_features] = df_filtered[binary_features].astype(int)

    # 전체 특징(feature) 목록 생성 (연속형 + 이진형)
    features = continuous_features + binary_features

    # 정규화 및 변환 후 데이터 정보 출력
    print("Data after normalization and transformation:")
    # 변환된 특징들과 Stock_ID 상위 5개 행 출력
    display(df_filtered[features + ['Stock_ID']].head())
else:
    # 데이터가 비어 있으면 정규화/변환 건너뜀
    print("Skipping normalization and transformation as df_filtered is empty.")
```

- `continuous_features`, `binary_features`: 모델에서 사용할 연속형 변수와 이진형 변수의 이름을 리스트로 정의합니다.
- `if not df_filtered.empty:`: 데이터가 있을 때만 아래 전처리를 수행합니다.
    - `df_filtered.loc[:, "Stock_ID"] = ...`: 기업명을 나타내는 'Stock' 컬럼을 범주형으로 바꾸고, 각 기업에 고유한 정수 ID('Stock_ID')를 할당합니다. 모델은 숫자로 된 ID를 사용합니다.
    - `minmax_scaler = MinMaxScaler()`: Scikit-learn의 MinMaxScaler를 생성합니다. 이 스케일러는 데이터를 0과 1 사이의 값으로 변환합니다.
    - `scaled_cont = minmax_scaler.fit_transform(...)`: 연속형 변수들에 Min-Max 스케일링을 적용합니다. `fit_transform`은 스케일러를 데이터에 맞게 학습시키고 동시에 데이터를 변환합니다.
    - `EPS = 1e-6`: Logit 변환 시 log(0)이나 0으로 나누는 것을 방지하기 위한 아주 작은 값입니다.
    - `scaled_cont = np.clip(...)`: 스케일링된 값을 `EPS`와 `1.0-EPS` 사이로 제한합니다.
    - `logit_cont = np.log(...)`: Logit 변환(로지스틱 함수의 역함수)을 적용합니다. 이는 (0, 1) 범위의 값을 실수 전체 범위로 매핑하여 모델이 더 잘 학습하도록 도울 수 있습니다.
    - `df_filtered.loc[:, continuous_features] = logit_cont`: 변환된 값을 다시 DataFrame에 저장합니다.
    - `df_filtered.loc[:, binary_features] = ...`: 이진 변수(True/False 또는 다른 값)를 0과 1의 정수형으로 변환합니다.
    - `features = ...`: 모든 변수 이름을 하나의 리스트로 합칩니다.
    - `print(...)`, `display(...)`: 변환 후의 데이터를 확인합니다.
- `else:`: 데이터가 없으면 건너뛴다는 메시지를 출력합니다.

## 3. 기업 단위 시퀀스 데이터 생성 (각 기업 13년치)

```python
# 필터링된 데이터가 비어 있지 않은 경우에만 실행
if not df_filtered.empty:
    # 고유한 기업(Stock) 목록 가져오기
    stocks = df_filtered["Stock"].unique()
    # 각 기업별 시퀀스 데이터를 저장할 리스트 초기화
    grouped_cont = []  # 연속형 변수 시퀀스
    grouped_bin = []   # 이진 변수 시퀀스
    grouped_year = []  # 연도 시퀀스
    grouped_stock = [] # 기업 ID (스칼라 값)

    # 각 기업별로 반복 처리
    for stock_val in stocks:
        # 현재 기업(stock_val)의 데이터만 선택하고 연도순으로 정렬
        df_stock = df_filtered[df_filtered["Stock"] == stock_val].sort_values(by="YEAR")
        # 해당 기업의 연속형 변수 값들을 NumPy 배열로 변환하여 리스트에 추가 (num_years, num_continuous_features) 형태
        grouped_cont.append(df_stock[continuous_features].values)
        # 해당 기업의 이진 변수 값들을 NumPy 배열로 변환하여 리스트에 추가 (num_years, num_binary_features) 형태
        grouped_bin.append(df_stock[binary_features].values)
        # 해당 기업의 연도 값들을 NumPy 배열로 변환하여 리스트에 추가 (num_years,) 형태
        grouped_year.append(df_stock["YEAR"].values)
        # 해당 기업의 Stock_ID (모든 연도에서 동일)를 리스트에 추가 (스칼라 값)
        grouped_stock.append(df_stock["Stock_ID"].iloc[0])

    # 리스트에 저장된 각 기업별 NumPy 배열들을 쌓아서 하나의 큰 텐서로 만듦
    # X_cont_seq: (num_stocks, num_years, num_continuous_features) 형태의 텐서
    X_cont_seq = np.stack(grouped_cont, axis=0)
    # X_bin_seq: (num_stocks, num_years, num_binary_features) 형태의 텐서
    X_bin_seq = np.stack(grouped_bin, axis=0)
    # year_seq: (num_stocks, num_years) 형태의 텐서
    year_seq = np.stack(grouped_year, axis=0)
    # stock_seq: (num_stocks,) 형태의 배열 (각 기업 ID)
    stock_seq = np.array(grouped_stock)

    # 목표(Target) 시퀀스 생성: 연속형과 이진형 변수를 마지막 차원에서 결합
    # target_seq: (num_stocks, num_years, num_features) 형태
    target_seq = np.concatenate([X_cont_seq, X_bin_seq], axis=-1)

    # NumPy 배열들을 PyTorch 텐서로 변환
    X_cont_tensor = torch.tensor(X_cont_seq, dtype=torch.float32) # 연속형 변수 텐서
    X_bin_tensor = torch.tensor(X_bin_seq, dtype=torch.float32)   # 이진 변수 텐서
    year_tensor_seq = torch.tensor(year_seq, dtype=torch.float32) # 연도 시퀀스 텐서
    stock_tensor_seq = torch.tensor(stock_seq, dtype=torch.long)  # 기업 ID 텐서 (Embedding 레이어 입력 위해 long 타입)
    target_tensor_seq = torch.tensor(target_seq, dtype=torch.float32) # 목표 변수 텐서

    # PyTorch TensorDataset 생성: 여러 텐서들을 하나의 데이터셋으로 묶음
    dataset_seq = TensorDataset(X_cont_tensor, X_bin_tensor, year_tensor_seq, stock_tensor_seq, target_tensor_seq)
    # PyTorch DataLoader 생성: 데이터셋을 배치 단위로 묶고 셔플링하는 기능 제공
    dataloader_seq = DataLoader(dataset_seq, batch_size=BATCH_SIZE, shuffle=True)

    # 생성된 텐서들의 형태(shape) 출력
    print(f"X_cont_tensor shape: {X_cont_tensor.shape}")
    print(f"X_bin_tensor shape: {X_bin_tensor.shape}")
    print(f"year_tensor_seq shape: {year_tensor_seq.shape}")
    print(f"stock_tensor_seq shape: {stock_tensor_seq.shape}")
    print(f"target_tensor_seq shape: {target_tensor_seq.shape}")
    # 최종 필터링된 고유 기업 수 저장 (Stock Embedding 레이어 크기 결정에 사용)
    NUM_STOCK_EMBEDDINGS = df_filtered["Stock_ID"].nunique()

else:
    # 데이터가 비어 있으면 시퀀스 생성 건너뜀
    print("Skipping sequence data generation as df_filtered is empty.")
    # 빈 텐서 또는 적절한 기본값으로 초기화 (이후 코드 오류 방지)
    X_cont_tensor = torch.empty(0)
    X_bin_tensor = torch.empty(0)
    year_tensor_seq = torch.empty(0)
    stock_tensor_seq = torch.empty(0)
    target_tensor_seq = torch.empty(0)
    dataloader_seq = [] # 빈 DataLoader 또는 리스트
    NUM_STOCK_EMBEDDINGS = 0 # 기업 수 0으로 설정
```

- 이 셀은 전처리된 데이터를 기업별 시계열 데이터 형태로 변환하고 PyTorch 모델 학습에 사용할 수 있도록 텐서 및 DataLoader를 생성합니다.
- `if not df_filtered.empty:`: 데이터가 있을 때만 실행합니다.
    - `stocks = df_filtered["Stock"].unique()`: 고유한 기업 목록을 얻습니다.
    - `grouped_... = []`: 각 기업의 연속형, 이진형, 연도, 기업 ID 데이터를 임시로 저장할 리스트를 만듭니다.
    - `for stock_val in stocks:`: 각 기업에 대해 반복합니다.
        - `df_stock = ...`: 해당 기업의 데이터만 가져와 연도순으로 정렬합니다.
        - `grouped_cont.append(...)`, `grouped_bin.append(...)`, `grouped_year.append(...)`: 각 변수 타입별로 NumPy 배열 형태로 값을 추출하여 리스트에 추가합니다.
        - `grouped_stock.append(...)`: 기업 ID를 리스트에 추가합니다.
    - `np.stack(...)`: 각 리스트에 저장된 기업별 NumPy 배열들을 쌓아 하나의 큰 배열로 만듭니다. `axis=0`은 기업 차원을 추가합니다.
    - `target_seq = np.concatenate(...)`: 연속형과 이진형 변수 배열을 마지막 차원(특징 차원)에서 합쳐 목표(target) 데이터를 만듭니다.
    - `torch.tensor(...)`: NumPy 배열들을 PyTorch 텐서로 변환합니다. 데이터 타입(`dtype`)을 적절히 지정합니다 (실수형은 `float32`, ID는 `long`).
    - `dataset_seq = TensorDataset(...)`: 생성된 텐서들을 묶어 PyTorch 데이터셋 객체를 만듭니다.
    - `dataloader_seq = DataLoader(...)`: 데이터셋을 배치 크기(`BATCH_SIZE`)로 나누고, 학습 시 데이터를 섞어주는(`shuffle=True`) DataLoader를 생성합니다.
    - `print(...)`: 생성된 텐서들의 최종 형태를 출력하여 확인합니다.
    - `NUM_STOCK_EMBEDDINGS = ...`: 고유한 기업 ID의 개수를 세어 저장합니다. 이는 나중에 Stock Embedding 레이어의 크기를 결정하는 데 사용됩니다.
- `else:`: 데이터가 없으면 건너뛰고 빈 텐서 등으로 초기화합니다.

## 4. sine-cosine 기반 연도 임베딩 함수 (sequence 지원)

```python
# 연도를 입력받아 Sine-Cosine 임베딩 벡터를 반환하는 함수 정의
def get_sine_cosine_year_embedding(years, dim=YEAR_DIM_PARAM):
    """
    주어진 연도 값을 Sine-Cosine 방식을 사용하여 고정된 차원의 임베딩 벡터로 변환합니다.
    Transformer의 Positional Encoding과 유사한 원리입니다.

    Args:
        years (torch.Tensor): 실제 연도 값 텐서. (batch, num_years) 또는 (num_samples,) 형태.
        dim (int): 생성될 임베딩 벡터의 차원. 기본값은 YEAR_DIM_PARAM.

    Returns:
        torch.Tensor: 연도 임베딩 텐서. 입력 `years`의 마지막 차원이 `dim`으로 바뀐 형태.
                       (..., dim) 형태.
    """
    # 입력 years 텐서의 차원이 3보다 작으면 마지막에 차원 추가 (연산 일관성)
    # 예: (batch, num_years) -> (batch, num_years, 1)
    if len(years.shape) < 3:
        years = years.unsqueeze(-1)

    # 임베딩 차원의 절반 계산 (Sine과 Cosine에 각각 할당)
    half_dim = dim // 2
    # 임베딩 차원의 각 위치(짝수 인덱스)에 대한 주파수 계산 (Transformer 공식 기반)
    freqs = torch.exp(
        torch.arange(0, half_dim, dtype=torch.float32) * (-np.log(10000.0) / half_dim)
    ).to(years.device) # freqs 텐서를 years와 같은 디바이스로 이동

    # 각 연도 값에 주파수를 곱하여 Sine/Cosine 함수의 입력값 생성 (브로드캐스팅 활용)
    sinusoidal_input = years * freqs

    # Sine 함수와 Cosine 함수를 적용하여 임베딩의 각 부분 계산
    sin_embed = torch.sin(sinusoidal_input)
    cos_embed = torch.cos(sinusoidal_input)

    # Sine 임베딩과 Cosine 임베딩을 마지막 차원에서 결합
    year_embedding = torch.cat([sin_embed, cos_embed], dim=-1)

    # 만약 목표 차원(dim)이 홀수여서 현재 임베딩 차원이 dim보다 작으면
    if year_embedding.shape[-1] < dim:
        # 부족한 차원 크기 계산 (보통 1)
        pad_size = dim - year_embedding.shape[-1]
        # 0으로 채워진 패딩 텐서 생성
        padding = torch.zeros(year_embedding.shape[:-1] + (pad_size,), device=year_embedding.device)
        # 기존 임베딩 뒤에 패딩을 붙여 목표 차원(dim) 맞춤
        year_embedding = torch.cat([year_embedding, padding], dim=-1)

    # 생성된 연도 임베딩 텐서의 형태 출력 (디버깅용)
    print(f"Year embedding shape: {year_embedding.shape}")

    # 최종 연도 임베딩 텐서 반환
    return year_embedding
```

- 이 함수는 Transformer의 위치 인코딩 아이디어를 차용하여, 연도(시간 정보)를 고정된 크기의 벡터로 변환합니다.
- `if len(years.shape) < 3:`: 입력 `years`의 차원을 최소 3으로 맞춰줍니다.
- `half_dim = dim // 2`: 임베딩 차원을 반으로 나눕니다.
- `freqs = torch.exp(...)`: 다양한 주파수를 계산합니다. 위치 인코딩 공식과 동일한 방식을 사용합니다.
- `sinusoidal_input = years * freqs`: 연도 값과 주파수를 곱합니다.
- `sin_embed = torch.sin(...)`, `cos_embed = torch.cos(...)`: 사인과 코사인 값을 계산합니다.
- `year_embedding = torch.cat(...)`: 사인과 코사인 결과를 합쳐 임베딩 벡터를 만듭니다.
- `if year_embedding.shape[-1] < dim:`: 만약 `dim`이 홀수라서 벡터 크기가 `dim`보다 1 작으면, 0으로 패딩하여 크기를 맞춰줍니다.
- `print(...)`: 생성된 임베딩의 형태를 출력합니다.
- `return year_embedding`: 최종 연도 임베딩 벡터를 반환합니다.

## 5. Positional Encoding (Sinusoidal)

```python
# 시퀀스 내 위치 정보를 인코딩하는 PositionalEncoding 클래스 정의
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Sinusoidal Positional Encoding을 계산하고 적용하는 클래스.
        Transformer 모델에서 토큰의 순서 정보를 모델에 알려주기 위해 사용됩니다.

        Args:
            d_model (int): 임베딩 벡터의 차원.
            max_len (int): 처리할 수 있는 최대 시퀀스 길이. 기본값 5000.
        """
        # 부모 클래스(nn.Module)의 생성자 호출
        super(PositionalEncoding, self).__init__()

        # 위치 인코딩 값을 저장할 (max_len, d_model) 크기의 0으로 채워진 텐서 생성
        pe = torch.zeros(max_len, d_model)
        # 0부터 max_len-1까지의 위치 인덱스 생성 후 차원 추가 -> (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 위치 인코딩 계산에 사용될 분모 항(div_term) 계산 (짝수 인덱스 기준)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # pe 텐서의 짝수 열(0, 2, 4, ...)에 sin(position * div_term) 값 저장
        pe[:, 0::2] = torch.sin(position * div_term)
        # d_model이 홀수인 경우, 마지막 홀수 열은 계산된 코사인 값에서 제외해야 함
        if d_model % 2 == 1:
            # pe 텐서의 홀수 열(1, 3, 5, ...)에 cos(position * div_term) 값 저장 (필요한 만큼만 슬라이싱)
            pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
        # d_model이 짝수인 경우
        else:
            # pe 텐서의 홀수 열(1, 3, 5, ...)에 cos(position * div_term) 값 저장
            pe[:, 1::2] = torch.cos(position * div_term)

        # 배치 차원 추가: (max_len, d_model) -> (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # 계산된 위치 인코딩 텐서 'pe'를 모델의 버퍼로 등록 (학습되지 않음)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        입력 텐서 x에 위치 인코딩을 더하여 반환합니다.

        Args:
            x (torch.Tensor): 입력 텐서. (batch_size, seq_len, d_model) 형태.

        Returns:
            torch.Tensor: 위치 인코딩이 더해진 텐서. (batch_size, seq_len, d_model) 형태.
        """
        # 입력 텐서 x의 시퀀스 길이(seq_len) 가져오기
        seq_len = x.size(1)
        # 입력 텐서 x에 미리 계산된 위치 인코딩 값(self.pe)을 더함
        # self.pe에서 실제 시퀀스 길이만큼만 슬라이싱하여 사용 (브로드캐스팅 활용)
        x = x + self.pe[:, :seq_len]
        # 위치 인코딩이 더해진 텐서 반환
        return x
```

- 이 클래스는 Transformer 모델에서 시퀀스 내 토큰의 순서(위치) 정보를 모델에 알려주기 위한 Positional Encoding을 구현합니다.
- `__init__`:
    - `d_model`: 입력 임베딩 및 Positional Encoding 벡터의 차원입니다.
    - `max_len`: 미리 계산해 둘 최대 시퀀스 길이를 지정합니다.
    - `pe = torch.zeros(...)`: Positional Encoding 값을 저장할 텐서를 0으로 초기화합니다.
    - `position = torch.arange(...)`: 0부터 `max_len-1`까지의 위치 인덱스를 생성합니다.
    - `div_term = torch.exp(...)`: Positional Encoding 공식에 사용되는 주파수 관련 항을 계산합니다.
    - `pe[:, 0::2] = torch.sin(...)`: `pe` 텐서의 짝수 열에 사인 값을 계산하여 저장합니다.
    - `if d_model % 2 == 1: ... else: ...`: `pe` 텐서의 홀수 열에 코사인 값을 계산하여 저장합니다. `d_model`이 홀수일 경우 차원을 맞추기 위해 슬라이싱합니다.
    - `pe = pe.unsqueeze(0)`: 배치 차원을 추가합니다.
    - `self.register_buffer('pe', pe)`: 계산된 `pe`를 모델의 버퍼로 등록하여 학습되지 않도록 하고 모델 저장/로드 시 포함되도록 합니다.
- `forward`:
    - `x`: 입력 텐서 (보통 단어 또는 특징 임베딩)입니다. `(batch_size, seq_len, d_model)` 형태입니다.
    - `seq_len = x.size(1)`: 입력의 실제 시퀀스 길이를 얻습니다.
    - `x = x + self.pe[:, :seq_len]`: 입력 `x`에 미리 계산된 Positional Encoding 값(`self.pe`) 중 실제 시퀀스 길이에 해당하는 부분만 더해줍니다.
    - `return x`: 위치 정보가 추가된 텐서를 반환합니다.

## 6. CompanySequenceModel: FT-Transformer + tst (Sequence 모델)

```python
# 기업 시계열 데이터를 처리하는 메인 모델 클래스 정의
class CompanySequenceModel(nn.Module):
    def __init__(self, cont_input_dim, bin_input_dim, year_dim, num_stock_embeddings, stock_dim=STOCK_DIM_PARAM, ft_out_dim=FT_OUT_DIM_PARAM, num_years=NUM_YEARS_PARAM):
        """
        기업의 시계열 재무 데이터를 입력받아 처리하는 모델.
        FT-Transformer와 Transformer Encoder (TST)를 결합한 구조.

        Args:
            cont_input_dim (int): 연속형 입력 특징의 수.
            bin_input_dim (int): 이진형 입력 특징의 수.
            year_dim (int): 연도 임베딩의 차원.
            num_stock_embeddings (int): 고유한 기업(Stock) ID의 개수 (Stock Embedding 레이어 크기).
            stock_dim (int): 기업 ID 임베딩의 차원. 기본값 STOCK_DIM_PARAM.
            ft_out_dim (int): FT-Transformer의 출력 차원. 기본값 FT_OUT_DIM_PARAM.
            num_years (int): 시퀀스 길이 (연도 수). 기본값 NUM_YEARS_PARAM.
        """
        super(CompanySequenceModel, self).__init__()
        self.num_years = num_years # 시퀀스 길이 저장

        # 입력 임베딩 레이어 정의
        # 연속형 변수를 32차원으로 임베딩하는 선형 레이어
        self.cont_embedding = nn.Linear(cont_input_dim, 32)
        # 이진형 변수를 16차원으로 임베딩하는 선형 레이어
        self.bin_embedding = nn.Linear(bin_input_dim, 16)
        # 기업 ID를 stock_dim 차원으로 임베딩하는 Embedding 레이어
        self.stock_embedding = nn.Embedding(num_embeddings=num_stock_embeddings, embedding_dim=stock_dim)

        # 모든 임베딩(연속형, 이진형, 연도, 기업ID)을 합친 후의 총 입력 차원 계산
        total_input_dim = 32 + 16 + year_dim + stock_dim
        # 합쳐진 임베딩을 128차원으로 변환하는 선형 레이어
        self.embedding = nn.Linear(total_input_dim, 128)
        # 배치 정규화(Batch Normalization) 레이어 (128차원)
        self.bn = nn.BatchNorm1d(128) # 1D BatchNorm은 (N, C) 또는 (N, C, L) 형태를 받음. 여기서는 (N*L, C) 형태로 적용될 것임.

        # FT-Transformer 모델 생성 (rtdl 라이브러리 활용)
        self.ft_transformer = rtdl.FTTransformer.make_default(
            n_num_features=128,       # 숫자형 특징의 수 (여기서는 통합된 임베딩 차원)
            cat_cardinalities=None,   # 범주형 특징 정보 (여기서는 사용 안 함)
            d_out=ft_out_dim          # 최종 출력 차원
        )

        # 1D Convolutional 레이어 (FT-Transformer 출력에 적용)
        # 채널 수(ft_out_dim)는 유지, 커널 크기 3, 패딩 1 (시퀀스 길이 유지)
        self.conv1d = nn.Conv1d(in_channels=ft_out_dim, out_channels=ft_out_dim, kernel_size=3, padding=1)
        # Positional Encoding 레이어 (Conv1D 출력에 적용)
        self.pos_encoder = PositionalEncoding(ft_out_dim, max_len=num_years)
        # Transformer Encoder Layer 정의 (Multi-Head Attention 등 포함)
        # d_model: 입력/출력 차원, nhead: 어텐션 헤드 수, dropout: 드롭아웃 비율, batch_first=True: 입력 형태 (batch, seq, feature)
        encoder_layer = nn.TransformerEncoderLayer(d_model=ft_out_dim, nhead=2, dropout=0.1, batch_first=True)
        # 여러 개의 Transformer Encoder Layer를 쌓은 Transformer Encoder 정의
        self.tst_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x_cont, x_bin, year_values, stock_id):
        """
        모델의 순전파 로직 정의.

        Args:
            x_cont (torch.Tensor): 연속형 변수 입력 텐서. (batch, num_years, cont_input_dim)
            x_bin (torch.Tensor): 이진형 변수 입력 텐서. (batch, num_years, bin_input_dim)
            year_values (torch.Tensor): 연도 값 텐서. (batch, num_years)
            stock_id (torch.Tensor): 기업 ID 텐서. (batch,)

        Returns:
            torch.Tensor: 모델의 최종 출력 텐서. (batch, num_years, ft_out_dim)
        """
        # 입력 텐서에서 배치 크기(batch)와 시퀀스 길이(num_years) 가져오기
        batch, num_years, _ = x_cont.shape
        # 연도 값을 임베딩 벡터로 변환 (이전에 정의한 함수 사용)
        # year_embed: (batch, num_years, year_dim)
        year_embed = get_sine_cosine_year_embedding(year_values, dim=YEAR_DIM_PARAM)

        # 각 입력 변수 임베딩
        # x_cont를 (batch * num_years, cont_input_dim) 형태로 펼쳐서 임베딩
        cont_emb = self.cont_embedding(x_cont.reshape(-1, x_cont.shape[-1]))
        # x_bin을 (batch * num_years, bin_input_dim) 형태로 펼쳐서 임베딩
        bin_emb = self.bin_embedding(x_bin.reshape(-1, x_bin.shape[-1]))

        # 기업 ID 임베딩
        # stock_id (batch,) -> stock_emb (batch, stock_dim)
        stock_emb = self.stock_embedding(stock_id)
        # 시퀀스 길이에 맞춰 복제: (batch, stock_dim) -> (batch, 1, stock_dim) -> (batch, num_years, stock_dim)
        stock_emb = stock_emb.unsqueeze(1).repeat(1, num_years, 1)

        # 모든 임베딩 벡터들을 마지막 차원에서 결합
        # 각 임베딩을 (batch * num_years, dim) 형태로 펼쳐서 결합
        x_all = torch.cat([
            cont_emb,                                       # (batch*num_years, 32)
            bin_emb,                                        # (batch*num_years, 16)
            year_embed.reshape(-1, year_embed.shape[-1]),   # (batch*num_years, year_dim)
            stock_emb.reshape(-1, stock_emb.shape[-1])      # (batch*num_years, stock_dim)
        ], dim=-1) # 결과: (batch*num_years, total_input_dim)

        # 결합된 임베딩을 128차원으로 변환
        x_all = self.embedding(x_all) # 결과: (batch*num_years, 128)
        # 배치 정규화 적용 (평균 0, 분산 1로 정규화하여 학습 안정화)
        x_all = self.bn(x_all) # 결과: (batch*num_years, 128)

        # FT-Transformer 통과 (x_num에 숫자형 특징 입력, x_cat은 없음)
        # 입력: (batch*num_years, 128), 출력: (batch*num_years, ft_out_dim)
        ft_out = self.ft_transformer(x_num=x_all, x_cat=None)
        # 출력을 원래 시퀀스 형태로 복원: (batch, num_years, ft_out_dim)
        ft_out = ft_out.view(batch, num_years, -1)

        # 1D Convolution 적용 (시계열 특징 추출 강화)
        # Conv1D는 (batch, channels, length) 형태를 입력으로 받으므로 차원 순서 변경
        conv_in = ft_out.transpose(1, 2) # (batch, ft_out_dim, num_years)
        conv_out = self.conv1d(conv_in)  # (batch, ft_out_dim, num_years)
        # 다시 원래 차원 순서로 복원
        conv_out = conv_out.transpose(1, 2) # (batch, num_years, ft_out_dim)

        # Positional Encoding 적용 (시간 순서 정보 추가)
        tst_input = self.pos_encoder(conv_out) # (batch, num_years, ft_out_dim)
        # Transformer Encoder 통과 (시퀀스 내 관계 학습)
        tst_output = self.tst_encoder(tst_input) # (batch, num_years, ft_out_dim)

        # 최종 출력 반환
        return tst_output
```

- 이 클래스는 기업의 시계열 데이터를 입력받아 처리하는 메인 모델입니다. FT-Transformer와 Transformer Encoder(TST) 구조를 결합했습니다.
- `__init__`:
    - 모델의 구성 요소(레이어)들을 초기화합니다.
    - `cont_embedding`, `bin_embedding`, `stock_embedding`: 각 입력 타입(연속형, 이진형, 기업 ID)에 대한 임베딩 레이어입니다.
    - `total_input_dim`: 모든 임베딩 벡터를 합쳤을 때의 총 차원을 계산합니다.
    - `embedding`: 합쳐진 임베딩을 고정된 크기(128)로 변환하는 레이어입니다.
    - `bn`: 배치 정규화 레이어로, 학습을 안정화시킵니다.
    - `ft_transformer`: `rtdl` 라이브러리의 FT-Transformer 모델을 사용합니다. 테이블 형태 데이터 처리에 강점이 있습니다. 여기서는 각 시간 스텝의 특징들을 독립적으로 처리하는 데 사용됩니다.
    - `conv1d`: 1D Convolution 레이어는 시계열 데이터에서 지역적인 패턴을 추출하는 데 도움을 줄 수 있습니다.
    - `pos_encoder`: 이전에 정의한 `PositionalEncoding` 클래스의 인스턴스입니다.
    - `encoder_layer`, `tst_encoder`: 표준 Transformer Encoder 구조입니다. 시퀀스 내의 시간적 의존성을 학습합니다.
- `forward`:
    - 모델의 입력 데이터를 받아 최종 출력을 계산하는 과정을 정의합니다.
    - `batch, num_years, _ = x_cont.shape`: 입력 데이터의 배치 크기와 시퀀스 길이를 얻습니다.
    - `year_embed = get_sine_cosine_year_embedding(...)`: 연도 값을 임베딩합니다.
    - `cont_emb = self.cont_embedding(...)`, `bin_emb = self.bin_embedding(...)`: 연속형/이진형 변수를 임베딩합니다. `reshape(-1, ...)`는 배치와 시퀀스 차원을 하나로 합쳐 각 시간 스텝별로 임베딩을 적용하기 위함입니다.
    - `stock_emb = self.stock_embedding(...)`: 기업 ID를 임베딩합니다.
    - `stock_emb = stock_emb.unsqueeze(1).repeat(...)`: 기업 ID 임베딩을 시퀀스 길이에 맞게 복제합니다. (모든 시간 스텝에서 동일한 기업 ID 임베딩 사용)
    - `x_all = torch.cat(...)`: 모든 임베딩 벡터를 합칩니다. `reshape(-1, ...)`로 펼쳐진 벡터들을 합칩니다.
    - `x_all = self.embedding(x_all)`: 합쳐진 벡터를 128차원으로 변환합니다.
    - `x_all = self.bn(x_all)`: 배치 정규화를 적용합니다.
    - `ft_out = self.ft_transformer(...)`: FT-Transformer를 통과시킵니다.
    - `ft_out = ft_out.view(...)`: FT-Transformer의 출력을 다시 `(batch, num_years, ft_out_dim)` 형태로 복원합니다.
    - `conv_in = ft_out.transpose(1, 2)`, `conv_out = self.conv1d(conv_in)`, `conv_out = conv_out.transpose(1, 2)`: 1D Convolution을 적용하기 위해 차원을 바꾸고, 적용 후 다시 원래대로 복원합니다.
    - `tst_input = self.pos_encoder(conv_out)`: Positional Encoding을 추가합니다.
    - `tst_output = self.tst_encoder(tst_input)`: Transformer Encoder를 통과시켜 최종 출력을 얻습니다.
    - `return tst_output`: 모델의 최종 출력을 반환합니다.

## 7. CompanySequenceModel 학습 (FT-Transformer + Tst)

### 7.1 모델 파라미터 및 초기화

```python
# 입력 차원 설정
cont_input_dim = len(continuous_features) # 연속형 변수 개수
bin_input_dim = len(binary_features)      # 이진 변수 개수

# 필터링된 데이터가 비어 있지 않은 경우에만 실행
if not df_filtered.empty:
    # 이진 변수(BIG4, LOSS) 각각의 양성(1) 비율 계산
    p_big4 = (df_filtered['BIG4'] == 1).mean()
    p_loss = (df_filtered['LOSS'] == 1).mean()
    # 수치 안정을 위해 비율 값을 EPS와 1-EPS 사이로 제한
    p_big4 = np.clip(p_big4, EPS, 1.0 - EPS)
    p_loss = np.clip(p_loss, EPS, 1.0 - EPS)
    # BCE 손실 함수에서 사용할 가중치(pos_weight) 계산 (클래스 불균형 완화 목적)
    # 양성 클래스 비율이 낮을수록 높은 가중치를 부여
    pos_w  = torch.tensor([
        ((1-p_big4)/p_big4)**0.5, # BIG4 가중치
        ((1-p_loss)/p_loss)**0.5  # LOSS 가중치
    ], device=device) # 계산된 가중치를 지정된 디바이스로 이동
else:
    # 데이터가 없으면 기본 가중치(1) 사용
    pos_w = torch.ones(2, device=device)

# 손실 함수 정의
# 이진 변수용: BCEWithLogitsLoss (Sigmoid + BCE), pos_weight 적용
bce_bin   = nn.BCEWithLogitsLoss(pos_weight=pos_w)
# 연속형 변수용: Mean Squared Error (MSE) Loss
mse_cont  = nn.MSELoss()
# 이진 변수 손실에 곱해줄 가중치 람다 값
λ_bin_enc = 10.0

# 고유 기업 수가 0보다 클 때 (즉, 학습할 데이터가 있을 때) 모델 초기화
if NUM_STOCK_EMBEDDINGS > 0:
    # CompanySequenceModel 인스턴스 생성 및 지정된 디바이스로 이동
    company_model = CompanySequenceModel(
        cont_input_dim, bin_input_dim, YEAR_DIM_PARAM,
        num_stock_embeddings=NUM_STOCK_EMBEDDINGS, # 계산된 고유 기업 수 전달
        stock_dim=STOCK_DIM_PARAM,
        ft_out_dim=FT_OUT_DIM_PARAM,
        num_years=NUM_YEARS_PARAM
    ).to(device)
    # Adam 옵티마이저 생성 (모델 파라미터와 학습률 전달)
    optimizer_company = optim.Adam(company_model.parameters(), lr=LEARNING_RATE_COMPANY)
    # TensorBoard 로깅을 위한 SummaryWriter 생성
    # 로그 디렉토리 이름에 현재 시간을 포함하여 실행마다 고유하게 만듦
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    writer_company = SummaryWriter(f'{TENSORBOARD_LOG_DIR_COMPANY}_{current_time_str}')
else:
    # 학습할 데이터가 없으면 모델 관련 변수들을 None으로 설정
    print("Skipping CompanySequenceModel initialization as there are no stocks after filtering.")
    company_model = None
    optimizer_company = None
    writer_company = None
```

- 이 셀은 `CompanySequenceModel` 학습을 위한 준비 단계입니다. 손실 함수, 옵티마이저, TensorBoard 로거 등을 설정합니다.
- `cont_input_dim`, `bin_input_dim`: 이전 셀에서 정의된 변수 리스트의 길이를 사용하여 입력 차원을 설정합니다.
- `if not df_filtered.empty:`: 데이터가 있을 때만 실행합니다.
    - `p_big4 = ...`, `p_loss = ...`: 이진 변수 'BIG4'와 'LOSS' 각각에서 값이 1인 비율(양성 클래스 비율)을 계산합니다.
    - `np.clip(...)`: 계산된 비율이 0 또는 1이 되지 않도록 작은 값(EPS) 범위 내로 제한합니다.
    - `pos_w = torch.tensor(...)`: 클래스 불균형 문제를 완화하기 위해 BCE 손실 함수에 적용할 `pos_weight`를 계산합니다. 양성 클래스 비율이 낮을수록 해당 클래스에 대한 손실에 더 높은 가중치를 부여합니다. `sqrt((1-p)/p)`는 일반적인 가중치 계산 방식 중 하나입니다.
- `else:`: 데이터가 없으면 기본 가중치 1을 사용합니다.
- `bce_bin = nn.BCEWithLogitsLoss(...)`: 이진 변수 예측을 위한 손실 함수입니다. 모델 출력(logit)에 시그모이드 함수를 적용한 후 Binary Cross Entropy를 계산하며, `pos_weight`를 사용하여 클래스 불균형을 조절합니다.
- `mse_cont = nn.MSELoss()`: 연속형 변수 예측을 위한 손실 함수로, 평균 제곱 오차를 사용합니다.
- `λ_bin_enc = 10.0`: 전체 손실 계산 시 이진 변수 손실에 곱해줄 가중치입니다. 이진 변수 학습의 중요도를 조절합니다.
- `if NUM_STOCK_EMBEDDINGS > 0:`: 학습할 기업 데이터가 있을 때만 모델과 옵티마이저 등을 초기화합니다.
    - `company_model = CompanySequenceModel(...)`: 정의된 `CompanySequenceModel` 클래스의 인스턴스를 생성하고 `.to(device)`를 통해 지정된 디바이스(GPU 또는 CPU)로 모델을 이동시킵니다.
    - `optimizer_company = optim.Adam(...)`: Adam 옵티마이저를 생성합니다. 모델의 학습 가능한 파라미터들과 설정된 학습률(`LEARNING_RATE_COMPANY`)을 전달합니다.
    - `current_time_str = time.strftime(...)`: 현재 시간을 문자열로 포맷팅합니다.
    - `writer_company = SummaryWriter(...)`: TensorBoard 로깅을 위한 `SummaryWriter` 객체를 생성합니다. 로그 파일이 저장될 디렉토리 이름을 현재 시간을 포함하여 고유하게 만듭니다.
- `else:`: 학습할 데이터가 없으면 관련 변수들을 `None`으로 설정하고 메시지를 출력합니다.

### 7.2 학습 루프

```python
# company_model과 dataloader_seq가 정상적으로 초기화되었는지 확인
if company_model and dataloader_seq:
    # TensorBoard writer가 초기화되었는지 확인
    if writer_company:
        try:
            # DataLoader에서 첫 번째 배치를 가져와 모델 그래프 로깅 시도
            data_iter_company = iter(dataloader_seq) # DataLoader의 iterator 생성
            # 첫 배치 데이터 언패킹 (타겟 데이터는 사용 안 함 '_')
            sample_cont_company, sample_bin_company, sample_year_company, sample_stock_company, _ = next(data_iter_company)
            # TensorBoard에 모델 구조(그래프) 기록 (모델과 샘플 입력 전달)
            writer_company.add_graph(company_model, [sample_cont_company.to(device), sample_bin_company.to(device), sample_year_company.to(device), sample_stock_company.to(device)])
            # 사용한 iterator 삭제 (메모리 관리)
            del data_iter_company
        except Exception as e:
            # 그래프 로깅 중 오류 발생 시 메시지 출력
            print(f"Error adding CompanySequenceModel graph to TensorBoard: {e}")

    # 지정된 에포크 수만큼 학습 반복
    for epoch in range(EPOCHS_COMPANY_MODEL):
        # 에포크별 손실 초기화
        epoch_loss_cont = 0.0 # 연속형 변수 손실 합계
        epoch_loss_bin = 0.0  # 이진 변수 손실 합계
        num_batches = 0       # 처리된 배치 수

        # DataLoader에서 배치 단위로 데이터 가져와 반복 처리
        for batch_cont, batch_bin, batch_year, batch_stock, batch_target in dataloader_seq:
            # 현재 배치의 텐서들을 지정된 디바이스로 이동
            batch_cont  = batch_cont.to(device)
            batch_bin   = batch_bin.to(device)
            batch_year  = batch_year.to(device)
            batch_stock = batch_stock.to(device)
            batch_target= batch_target.to(device) # 목표(정답) 텐서

            # 옵티마이저의 그래디언트 초기화 (이전 배치의 그래디언트 제거)
            optimizer_company.zero_grad()
            # 모델의 forward 함수 호출하여 예측값(pred) 계산
            pred = company_model(batch_cont, batch_bin, batch_year, batch_stock)

            # 예측값(pred)과 목표값(batch_target)을 연속형/이진형으로 분리
            # pred: (batch, num_years, ft_out_dim) -> 분리 후 각각 (batch, num_years, cont_input_dim), (batch, num_years, bin_input_dim)
            pred_cont, pred_bin = pred[:, :, :cont_input_dim], pred[:, :, cont_input_dim:]
            # batch_target: (batch, num_years, num_features) -> 분리 후 각각 (batch, num_years, cont_input_dim), (batch, num_years, bin_input_dim)
            tgt_cont , tgt_bin  = batch_target[:, :, :cont_input_dim], batch_target[:, :, cont_input_dim:]

            # 손실 계산
            loss_cont = mse_cont(pred_cont, tgt_cont) # 연속형 변수 MSE 손실
            loss_bin  = bce_bin(pred_bin, tgt_bin)    # 이진 변수 BCE 손실 (pos_weight 적용됨)
            # 최종 손실: 연속형 손실 + 가중치가 적용된 이진형 손실
            loss      = loss_cont + λ_bin_enc * loss_bin

            # 손실에 대한 그래디언트 계산 (역전파)
            loss.backward()

            # 특정 에포크마다 또는 마지막 에포크에 그래디언트 히스토그램 로깅
            if writer_company and (epoch % (EPOCHS_COMPANY_MODEL // 5 if EPOCHS_COMPANY_MODEL >=5 else 1) == 0 or epoch == EPOCHS_COMPANY_MODEL -1):
                # 모델의 각 파라미터에 대해 반복
                for name, param in company_model.named_parameters():
                    # 그래디언트가 존재하는 경우 (requires_grad=True인 파라미터)
                    if param.grad is not None:
                        # TensorBoard에 그래디언트 값 분포(히스토그램) 기록
                        writer_company.add_histogram(f'Gradients_Company/{name.replace(".", "/")}', param.grad, epoch)

            # 옵티마이저가 초기화되었는지 확인 후 파라미터 업데이트 수행
            if optimizer_company:
                optimizer_company.step()

            # 에포크 손실 누적
            epoch_loss_cont += loss_cont.item() # .item()으로 텐서에서 스칼라 값 추출
            epoch_loss_bin += loss_bin.item()
            # 배치 수 증가
            num_batches += 1

        # 에포크 평균 손실 계산
        avg_loss_cont = epoch_loss_cont / num_batches if num_batches > 0 else 0
        avg_loss_bin = epoch_loss_bin / num_batches if num_batches > 0 else 0
        # 가중치가 적용된 전체 평균 손실 계산
        total_avg_loss = avg_loss_cont + λ_bin_enc * avg_loss_bin

        # TensorBoard writer가 있으면 에포크별 평균 손실 로깅
        if writer_company:
            writer_company.add_scalar('Loss_Company/Continuous_Train', avg_loss_cont, epoch)
            writer_company.add_scalar('Loss_Company/Binary_Train', avg_loss_bin, epoch)
            writer_company.add_scalar('Loss_Company/Total_Train', total_avg_loss, epoch)

        # 특정 에포크마다 또는 마지막 에포크에 평균 손실 출력 및 가중치 히스토그램 로깅
        if epoch % (EPOCHS_COMPANY_MODEL // 5 if EPOCHS_COMPANY_MODEL >=5 else 1) == 0 or epoch == EPOCHS_COMPANY_MODEL -1:
            print(f"[CompanySequenceModel] Epoch {epoch:03d} | "
                  f"loss_cont={avg_loss_cont:.4f}  "
                  f"loss_bin={avg_loss_bin:.4f}")

            # TensorBoard writer가 있으면 모델 파라미터(가중치) 히스토그램 로깅
            if writer_company:
                for name, param in company_model.named_parameters():
                    # 학습 가능한 파라미터인 경우
                    if param.requires_grad:
                        # TensorBoard에 파라미터 값 분포(히스토그램) 기록
                        writer_company.add_histogram(f'Weights_Company/{name.replace(".", "/")}', param.data, epoch)

    # 학습 완료 후 TensorBoard writer 닫기
    if writer_company: writer_company.close()

else:
    # 모델 또는 데이터로더가 초기화되지 않았으면 학습 건너뜀
    print("Skipping CompanySequenceModel training as model or dataloader is not initialized.")

```

- 이 셀은 `CompanySequenceModel`의 실제 학습 과정을 수행합니다.
- `if company_model and dataloader_seq:`: 모델과 데이터로더가 준비되었는지 확인합니다.
    - `if writer_company:`: TensorBoard 로거가 준비되었는지 확인합니다.
        - `try...except`: 모델 그래프 로깅을 시도합니다. `iter(dataloader_seq)`로 이터레이터를 만들고 `next()`로 첫 배치를 가져와 `writer_company.add_graph()`에 전달하여 모델 구조를 TensorBoard에 기록합니다.
    - `for epoch in range(EPOCHS_COMPANY_MODEL):`: 설정된 에포크 수만큼 반복합니다.
        - `epoch_loss_cont = 0.0`, `epoch_loss_bin = 0.0`, `num_batches = 0`: 에포크별 손실과 배치 수를 초기화합니다.
        - `for batch_cont, ... in dataloader_seq:`: 데이터로더로부터 미니배치를 받아옵니다.
            - `batch_... = batch_....to(device)`: 배치 데이터를 지정된 디바이스로 옮깁니다.
            - `optimizer_company.zero_grad()`: 이전 배치의 그래디언트를 초기화합니다.
            - `pred = company_model(...)`: 모델에 입력을 넣어 예측값을 계산합니다 (순전파).
            - `pred_cont, pred_bin = ...`, `tgt_cont, tgt_bin = ...`: 예측값과 실제 목표값을 연속형과 이진형으로 분리합니다.
            - `loss_cont = mse_cont(...)`, `loss_bin = bce_bin(...)`: 각 변수 타입별 손실을 계산합니다.
            - `loss = loss_cont + λ_bin_enc * loss_bin`: 최종 손실을 계산합니다 (이진 손실에 가중치 적용).
            - `loss.backward()`: 손실에 대한 그래디언트를 계산합니다 (역전파).
            - `if writer_company and ...`: 특정 주기로 그래디언트 값의 분포를 TensorBoard에 기록합니다.
            - `if optimizer_company: optimizer_company.step()`: 옵티마이저를 사용하여 모델 파라미터를 업데이트합니다.
            - `epoch_loss_... += loss_....item()`: 에포크 손실을 누적합니다.
            - `num_batches += 1`: 배치 카운터를 증가시킵니다.
        - `avg_loss_... = ...`: 에포크의 평균 손실을 계산합니다.
        - `if writer_company:`: 계산된 평균 손실들을 TensorBoard에 기록합니다.
        - `if epoch % ...`: 특정 주기로 콘솔에 평균 손실을 출력하고, 모델 파라미터(가중치) 값의 분포를 TensorBoard에 기록합니다.
    - `if writer_company: writer_company.close()`: 모든 에포크가 끝나면 TensorBoard 로거를 닫습니다.
- `else:`: 모델이나 데이터로더가 없으면 학습을 건너뛴다는 메시지를 출력합니다.

### 7.3 학습 완료 후 결과 수집 및 평탄화

```python
# 학습된 모델의 출력을 저장할 리스트 초기화
all_outputs = []       # 모델 출력 (시퀀스 형태)
all_year_outputs = []  # 해당 연도 (시퀀스 형태)
all_stock_outputs = [] # 해당 기업 ID (시퀀스 형태, 복제됨)

# company_model과 dataloader_seq가 정상적으로 초기화되었는지 확인
if company_model and dataloader_seq:
    # 모델을 평가 모드로 설정 (Dropout, BatchNorm 등 비활성화)
    company_model.eval()
    # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
    with torch.no_grad():
        # DataLoader에서 배치 단위로 데이터 가져와 반복 처리 (셔플 안 함)
        # 참고: dataloader_seq는 shuffle=True로 생성되었으므로, 엄밀히는 순서가 보장되지 않음.
        #       만약 순서가 중요하다면 shuffle=False인 DataLoader를 따로 만들어야 함.
        for batch_cont, batch_bin, batch_year, batch_stock, _ in dataloader_seq:
            # 현재 배치의 입력 텐서들을 지정된 디바이스로 이동
            batch_cont = batch_cont.to(device)
            batch_bin = batch_bin.to(device)
            batch_year = batch_year.to(device)
            batch_stock = batch_stock.to(device)
            # 모델의 forward 함수 호출하여 출력(out) 계산
            out = company_model(batch_cont, batch_bin, batch_year, batch_stock)
            # 계산된 출력을 CPU로 이동하여 리스트에 추가
            all_outputs.append(out.cpu())
            # 해당 배치의 연도 텐서를 CPU로 이동하여 리스트에 추가
            all_year_outputs.append(batch_year.cpu())
            # 기업 ID 텐서(batch_stock)를 시퀀스 길이에 맞게 복제
            # (batch,) -> (batch, 1) -> (batch, NUM_YEARS_PARAM)
            batch_stock_expanded = batch_stock.unsqueeze(1).repeat(1, NUM_YEARS_PARAM)
            # 복제된 기업 ID 텐서를 CPU로 이동하여 리스트에 추가
            all_stock_outputs.append(batch_stock_expanded.cpu())

    # 리스트에 저장된 배치별 출력 텐서들을 하나로 결합 (첫 번째 차원 기준)
    # output_tensor_seq: (num_total_samples, num_years, ft_out_dim)
    output_tensor_seq = torch.cat(all_outputs, dim=0)
    # 리스트에 저장된 배치별 연도 텐서들을 하나로 결합
    # year_tensor_seq_output: (num_total_samples, num_years)
    year_tensor_seq_output = torch.cat(all_year_outputs, dim=0)
    # 리스트에 저장된 배치별 복제된 기업 ID 텐서들을 하나로 결합
    # stock_tensor_seq_expanded: (num_total_samples, num_years)
    stock_tensor_seq_expanded = torch.cat(all_stock_outputs, dim=0)

    # 시퀀스 형태의 텐서들을 평탄화(flatten)하여 1차원 시퀀스로 만듦
    # output_tensor_flat: (num_total_samples * num_years, ft_out_dim)
    output_tensor_flat = output_tensor_seq.reshape(-1, FT_OUT_DIM_PARAM)
    # year_tensor_flat: (num_total_samples * num_years,)
    year_tensor_flat = year_tensor_seq_output.reshape(-1)
    # 평탄화된 연도 값으로 연도 임베딩 계산 (디버깅 또는 추가 분석용)
    year_embed_flat = get_sine_cosine_year_embedding(year_tensor_flat, dim=YEAR_DIM_PARAM)
    # stock_tensor_flat: (num_total_samples * num_years,)
    stock_tensor_flat = stock_tensor_seq_expanded.reshape(-1)

    # 평탄화된 텐서들의 형태 출력
    print(f"output_tensor_seq shape: {output_tensor_seq.shape}") # 시퀀스 형태 출력 텐서
    print(f"output_tensor_flat shape: {output_tensor_flat.shape}") # 평탄화된 출력 텐서
else:
    # 모델 또는 데이터로더가 초기화되지 않았으면 결과 수집 건너뜀
    print("Skipping CompanySequenceModel output collection as model or dataloader is not initialized.")
    # 빈 텐서로 초기화
    output_tensor_seq = torch.empty(0, NUM_YEARS_PARAM, FT_OUT_DIM_PARAM)
    year_tensor_seq_output = torch.empty(0, NUM_YEARS_PARAM)
    stock_tensor_seq_expanded = torch.empty(0, NUM_YEARS_PARAM)
```

- 이 셀은 학습이 완료된 `CompanySequenceModel`을 사용하여 전체 데이터셋에 대한 출력을 생성하고, 이를 다음 단계(Diffusion 모델)의 입력으로 사용하기 위해 처리합니다.
- `if company_model and dataloader_seq:`: 모델과 데이터로더가 있는지 확인합니다.
    - `company_model.eval()`: 모델을 평가 모드로 전환합니다. Dropout 등이 비활성화됩니다.
    - `with torch.no_grad():`: 그래디언트 계산을 중지하여 메모리 사용량을 줄이고 계산 속도를 높입니다.
        - `for batch_cont, ... in dataloader_seq:`: 데이터로더에서 배치를 가져옵니다.
            - `batch_... = batch_....to(device)`: 입력 데이터를 디바이스로 옮깁니다.
            - `out = company_model(...)`: 모델을 통해 출력을 계산합니다.
            - `all_outputs.append(out.cpu())`: 모델 출력을 CPU로 옮겨 리스트에 저장합니다.
            - `all_year_outputs.append(batch_year.cpu())`: 해당 배치의 연도 정보도 CPU로 옮겨 리스트에 저장합니다.
            - `batch_stock_expanded = ...`: 기업 ID를 시퀀스 길이만큼 복제합니다. (예: [101] -> [[101, 101, ..., 101]])
            - `all_stock_outputs.append(batch_stock_expanded.cpu())`: 복제된 기업 ID를 CPU로 옮겨 리스트에 저장합니다.
    - `output_tensor_seq = torch.cat(...)`: 배치별로 저장된 출력, 연도, 기업 ID 텐서들을 각각 `dim=0` 기준으로 합쳐 전체 데이터셋에 대한 텐서를 만듭니다.
    - `output_tensor_flat = output_tensor_seq.reshape(-1, ...)`: 시퀀스 형태의 텐서들을 평탄화합니다. `(N, S, F)` 형태를 `(N*S, F)` 형태로 바꿉니다. 이는 각 시간 스텝을 독립적인 샘플처럼 다루기 위함일 수 있습니다 (Diffusion 모델 입력 형태에 따라 다름).
    - `year_tensor_flat = ...`, `stock_tensor_flat = ...`: 연도와 기업 ID 텐서도 평탄화합니다.
    - `year_embed_flat = get_sine_cosine_year_embedding(...)`: 평탄화된 연도로 임베딩을 다시 계산합니다 (주로 확인용).
    - `print(...)`: 생성된 텐서들의 형태를 출력합니다.
- `else:`: 모델이나 데이터로더가 없으면 건너뛰고 빈 텐서로 초기화합니다.

## 8'. Transformer-Denoiser 기반 Diffusion (시계열 컨텍스트 활용)

### 8.1 Diffusion 하이퍼파라미터

```python
# Diffusion 프로세스 하이퍼파라미터 설정
T_diff      = 10       # 총 Diffusion 타임스텝 수
beta_start  = 1e-4     # 노이즈 스케줄(beta)의 시작 값
beta_end    = 2e-2     # 노이즈 스케줄(beta)의 종료 값
# beta_start부터 beta_end까지 T_diff개의 값을 선형적으로 생성 (노이즈 스케줄)
betas       = torch.linspace(beta_start, beta_end, T_diff, device=device)
# alphas = 1 - betas
alphas      = 1.0 - betas
# alpha_bars = 누적 곱(alphas) (alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t)
alpha_bars  = torch.cumprod(alphas, dim=0)
```

- 이 셀은 Diffusion 모델의 핵심 하이퍼파라미터를 정의합니다.
- `T_diff`: Diffusion 프로세스를 몇 단계로 나눌지 결정합니다 (노이즈 추가/제거 단계 수).
- `beta_start`, `beta_end`: 각 타임스텝에서 추가되는 노이즈의 양(분산)을 결정하는 $\beta_t$ 값의 범위를 설정합니다. 보통 작은 값에서 시작하여 점진적으로 증가합니다.
- `betas = torch.linspace(...)`: `beta_start`와 `beta_end` 사이를 선형적으로 `T_diff`개의 구간으로 나눈 $\beta_t$ 값들을 생성합니다.
- `alphas = 1.0 - betas`: $\alpha_t = 1 - \beta_t$ 를 계산합니다. 이는 원본 데이터의 비율을 나타냅니다.
- `alpha_bars = torch.cumprod(...)`: $\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$ 를 계산합니다. 이는 타임스텝 $t$까지 원본 데이터가 얼마나 남아있는지를 나타내는 누적 값입니다. Diffusion 수식에서 자주 사용됩니다.

### 8.2 학습/평가용 Dataset (Diffusion)

```python
# 원본 데이터(df_filtered)가 처리되었고 고유 기업 수가 0보다 큰 경우
if not df_filtered.empty and NUM_STOCK_EMBEDDINGS > 0:
    # Diffusion 모델 학습 시 사용할 기업 ID 텐서 (CompanySequenceModel 입력과 동일)
    stock_scalar_seq_diff = torch.tensor(stock_seq, dtype=torch.long)
    # Diffusion 모델 학습 시 사용할 실제 이진 변수 레이블 텐서
    bin_label_tensor_seq_diff = torch.tensor(X_bin_seq, dtype=torch.float32)
else:
    # 데이터가 없으면 빈 텐서로 초기화
    stock_scalar_seq_diff = torch.empty(0, dtype=torch.long)
    bin_label_tensor_seq_diff = torch.empty(0, NUM_YEARS_PARAM, bin_input_dim)

# CompanySequenceModel의 출력(output_tensor_seq)이 비어 있지 않은 경우
if output_tensor_seq.numel() > 0: # .numel()은 텐서의 총 요소 수를 반환
    # Diffusion 모델 학습을 위한 TensorDataset 생성
    diff_dataset = TensorDataset(
        output_tensor_seq,          # Company 모델 출력 (Diffusion 모델의 입력 x0)
        year_tensor_seq_output,     # 해당 연도 시퀀스 (컨디셔닝 정보)
        stock_scalar_seq_diff,      # 해당 기업 ID (컨디셔닝 정보)
        bin_label_tensor_seq_diff   # 실제 이진 변수 값 (Denoiser의 이진 출력 학습용)
    )
    # Diffusion 모델 학습용 DataLoader 생성
    diff_dataloader = DataLoader(diff_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 생성된 데이터셋 크기 출력
    print(f"Diffusion dataset size: {len(diff_dataset)}")
else:
    # Company 모델 출력이 없으면 DataLoader 생성 건너뜀
    print("Skipping Diffusion DataLoader creation as CompanySequenceModel output is empty.")
    diff_dataloader = [] # 빈 DataLoader 또는 리스트
```

- 이 셀은 Diffusion 모델(Denoiser) 학습에 필요한 데이터셋과 데이터로더를 준비합니다.
- `if not df_filtered.empty and NUM_STOCK_EMBEDDINGS > 0:`: 원본 데이터가 성공적으로 처리되었는지 확인합니다.
    - `stock_scalar_seq_diff = torch.tensor(stock_seq, ...)`: 이전 단계에서 생성된 기업 ID 배열(`stock_seq`)을 텐서로 변환합니다. Denoiser 모델의 컨디셔닝 정보로 사용됩니다.
    - `bin_label_tensor_seq_diff = torch.tensor(X_bin_seq, ...)`: 원본 데이터의 이진 변수 시퀀스(`X_bin_seq`)를 텐서로 변환합니다. Denoiser가 예측할 이진 변수의 실제 정답(레이블)으로 사용됩니다.
- `else:`: 데이터가 없으면 빈 텐서로 초기화합니다.
- `if output_tensor_seq.numel() > 0:`: `CompanySequenceModel`의 출력이 존재하는지 확인합니다.
    - `diff_dataset = TensorDataset(...)`: Diffusion 모델 학습에 필요한 입력들을 묶어 `TensorDataset`을 생성합니다.
        - `output_tensor_seq`: Company 모델의 출력이 Diffusion 모델의 '깨끗한 원본 데이터($x_0$)' 역할을 합니다.
        - `year_tensor_seq_output`: 연도 정보는 Denoiser의 컨디셔닝(조건) 정보로 사용됩니다.
        - `stock_scalar_seq_diff`: 기업 ID 정보도 Denoiser의 컨디셔닝 정보로 사용됩니다.
        - `bin_label_tensor_seq_diff`: Denoiser가 예측해야 할 이진 변수의 실제 값입니다.
    - `diff_dataloader = DataLoader(...)`: 생성된 `diff_dataset`으로 DataLoader를 만듭니다. 학습 시 배치 단위로 데이터를 공급하고 섞어줍니다.
    - `print(...)`: 생성된 데이터셋의 샘플 수를 출력합니다.
- `else:`: Company 모델 출력이 없으면 건너뛰고 빈 리스트로 초기화합니다.

### 8.3 Sinusoidal 시간-스텝 임베딩 (`TimeEmbedding`)

```python
# Diffusion 타임스텝(t)을 임베딩하는 클래스 정의
class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        """
        Sinusoidal 방식을 사용하여 Diffusion 타임스텝 t를 d_model 차원의 벡터로 임베딩합니다.
        Positional Encoding과 유사한 원리입니다.

        Args:
            d_model (int): 생성될 임베딩 벡터의 차원.
        """
        super().__init__()
        # 임베딩 차원의 절반 계산
        half_d_model = d_model // 2
        # 주파수 계산 (Positional Encoding과 유사)
        # 1 / (10000^(2i / d_model)) 계산
        inv_freq = 1. / (10000 ** (torch.arange(0, half_d_model, dtype=torch.float32) / half_d_model))
        # 계산된 inv_freq를 모델 버퍼로 등록 (학습되지 않음)
        self.register_buffer("inv_freq", inv_freq)
        # 임베딩 차원 저장
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        입력 타임스텝 t를 임베딩 벡터로 변환합니다.

        Args:
            t (torch.Tensor): Diffusion 타임스텝 텐서. (batch_size, 1) 또는 (batch_size,) 형태.

        Returns:
            torch.Tensor: 타임스텝 임베딩 텐서. (batch_size, d_model) 형태.
        """
        # 입력 타임스텝 t와 주파수(inv_freq) 곱하기
        sinusoid_input = t * self.inv_freq
        # Sine과 Cosine 값을 계산하여 결합
        emb = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)
        # 만약 d_model이 홀수이면, 마지막 차원에 0 패딩 추가
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        # 최종 타임스텝 임베딩 반환
        return emb
```

- 이 클래스는 Diffusion 프로세스의 타임스텝 $t$ (정수 값)를 고정된 차원(`d_model`)의 벡터로 변환하는 역할을 합니다. 이는 모델이 현재 어느 정도의 노이즈가 추가된 상태인지를 알 수 있도록 정보를 제공합니다.
- `__init__`:
    - `d_model`: 출력 임베딩 벡터의 차원입니다.
    - `half_d_model`: 차원의 절반입니다.
    - `inv_freq = 1. / ...`: Positional Encoding과 유사한 방식으로 주파수 관련 항(`1 / (10000^(2i / d_model))`)을 계산합니다.
    - `self.register_buffer("inv_freq", inv_freq)`: 계산된 `inv_freq`를 버퍼로 등록합니다.
    - `self.d_model = d_model`: 차원 값을 저장합니다.
- `forward`:
    - `t`: 입력 타임스텝 텐서입니다.
    - `sinusoid_input = t * self.inv_freq`: 타임스텝과 주파수를 곱합니다.
    - `emb = torch.cat(...)`: 사인과 코사인 값을 계산하여 합칩니다.
    - `if self.d_model % 2 == 1:`: `d_model`이 홀수이면 0으로 패딩하여 차원을 맞춥니다.
    - `return emb`: 최종 타임스텝 임베딩 벡터를 반환합니다.

### 8.4 Transformer-Denoiser

```python
# 노이즈가 추가된 데이터에서 원본 데이터를 예측(denoise)하는 Transformer 기반 모델 정의
class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        num_stock_embeddings_denoiser, # Denoiser 내부 Stock Embedding 레이어 크기
        feat_dim=FT_OUT_DIM_PARAM,     # 입력 특징(Company 모델 출력)의 차원
        d_model=DENOISER_D_MODEL,      # Denoiser 모델의 내부 차원
        nhead=4, num_layers=4,         # Transformer Encoder의 헤드 수 및 레이어 수
        stock_emb_dim=STOCK_DIM_PARAM, # 기업 ID 임베딩 차원
        year_pos_dim=YEAR_DIM_PARAM,   # 연도 임베딩 차원
    ):
        super().__init__()
        self.feat_dim = feat_dim # 입력 특징 차원 저장
        self.d_model = d_model   # 모델 내부 차원 저장

        # 연도 임베딩을 d_model 차원으로 변환하는 선형 레이어
        self.year_proj = nn.Linear(year_pos_dim, d_model)
        # 기업 ID를 임베딩하는 레이어 (Denoiser용, Company 모델과 별개일 수 있음)
        self.stock_emb = nn.Embedding(num_stock_embeddings_denoiser, stock_emb_dim)
        # 입력 특징(feat_dim)과 기업 ID 임베딩(stock_emb_dim)을 합친 후 d_model 차원으로 변환하는 선형 레이어
        self.in_proj = nn.Linear(feat_dim + stock_emb_dim, d_model)

        # 타임스텝(t) 임베딩을 처리하는 모듈 (TimeEmbedding + Linear + SiLU 활성화)
        self.t_embed = nn.Sequential(
            TimeEmbedding(d_model), # 타임스텝 임베딩 생성
            nn.Linear(d_model, d_model), # 선형 변환
            nn.SiLU() # SiLU 활성화 함수
        )
        # Positional Encoding 레이어 (시퀀스 내 위치 정보 추가)
        self.pos_enc = PositionalEncoding(d_model, max_len=NUM_YEARS_PARAM)

        # Transformer Encoder Layer 정의
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,           # 모델 차원, 어텐션 헤드 수
            dim_feedforward=d_model * 4,            # FeedForward 네트워크의 내부 차원
            dropout=0.1, batch_first=True           # 드롭아웃 비율, 입력 형태 (batch, seq, feature)
        )
        # 여러 개의 Transformer Encoder Layer를 쌓은 Transformer Encoder 정의
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Transformer Encoder 출력을 원래 데이터 형태로 변환하는 출력 레이어
        # 연속형 변수 예측용 선형 레이어
        self.out_cont = nn.Linear(d_model, cont_input_dim)
        # 이진 변수 예측용 선형 레이어 (Logit 출력)
        self.out_bin  = nn.Linear(d_model,  bin_input_dim)

    def forward(self, x_t, years, stock_id, t_norm):
        """
        노이즈가 추가된 입력(x_t)과 컨디셔닝 정보(years, stock_id, t_norm)를 받아
        원본 데이터의 연속형 및 이진형 변수를 예측합니다.

        Args:
            x_t (torch.Tensor): 타임스텝 t에서 노이즈가 추가된 데이터. (B, S, feat_dim)
                                (B: batch_size, S: sequence_length)
            years (torch.Tensor): 연도 값 텐서. (B, S)
            stock_id (torch.Tensor): 기업 ID 텐서. (B,)
            t_norm (torch.Tensor): 정규화된 타임스텝 텐서 (t / T_diff). (B, 1)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - 예측된 연속형 변수 텐서. (B, S, cont_input_dim)
                - 예측된 이진 변수(logit) 텐서. (B, S, bin_input_dim)
        """
        # 입력 x_t의 형태에서 배치 크기(B)와 시퀀스 길이(S) 가져오기
        B, S, _ = x_t.shape
        # 연도 값을 임베딩 (get_sine_cosine_year_embedding 함수 사용)
        # 입력 차원은 year_proj 레이어의 입력 차원(year_pos_dim)과 동일해야 함
        year_embed_raw = get_sine_cosine_year_embedding(
            years, dim=self.year_proj.in_features # year_proj의 입력 차원 사용
        )
        # 연도 임베딩을 d_model 차원으로 변환
        year_embed = self.year_proj(year_embed_raw) # (B, S, d_model)

        # 기업 ID 임베딩 및 시퀀스 길이에 맞게 복제
        # stock_id (B,) -> (B, stock_emb_dim) -> (B, 1, stock_emb_dim) -> (B, S, stock_emb_dim)
        stock_embed_val = self.stock_emb(stock_id).unsqueeze(1).repeat(1, S, 1)

        # 입력 특징(x_t)과 기업 ID 임베딩을 결합하고 d_model 차원으로 변환
        # x_t: (B, S, feat_dim), stock_embed_val: (B, S, stock_emb_dim)
        # -> cat: (B, S, feat_dim + stock_emb_dim) -> in_proj: (B, S, d_model)
        h = self.in_proj(torch.cat([x_t, stock_embed_val], dim=-1))

        # 입력 임베딩(h)에 Positional Encoding, 연도 임베딩, 타임스텝 임베딩 더하기
        # t_embed(t_norm): (B, d_model) -> unsqueeze(1): (B, 1, d_model) (브로드캐스팅 위해)
        h = self.pos_enc(h) + year_embed + self.t_embed(t_norm).unsqueeze(1) # (B, S, d_model)

        # Transformer Encoder 통과
        h = self.encoder(h) # (B, S, d_model)

        # 최종 출력을 연속형과 이진형으로 분리하여 반환
        # out_cont(h): (B, S, cont_input_dim)
        # out_bin(h): (B, S, bin_input_dim)
        return self.out_cont(h), self.out_bin(h)
```

- 이 클래스는 Diffusion 모델의 핵심인 Denoiser를 구현합니다. 노이즈가 섞인 데이터($x_t$)와 컨디셔닝 정보(시간 $t$, 연도, 기업 ID)를 입력받아 원본 데이터($x_0$)를 예측하는 역할을 합니다. Transformer 구조를 기반으로 합니다.
- `__init__`:
    - `num_stock_embeddings_denoiser`: Denoiser 내부에서 사용할 Stock Embedding 레이어의 크기입니다.
    - `feat_dim`: 입력 특징($x_t$)의 차원 (Company 모델의 출력 차원 `FT_OUT_DIM_PARAM`).
    - `d_model`: Denoiser 모델 내부에서 사용할 기본 차원입니다.
    - `nhead`, `num_layers`: Transformer Encoder의 하이퍼파라미터입니다.
    - `stock_emb_dim`, `year_pos_dim`: 기업 ID와 연도 임베딩의 차원입니다.
    - `self.year_proj`: 연도 임베딩을 `d_model` 차원으로 변환하는 레이어입니다.
    - `self.stock_emb`: 기업 ID를 임베딩하는 레이어입니다.
    - `self.in_proj`: 입력 특징($x_t$)과 기업 ID 임베딩을 합친 후 `d_model` 차원으로 변환하는 레이어입니다.
    - `self.t_embed`: 이전에 정의한 `TimeEmbedding` 클래스를 사용하여 타임스텝 $t$를 `d_model` 차원으로 임베딩하고 추가 처리를 하는 모듈입니다.
    - `self.pos_enc`: 시퀀스 내 위치 정보를 위한 `PositionalEncoding` 인스턴스입니다.
    - `enc_layer`, `self.encoder`: 표준 Transformer Encoder 구조입니다.
    - `self.out_cont`, `self.out_bin`: 최종 예측을 위한 출력 레이어입니다. 각각 연속형 변수와 이진 변수(logit)를 예측합니다.
- `forward`:
    - `x_t`: 노이즈가 추가된 입력 데이터입니다.
    - `years`, `stock_id`, `t_norm`: 컨디셔닝 정보 (연도, 기업 ID, 정규화된 타임스텝)입니다.
    - `B, S, _ = x_t.shape`: 배치 크기와 시퀀스 길이를 얻습니다.
    - `year_embed_raw = get_sine_cosine_year_embedding(...)`: 연도 값을 임베딩합니다.
    - `year_embed = self.year_proj(...)`: 연도 임베딩을 `d_model` 차원으로 변환합니다.
    - `stock_embed_val = self.stock_emb(...).unsqueeze(1).repeat(...)`: 기업 ID를 임베딩하고 시퀀스 길이에 맞게 복제합니다.
    - `h = self.in_proj(torch.cat(...))`: 입력 특징($x_t$)과 기업 ID 임베딩을 합쳐 `d_model` 차원으로 만듭니다.
    - `h = self.pos_enc(h) + year_embed + self.t_embed(t_norm).unsqueeze(1)`: 계산된 임베딩 `h`에 위치 인코딩, 연도 임베딩, 타임스텝 임베딩을 더해줍니다. 타임스텝 임베딩은 `(B, d_model)` 형태이므로 `unsqueeze(1)`을 통해 `(B, 1, d_model)`로 만들어 브로드캐스팅이 가능하게 합니다.
    - `h = self.encoder(h)`: 모든 정보가 합쳐진 `h`를 Transformer Encoder에 통과시켜 시퀀스 내의 관계를 학습합니다.
    - `return self.out_cont(h), self.out_bin(h)`: 최종 Transformer 출력을 각각 연속형 변수와 이진 변수 예측을 위한 선형 레이어에 통과시켜 결과를 반환합니다.

### 8.5 Forward Diffusion (`q_sample`)

```python
# Forward Diffusion Process: 원본 데이터(x0)에 노이즈를 추가하여 x_t를 생성하는 함수
def q_sample(x0_seq, t_int):
    """
    Diffusion의 Forward Process를 구현합니다.
    주어진 원본 데이터 x0_seq와 타임스텝 t_int에 대해 노이즈가 추가된 데이터 x_t와
    추가된 노이즈(epsilon)를 반환합니다.
    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon

    Args:
        x0_seq (torch.Tensor): 원본 데이터 시퀀스 텐서. (B, S, F)
                               (B: batch_size, S: sequence_length, F: feature_dim)
        t_int (torch.Tensor): 타임스텝 정수 텐서. (B,) 또는 (B, 1) 형태. 각 샘플에 대한 타임스텝.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_t (torch.Tensor): 타임스텝 t에서 노이즈가 추가된 데이터. (B, S, F)
            - noise (torch.Tensor): 추가된 노이즈 (epsilon). (B, S, F)
    """
    # 타임스텝 인덱스 계산 (alpha_bars는 0부터 시작하므로 t-1)
    t_idx = t_int.long() - 1
    # 해당 타임스텝의 sqrt(alpha_bar_t) 값 가져오기 및 차원 확장 (B, 1, 1)
    sqrt_ab  = torch.sqrt(alpha_bars[t_idx]).view(-1,1,1)
    # 해당 타임스텝의 sqrt(1 - alpha_bar_t) 값 가져오기 및 차원 확장 (B, 1, 1)
    sqrt_1m_ab  = torch.sqrt(1-alpha_bars[t_idx]).view(-1,1,1)
    # 원본 데이터(x0_seq)와 동일한 형태의 표준 정규 분포 노이즈 생성
    noise    = torch.randn_like(x0_seq)
    # Forward Process 공식에 따라 x_t 계산
    # 브로드캐스팅을 통해 각 배치 샘플에 맞는 alpha_bar 값 적용
    return sqrt_ab*x0_seq + sqrt_1m_ab*noise, noise
```

- 이 함수는 Diffusion 모델의 Forward Process, 즉 원본 데이터 $x_0$에 점진적으로 노이즈를 추가하여 $x_t$를 만드는 과정을 수학적으로 구현합니다.
- `x0_seq`: 노이즈를 추가할 원본 데이터입니다. 여기서는 `CompanySequenceModel`의 출력이 됩니다.
- `t_int`: 노이즈를 추가할 타임스텝입니다. (1부터 `T_diff`까지의 정수)
- `t_idx = t_int.long() - 1`: `alpha_bars` 배열은 인덱스 0부터 시작하므로, `t_int`에서 1을 빼서 해당 인덱스를 구합니다.
- `sqrt_ab = torch.sqrt(alpha_bars[t_idx]).view(-1,1,1)`: $t$ 시점의 $\sqrt{\bar{\alpha}_t}$ 값을 가져옵니다. `.view(-1, 1, 1)`은 이 값을 `(B, 1, 1)` 형태로 만들어 `x0_seq`(`(B, S, F)`)와의 브로드캐스팅 곱셈이 가능하도록 합니다.
- `sqrt_1m_ab = torch.sqrt(1-alpha_bars[t_idx]).view(-1,1,1)`: $t$ 시점의 $\sqrt{1 - \bar{\alpha}_t}$ 값을 가져와 동일하게 차원을 조정합니다.
- `noise = torch.randn_like(x0_seq)`: $x_0$와 동일한 크기의 표준 정규 분포(평균 0, 분산 1) 노이즈 $\epsilon$을 생성합니다.
- `return sqrt_ab*x0_seq + sqrt_1m_ab*noise, noise`: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 공식을 사용하여 $x_t$를 계산하고, 생성된 $x_t$와 사용된 노이즈 $\epsilon$을 함께 반환합니다. (Denoiser는 보통 $\epsilon$을 예측하도록 학습되기도 합니다.)

### 8.6 Denoiser 학습 설정 및 `snr_weight`

```python
# 고유 기업 수가 0보다 클 때 (즉, 학습할 데이터가 있을 때) Denoiser 모델 초기화
if NUM_STOCK_EMBEDDINGS > 0:
    # TransformerDenoiser 인스턴스 생성 및 지정된 디바이스로 이동
    denoiser = TransformerDenoiser(
        num_stock_embeddings_denoiser=NUM_STOCK_EMBEDDINGS, # 고유 기업 수 전달
        feat_dim=FT_OUT_DIM_PARAM,
        d_model=DENOISER_D_MODEL,
        year_pos_dim=YEAR_DIM_PARAM,
        stock_emb_dim=STOCK_DIM_PARAM
    ).to(device)
    # AdamW 옵티마이저 생성 (Denoiser 파라미터와 학습률 전달)
    opt_denoiser = optim.AdamW(denoiser.parameters(), lr=LEARNING_RATE_DENOISER) # 옵티마이저 이름 변경 (company와 구분)
    # Denoiser용 TensorBoard SummaryWriter 생성 (고유한 로그 디렉토리 이름 사용)
    current_time_str_denoiser = time.strftime("%Y%m%d-%H%M%S")
    writer_denoiser = SummaryWriter(f'{TENSORBOARD_LOG_DIR_DENOISER}_{current_time_str_denoiser}')
else:
    # 학습할 데이터가 없으면 Denoiser 관련 변수들을 None으로 설정
    print("Skipping Denoiser model initialization as there are no stocks.")
    denoiser = None
    opt_denoiser = None
    writer_denoiser = None

# Diffusion 손실 가중치 계산 함수 (SNR 기반)
def snr_weight(t_idx: torch.Tensor,
               alpha_bars_local: torch.Tensor,
               strategy: str = "karras", # 가중치 계산 전략 (기본값 "karras")
               rho: float = 1.2) -> torch.Tensor: # Karras 전략 사용 시 파라미터
    """
    타임스텝별 Signal-to-Noise Ratio (SNR)를 기반으로 손실 가중치를 계산합니다.
    노이즈 레벨이 다른 타임스텝에 대해 학습 중요도를 조절하는 데 사용됩니다.

    Args:
        t_idx (torch.Tensor): 타임스텝 인덱스 텐서 (0부터 T_diff-1). (B,) 형태.
        alpha_bars_local (torch.Tensor): 미리 계산된 alpha_bars 텐서. (T_diff,) 형태.
        strategy (str): 가중치 계산 방식 ("karras" 또는 "simple").
        rho (float): Karras 전략에서 사용되는 파라미터.

    Returns:
        torch.Tensor: 계산된 손실 가중치 텐서. (B,) 형태.
    """
    # 해당 타임스텝 인덱스(t_idx)의 alpha_bar 값 가져오기
    ab = alpha_bars_local[t_idx]
    # SNR 계산: SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
    snr = ab / (1.0 - ab)
    # Karras et al. (2022) 방식의 가중치 계산
    if strategy == "karras":
        # weight = (SNR(t) + 1)^(-rho)
        weight = (snr + 1.0).pow(-rho)
    # 단순 방식의 가중치 계산
    elif strategy == "simple":
        # weight = 1 / (SNR(t) + 1)
        weight = 1.0 / (snr + 1.0)
    # 알 수 없는 전략인 경우 오류 발생
    else:
        raise ValueError(f"unknown strategy {strategy}")
    # 계산된 가중치 반환
    return weight
```

- 이 셀은 Denoiser 모델과 옵티마이저를 초기화하고, 학습 시 사용할 손실 가중치 계산 함수(`snr_weight`)를 정의합니다.
- `if NUM_STOCK_EMBEDDINGS > 0:`: 학습 데이터가 있을 때 Denoiser 모델(`TransformerDenoiser`) 인스턴스를 생성하고 디바이스로 옮깁니다. AdamW 옵티마이저(`opt_denoiser`)와 TensorBoard 로거(`writer_denoiser`)도 생성합니다.
- `else:`: 데이터가 없으면 관련 변수들을 `None`으로 초기화합니다.
- `snr_weight` 함수:
    - Diffusion 모델 학습 시, 타임스텝 $t$에 따라 노이즈의 양이 달라지므로, 각 타임스텝의 손실에 다른 가중치를 부여하는 것이 효과적일 수 있습니다. 이 함수는 Signal-to-Noise Ratio (SNR)를 기반으로 가중치를 계산합니다.
    - `t_idx`: 타임스텝 인덱스 (0부터 `T_diff-1`).
    - `alpha_bars_local`: 미리 계산된 $\bar{\alpha}_t$ 값들.
    - `strategy`: 가중치 계산 방식을 선택합니다 ("karras" 또는 "simple").
    - `rho`: "karras" 전략에서 사용되는 하이퍼파라미터입니다.
    - `ab = alpha_bars_local[t_idx]`: 해당 타임스텝의 $\bar{\alpha}_t$ 값을 가져옵니다.
    - `snr = ab / (1.0 - ab)`: SNR($t$) = $\bar{\alpha}_t / (1 - \bar{\alpha}_t)$를 계산합니다.
    - `if strategy == "karras": ... elif strategy == "simple": ...`: 선택된 전략에 따라 가중치를 계산합니다. Karras 방식은 $(SNR(t) + 1)^{-\rho}$ 이고, simple 방식은 $1 / (SNR(t) + 1)$ 입니다.
    - `return weight`: 계산된 가중치를 반환합니다.

### 8.7 Denoiser 학습 루프

```python
# Denoiser 학습용 손실 함수 정의
# 연속형 변수용 MSE 손실 (reduction='none'으로 설정하여 샘플별 손실 유지)
criterion_c = nn.MSELoss(reduction='none')
# 이진 변수용 BCE 손실 (reduction='none', pos_weight 적용)
bce_fn      = nn.BCEWithLogitsLoss(pos_weight=pos_w,
                                   reduction='none')

# Denoiser의 이진 변수 손실에 곱해줄 가중치 람다 값
λ_bin_denoiser = 10.0

# denoiser 모델과 diff_dataloader가 정상적으로 초기화되었는지 확인
if denoiser and diff_dataloader:
    # TensorBoard writer가 초기화되었는지 확인
    if writer_denoiser:
        try:
            # DataLoader에서 첫 번째 배치를 가져와 모델 그래프 로깅 시도
            data_iter_denoiser = iter(diff_dataloader) # DataLoader의 iterator 생성
            # 첫 배치 데이터 언패킹 (이진 레이블은 사용 안 함 '_')
            sample_x0_diff, sample_yrs_diff, sample_st_diff, _ = next(data_iter_denoiser)
            # 랜덤 타임스텝 생성 (그래프 로깅용)
            sample_t_norm = torch.rand(sample_x0_diff.size(0), 1, device=device)
            # TensorBoard에 Denoiser 모델 구조(그래프) 기록
            writer_denoiser.add_graph(denoiser, [sample_x0_diff.to(device), sample_yrs_diff.to(device), sample_st_diff.to(device), sample_t_norm])
            # 사용한 iterator 삭제
            del data_iter_denoiser
        except Exception as e:
            # 그래프 로깅 중 오류 발생 시 메시지 출력
            print(f"Error adding Denoiser graph to TensorBoard: {e}")

    # 지정된 에포크 수만큼 학습 반복
    for ep in range(EPOCHS_DENOISER_MODEL):
        # 에포크별 손실 초기화
        epoch_loss_denoiser = 0.0
        num_batches_denoiser = 0

        # Diffusion DataLoader에서 배치 단위로 데이터 가져와 반복 처리
        # x0_diff: 원본 데이터(Company 모델 출력), yrs_diff: 연도, st_diff: 기업 ID, bin_true_diff: 실제 이진 값
        for x0_diff, yrs_diff, st_diff, bin_true_diff in diff_dataloader:
            # 현재 배치의 텐서들을 지정된 디바이스로 이동
            x0_diff, yrs_diff, st_diff = x0_diff.to(device), yrs_diff.to(device), st_diff.to(device)
            bin_true_diff = bin_true_diff.to(device)
            # 배치 크기(B) 가져오기
            B = x0_diff.size(0)

            # 현재 배치에 대해 랜덤 타임스텝(t) 생성 (1부터 T_diff까지 정수) 및 정렬 (선택 사항)
            t_int_rand, _ = torch.sort(torch.randint(1, T_diff + 1, (B,), device=device))
            # Forward Diffusion(q_sample)을 사용하여 노이즈가 추가된 데이터(x_t) 생성
            x_t, _   = q_sample(x0_diff, t_int_rand) # 추가된 노이즈(epsilon)는 여기선 사용 안 함
            # 타임스텝(t)을 [0, 1] 범위로 정규화 (t / T_diff) 및 차원 추가 (B, 1)
            t_norm   = t_int_rand.float().unsqueeze(1) / T_diff

            # Denoiser 모델 호출하여 원본 데이터 예측 (연속형, 이진형 분리)
            cont_hat, bin_hat = denoiser(x_t, yrs_diff, st_diff, t_norm)
            # 이진 변수 예측값(logit)의 범위를 제한하여 수치 안정성 확보
            bin_hat = bin_hat.clamp(-15, 15)

            # 예측 목표(Target) 설정
            # 연속형 변수 목표: 원본 데이터(x0_diff)의 연속형 부분
            cont_tgt = x0_diff[:, :, :cont_input_dim]
            # 이진 변수 목표: 실제 이진 변수 값 (bin_true_diff)
            bin_tgt  = bin_true_diff

            # 샘플별 손실 계산 (reduction='none' 사용)
            # 연속형 MSE 손실 계산 후 시퀀스 및 특징 차원에 대해 평균 -> (B,) 형태
            mse = criterion_c(cont_hat, cont_tgt).mean(dim=(1,2))
            # 이진형 BCE 손실 계산 후 시퀀스 및 특징 차원에 대해 평균 -> (B,) 형태
            bce = bce_fn(bin_hat, bin_tgt).mean(dim=(1, 2))
            # 해당 타임스텝(t_int_rand)에 대한 SNR 기반 손실 가중치(w) 계산
            w   = snr_weight(t_int_rand - 1, alpha_bars, "karras", rho=1.2) # t_idx는 0부터 시작
            # 최종 배치 손실 계산: 가중치가 적용된 MSE와 BCE의 평균
            loss = (w * mse + λ_bin_denoiser * bce).mean() # 배치 차원에 대해 평균

            # 옵티마이저 그래디언트 초기화
            opt_denoiser.zero_grad()
            # 손실에 대한 그래디언트 계산 (역전파)
            loss.backward()

            # 특정 에포크마다 또는 마지막 에포크에 그래디언트 히스토그램 로깅
            if writer_denoiser and (ep % (EPOCHS_DENOISER_MODEL // 5 if EPOCHS_DENOISER_MODEL >=5 else 1) == 0 or ep == EPOCHS_DENOISER_MODEL -1):
                for name, param in denoiser.named_parameters():
                    if param.grad is not None:
                        writer_denoiser.add_histogram(f'Gradients_Denoiser/{name.replace(".", "/")}', param.grad, ep)

            # 옵티마이저를 사용하여 파라미터 업데이트
            opt_denoiser.step()

            # 에포크 손실 누적 및 배치 수 증가
            epoch_loss_denoiser += loss.item()
            num_batches_denoiser +=1

        # 에포크 평균 손실 계산
        avg_loss_denoiser = epoch_loss_denoiser / num_batches_denoiser if num_batches_denoiser > 0 else 0
        # TensorBoard writer가 있으면 에포크별 평균 손실 로깅
        if writer_denoiser:
            writer_denoiser.add_scalar('Loss_Denoiser/Total_Train', avg_loss_denoiser, ep)

        # 특정 에포크마다 또는 마지막 에포크에 평균 손실 출력 및 가중치 히스토그램 로깅
        if ep % (EPOCHS_DENOISER_MODEL // 5 if EPOCHS_DENOISER_MODEL >=5 else 1) == 0 or ep == EPOCHS_DENOISER_MODEL -1:
            print(f"[Denoiser] ep {ep:03d} | loss {avg_loss_denoiser:.5f}")

            # TensorBoard writer가 있으면 모델 파라미터(가중치) 히스토그램 로깅
            if writer_denoiser:
                for name, param in denoiser.named_parameters():
                    if param.requires_grad:
                        writer_denoiser.add_histogram(f'Weights_Denoiser/{name.replace(".", "/")}', param.data, ep)

    # 학습 완료 후 TensorBoard writer 닫기
    if writer_denoiser: writer_denoiser.close()

else:
    # 모델 또는 데이터로더가 초기화되지 않았으면 학습 건너뜀
    print("Skipping Denoiser training as model or dataloader is not initialized.")
```

- 이 셀은 `TransformerDenoiser` 모델의 학습 과정을 구현합니다.
- `criterion_c = nn.MSELoss(reduction='none')`, `bce_fn = nn.BCEWithLogitsLoss(...)`: 손실 함수를 정의합니다. `reduction='none'`은 배치 내 각 샘플에 대한 손실 값을 그대로 반환하도록 하여, 이후 `snr_weight`로 계산된 가중치를 샘플별로 적용할 수 있게 합니다.
- `λ_bin_denoiser = 10.0`: Denoiser 학습 시 이진 변수 손실에 적용할 가중치입니다.
- `if denoiser and diff_dataloader:`: 모델과 데이터로더가 준비되었는지 확인합니다.
    - `if writer_denoiser:`: TensorBoard 로거가 있는지 확인하고 모델 그래프를 기록합니다.
    - `for ep in range(EPOCHS_DENOISER_MODEL):`: 설정된 에포크 수만큼 반복합니다.
        - `epoch_loss_denoiser = 0.0`, `num_batches_denoiser = 0`: 에포크 손실과 배치 수를 초기화합니다.
        - `for x0_diff, ... in diff_dataloader:`: 데이터로더에서 미니배치를 가져옵니다. `x0_diff`는 Company 모델의 출력(여기서는 $x_0$ 역할), `yrs_diff`, `st_diff`는 컨디셔닝 정보, `bin_true_diff`는 이진 변수의 실제 값입니다.
            - `x0_diff, ... = ....to(device)`: 데이터를 디바이스로 옮깁니다.
            - `t_int_rand, _ = torch.sort(torch.randint(...))`: 배치 크기만큼 1부터 `T_diff` 사이의 랜덤 정수 타임스텝 $t$를 생성합니다.
            - `x_t, _ = q_sample(x0_diff, t_int_rand)`: 생성된 $t$를 사용하여 $x_0$(`x0_diff`)로부터 노이즈가 추가된 $x_t$를 만듭니다.
            - `t_norm = t_int_rand.float().unsqueeze(1) / T_diff`: 타임스텝 $t$를 [0, 1] 범위로 정규화합니다.
            - `cont_hat, bin_hat = denoiser(x_t, yrs_diff, st_diff, t_norm)`: Denoiser 모델에 $x_t$와 컨디셔닝 정보를 입력하여 원본 데이터($x_0$)의 연속형(`cont_hat`)과 이진형(`bin_hat`) 예측값을 얻습니다.
            - `bin_hat = bin_hat.clamp(-15, 15)`: 이진 예측값(logit)이 너무 커지거나 작아지는 것을 방지합니다.
            - `cont_tgt = x0_diff[:, :, :cont_input_dim]`, `bin_tgt = bin_true_diff`: 예측의 목표값(정답)을 설정합니다. 연속형은 $x_0$의 연속형 부분, 이진형은 실제 이진 값입니다.
            - `mse = criterion_c(...).mean(dim=(1,2))`, `bce = bce_fn(...).mean(dim=(1, 2))`: 샘플별 손실을 계산합니다. `mean(dim=(1,2))`는 시퀀스와 특징 차원에 대해 평균을 내어 각 배치 샘플당 하나의 손실 값(`(B,)` 형태)을 얻습니다.
            - `w = snr_weight(...)`: 현재 타임스텝 `t_int_rand`에 대한 손실 가중치 `w`를 계산합니다.
            - `loss = (w * mse + λ_bin_denoiser * bce).mean()`: 샘플별 가중 손실을 계산하고, 배치 전체에 대해 평균을 내어 최종 손실 값을 얻습니다.
            - `opt_denoiser.zero_grad()`, `loss.backward()`, `opt_denoiser.step()`: 표준적인 PyTorch 학습 단계 (그래디언트 초기화, 역전파, 파라미터 업데이트)를 수행합니다.
            - `if writer_denoiser and ...`: 특정 주기로 그래디언트 히스토그램을 로깅합니다.
            - `epoch_loss_denoiser += loss.item()`: 에포크 손실을 누적합니다.
        - `avg_loss_denoiser = ...`: 에포크 평균 손실을 계산합니다.
        - `if writer_denoiser:`: 평균 손실을 TensorBoard에 기록합니다.
        - `if ep % ...`: 특정 주기로 콘솔에 손실을 출력하고, 모델 가중치 히스토그램을 TensorBoard에 기록합니다.
    - `if writer_denoiser: writer_denoiser.close()`: 학습 완료 후 로거를 닫습니다.
- `else:`: 모델이나 데이터로더가 없으면 학습을 건너뜁니다.

### 8.8 Reverse Diffusion Sampler (`p_sample_loop`)

```python
# Reverse Diffusion Process: 노이즈에서 시작하여 원본 데이터를 복원(샘플링)하는 함수
@torch.no_grad() # 샘플링 과정에서는 그래디언트 계산이 필요 없음
def p_sample_loop(model, years_vec, stock_id, seq_len=NUM_YEARS_PARAM):
    """
    Diffusion 모델의 Reverse Process를 수행하여 새로운 데이터 샘플을 생성합니다.
    순수 노이즈에서 시작하여 점진적으로 노이즈를 제거해 나갑니다.

    Args:
        model (nn.Module): 학습된 Denoiser 모델 (TransformerDenoiser).
        years_vec (torch.Tensor): 생성할 데이터의 연도 시퀀스 텐서. (seq_len,) 형태.
        stock_id (int): 생성할 데이터의 기업 ID.
        seq_len (int): 생성할 시퀀스의 길이. 기본값 NUM_YEARS_PARAM.

    Returns:
        torch.Tensor: 생성된 데이터 샘플 텐서 (x0 형태). (seq_len, cont_input_dim + bin_input_dim)
    """
    # 모델을 평가 모드로 설정
    model.eval()
    # 초기 상태: 표준 정규 분포 노이즈 생성 (샘플 1개 기준)
    # x: (1, seq_len, model.feat_dim) 형태 (Denoiser 입력 특징 차원)
    x = torch.randn(1, seq_len, model.feat_dim, device=device)
    # 연도 벡터를 배치 차원 추가하여 디바이스로 이동
    years = years_vec.unsqueeze(0).to(device) # (1, seq_len)
    # 기업 ID를 텐서로 변환하고 배치 차원 추가하여 디바이스로 이동
    stock = torch.tensor([stock_id], device=device) # (1,)

    # 타임스텝 T_diff부터 1까지 역순으로 반복 (노이즈 제거 과정)
    for t_val in range(T_diff, 0, -1):
