import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding
from sklearn.preprocessing import MinMaxScaler
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
import transformers
import deepspeed
import json

# DeepSpeed 초기화를 가장 먼저 수행
deepspeed.init_distributed()

# 로컬 랭크 가져오기
local_rank = int(os.getenv("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# 디버깅 정보 출력
print(f"Local Rank: {local_rank}")
print(f"World Size: {os.getenv('WORLD_SIZE')}")
print(f"CUDA Device: {torch.cuda.current_device()}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(local_rank)}")

###############################################################################
# 1) Device 및 cudnn 설정
###############################################################################
cudnn.benchmark = True

###############################################################################
# 2) DeepSpeed 설정
###############################################################################
ds_config = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,                     # dynamic loss scaling 사용
        "loss_scale_window": 1000,
        "initial_scale_power": 12,           # 기존 16 -> 12로 조정 (4096로 초기 scale)
        "loss_scale_power": 1,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "eps": 1e-8
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-5,
            "warmup_num_steps": 100
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 5e6
    },
    "gradient_accumulation_steps": 16,
    "gradient_clipping": 1.0,
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 16,
    "wall_clock_breakdown": False
}

# DeepSpeed 설정 파일 저장
with open('ds_config.json', 'w') as f:
    json.dump(ds_config, f)

###############################################################################
# 3) 데이터 로드 및 전처리
###############################################################################
file_path_csv = "data1.csv"
df = pd.read_csv(file_path_csv)
df = df.drop(columns=["Name"], errors="ignore")
if "YEAR" not in df.columns:
    raise KeyError("YEAR 컬럼이 데이터셋에 없습니다.")

binary_columns = ["BIG4", "LOSS"]
continuous_columns = ["OWN", "FORN", "SIZE", "LEV", "CUR", "GRW", "ROA", "ROE", "CFO", "PPE", "AGE", "INVREC", "MB", "TQ"]

scaler = MinMaxScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])
df["Stock"], unique_stocks = pd.factorize(df["Stock"])
df["YEAR"] = df["YEAR"] - df["YEAR"].min()

num_stocks = len(unique_stocks)
year_count = df["YEAR"].nunique()

###############################################################################
# 4) 임베딩 레이어 정의
###############################################################################
company_embedding_layer = nn.Embedding(num_stocks, 16)
year_embedding_layer = nn.Embedding(year_count, 8)
binary_embedding_layer = nn.Linear(len(binary_columns), 8)
feature_embedding_layer = nn.Linear(len(continuous_columns), 16)

###############################################################################
# 5) 모델 로드 및 Weight Tying 적용
###############################################################################
model_name = "meta-llama/Llama-3.2-3B-Instruct"
config = AutoConfig.from_pretrained(model_name)
config.output_hidden_states = True
config.return_dict_in_generate = True

# 모델 로드 (device 지정 제거)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    config=config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Weight tying: lm_head와 embed_tokens의 가중치를 공유
base_model.lm_head.weight = base_model.model.embed_tokens.weight

###############################################################################
# 6) 전체 모델을 하나의 nn.Module로 묶기
###############################################################################
class FullModel(nn.Module):
    def __init__(self, base_model, company_emb, year_emb, binary_emb, feature_emb):
        super().__init__()
        self.model = base_model
        self.company_embedding_layer = company_emb
        self.year_embedding_layer = year_emb
        self.binary_embedding_layer = binary_emb
        self.feature_embedding_layer = feature_emb
        self.embedding_projection = nn.Linear(48, base_model.config.hidden_size)

    def forward(self, input_ids, attention_mask, labels=None, combined_embedding=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if combined_embedding is not None:
            proj = self.embedding_projection(combined_embedding)
            hidden_state_summary = outputs.hidden_states[-1].mean(dim=1).unsqueeze(1)  # [batch_size, 1, hidden_size]
            aux_loss = nn.functional.mse_loss(proj, hidden_state_summary)
            outputs.loss = outputs.loss + aux_loss
        return outputs

full_model = FullModel(
    base_model,
    company_embedding_layer,
    year_embedding_layer,
    binary_embedding_layer,
    feature_embedding_layer
)

# DeepSpeed 초기화
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=full_model,
    config=ds_config,
    model_parameters=full_model.parameters(),
    dist_init_required=True
)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"

###############################################################################
# 7) Dataset 정의 및 Custom Collate Function
###############################################################################
def prepare_data_for_llama(df):
    """
    각 행(올해 데이터)에 대해 프롬프트(Instruction, Input)와 정답(Answer)을 구성하고 토큰화합니다.
    프롬프트 부분은 -100으로 마스킹해 손실 계산에서 제외합니다.
    추가 임베딩 계산에 필요한 정보도 함께 반환합니다.
    """
    processed_data = []
    stock_ids = []
    years = []
    binary_values = []
    feature_values = []
    
    for _, row in df.iterrows():
        # 현재 행(예: 2011년)의 재무 지표 문자열 구성 (프롬프트용)
        current_year_metrics = ", ".join([
            f"{col}: {row[col]:.4f}" for col in (continuous_columns + binary_columns)
        ])
        
        # 다음 해 데이터(Year + 1) 확인 (예: 2012년 데이터)
        next_year = row["YEAR"] + 1
        next_year_row = df[(df["Stock"] == row["Stock"]) & (df["YEAR"] == next_year)]
        if next_year_row.empty:
            continue
        else:
            ny_row = next_year_row.iloc[0]
            next_year_metrics = ", ".join([
                f"{col}: {ny_row[col]:.4f}" for col in (continuous_columns + binary_columns)
            ])
        
        # 프롬프트 템플릿 구성
        prompt = (
            "# Instruction:\n"
            "Given the financial statement data below, predict the financial statement for the next year.\n\n"
            "# Input:\n"
            f"Company: {row['Stock']}\n"
            f"Year: {int(row['YEAR'])}\n"
            f"Metrics: {current_year_metrics}\n\n"
            "# Answer:\n"
        )
        # 정답 템플릿 구성 (내년 데이터)
        output_text = (
            f"Company: {row['Stock']}\n"
            f"Year: {int(next_year)}\n"
            f"Metrics: {next_year_metrics}"
        )
        
        full_text = prompt + output_text
        tokenized = tokenizer(full_text, truncation=True, max_length=512)
        
        # 프롬프트 부분 길이 계산 및 -100 마스킹
        prompt_tokenized = tokenizer(prompt, truncation=True, max_length=512)
        prompt_length = len(prompt_tokenized.input_ids)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        labels = labels[:len(input_ids)]
        tokenized["labels"] = labels
        
        processed_data.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": tokenized["labels"]
        })
        stock_ids.append(row["Stock"])
        years.append(row["YEAR"])
        binary_values.append(row[binary_columns].values)
        feature_values.append(row[continuous_columns].values)
        
    return processed_data, stock_ids, years, binary_values, feature_values

processed_data, stock_ids, years, binary_values, feature_values = prepare_data_for_llama(df)

class FinancialDataset(Dataset):
    def __init__(self, processed_data, stock_ids, years, binary_values, feature_values,
                 tokenizer,
                 company_embedding_layer, year_embedding_layer,
                 binary_embedding_layer, feature_embedding_layer):
        self.data = processed_data  # 토큰화된 전체 시퀀스 (labels 포함)
        self.stock_ids = stock_ids
        self.years = years
        self.binary_values = binary_values
        self.feature_values = feature_values
        self.tokenizer = tokenizer
        self.company_embedding_layer = company_embedding_layer
        self.year_embedding_layer = year_embedding_layer
        self.binary_embedding_layer = binary_embedding_layer
        self.feature_embedding_layer = feature_embedding_layer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item["input_ids"], dtype=torch.long),
            torch.tensor(item["attention_mask"], dtype=torch.long),
            torch.tensor(item["labels"], dtype=torch.long),
            # 추가 임베딩 정보
            torch.tensor([self.stock_ids[idx]], dtype=torch.long),
            torch.tensor([self.years[idx]], dtype=torch.long),
            torch.tensor(self.binary_values[idx], dtype=torch.float16),
            torch.tensor(self.feature_values[idx], dtype=torch.float16)
        )

def custom_collate_fn(batch):
    """
    각 배치의 토큰화 결과(input_ids, attention_mask, labels)를 동적 패딩하고,
    추가 임베딩 정보는 stack합니다.
    """
    input_ids_list, attention_mask_list, labels_list, stock_list, year_list, binary_list, feature_list = zip(*batch)
    
    batch_dict = {
        "input_ids": list(input_ids_list),
        "attention_mask": list(attention_mask_list)
    }
    data_collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
    padded = data_collator(batch_dict)
    
    # labels: pad with -100
    max_len = padded["input_ids"].shape[1]
    padded_labels = []
    for lab in labels_list:
        pad_length = max_len - len(lab)
        padded_lab = torch.tensor(lab.tolist() + [-100] * pad_length, dtype=torch.long)
        padded_labels.append(padded_lab)
    padded_labels = torch.stack(padded_labels)
    
    # 추가 임베딩 정보 (이미 고정 크기) stack
    stock_tensor = torch.stack(stock_list)
    year_tensor = torch.stack(year_list)
    binary_tensor = torch.stack(binary_list)
    feature_tensor = torch.stack(feature_list)
    
    return (
        padded["input_ids"].to(device),
        padded["attention_mask"].to(device),
        padded_labels.to(device),
        stock_tensor.to(device),
        year_tensor.to(device),
        binary_tensor.to(device),
        feature_tensor.to(device)
    )

print("Data prepared successfully.")
train_dataset = FinancialDataset(
    processed_data,
    stock_ids, years,
    binary_values, feature_values,
    tokenizer,
    company_embedding_layer, year_embedding_layer,
    binary_embedding_layer, feature_embedding_layer
)

train_loader = DataLoader(train_dataset, batch_size=16, num_workers=0, collate_fn=custom_collate_fn)

###############################################################################
# 8) 학습 루프
###############################################################################
epochs = 1
for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels, stock_tensor, year_tensor, binary_tensor, feature_tensor = batch
        
        # 각 배치마다 추가 임베딩 계산
        stock_emb = company_embedding_layer(stock_tensor)
        year_emb = year_embedding_layer(year_tensor)
        binary_emb = binary_embedding_layer(binary_tensor.unsqueeze(1))  # 차원 맞추기 위해 unsqueeze
        feature_emb = feature_embedding_layer(feature_tensor.unsqueeze(1))
        combined_embedding = torch.cat([stock_emb, year_emb, binary_emb, feature_emb], dim=-1)
        
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            combined_embedding=combined_embedding
        )
        
        loss = outputs.loss.mean()
        model_engine.backward(loss)
        model_engine.step()
        
        if batch_idx % 100 == 0:
            print(f"✅ Epoch {epoch+1}, Step {batch_idx}, Loss: {loss.item()}")

# 학습 루프 종료 후 모든 프로세스가 도달하도록 barrier 호출
torch.distributed.barrier()
print(f"[Rank {model_engine.local_rank}] Passed barrier after training.")

# checkpoint 저장 단계까지 모두 유지되도록 잠시 대기 (예: 10초)
import time
time.sleep(10)

# checkpoint 저장 전 추가 barrier: 모든 프로세스가 checkpoint 단계에 도달했음을 보장
torch.distributed.barrier()
print(f"[Rank {model_engine.local_rank}] All ranks reached pre-checkpoint barrier.")

# Rank 0에서 checkpoint 저장 시도
if model_engine.local_rank == 0:
    os.makedirs("trained_llama_model", exist_ok=True)
    try:
        model_engine.save_checkpoint("trained_llama_model")
        print("Model saved successfully on Rank 0.")
    except Exception as e:
        print(f"Checkpoint save failed on Rank 0: {e}")

# 저장 후에도 모든 프로세스가 barrier에서 동기화하도록 호출
torch.distributed.barrier()
print(f"[Rank {model_engine.local_rank}] Passed barrier after checkpoint save.")

# 저장 완료를 기다리기 위해 5분(300초) 동안 대기
exit_wait_time = 300  # 300초 = 5분
print(f"[Rank {model_engine.local_rank}] Waiting {exit_wait_time} seconds before exit.")
time.sleep(exit_wait_time)
print(f"[Rank {model_engine.local_rank}] Exiting process.")

