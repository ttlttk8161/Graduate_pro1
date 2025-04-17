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

#############################################
# 1. DeepSpeed 및 디바이스 초기화
#############################################
deepspeed.init_distributed()
local_rank = int(os.getenv("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

print(f"Local Rank: {local_rank}")
print(f"World Size: {os.getenv('WORLD_SIZE')}")
print(f"CUDA Device: {torch.cuda.current_device()}")
print(f"CUDA Device Name: {torch.cuda.get_device_name(local_rank)}")
cudnn.benchmark = True

#############################################
# 2. DeepSpeed 설정
#############################################
ds_config = {
    "fp16": {
        "enabled": True,
        "loss_scale": 0,                     
        "loss_scale_window": 1000,
        "initial_scale_power": 12,
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

with open('ds_config.json', 'w') as f:
    json.dump(ds_config, f)

#############################################
# 3. 데이터 로드 및 전처리
#############################################
file_path_csv = "data.csv"
df = pd.read_csv(file_path_csv)
df = df.drop(columns=["Name"], errors="ignore")
if "YEAR" not in df.columns:
    raise KeyError("YEAR 컬럼이 데이터셋에 없습니다.")

# 숫자 특성 및 이진 특성 컬럼 정의
binary_columns = ["BIG4", "LOSS"]
continuous_columns = ["OWN", "FORN", "SIZE", "LEV", "CUR", "GRW", "ROA", "ROE", "CFO", "PPE", "AGE", "INVREC", "MB", "TQ"]

# 연속형 데이터 scaling
scaler = MinMaxScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

# Stock 컬럼 factorize 및 연도 보정
df["Stock"], unique_stocks = pd.factorize(df["Stock"])
df["YEAR"] = df["YEAR"] - df["YEAR"].min()
num_stocks = len(unique_stocks)
year_count = df["YEAR"].nunique()

#############################################
# 4. 추가 임베딩 레이어 정의
#############################################
company_embedding_layer = nn.Embedding(num_stocks, 16)
year_embedding_layer = nn.Embedding(year_count, 8)
binary_embedding_layer = nn.Linear(len(binary_columns), 8)
feature_embedding_layer = nn.Linear(len(continuous_columns), 16)

#############################################
# 5. Advanced Llama 스타일 Cross Attention Block (Residual Branch 포함)
#############################################
class AdvancedLlamaCrossBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        """
        구성:
          (a) LayerNorm → Cross Attention → [Residual + 추가 residual branch]  
          (b) LayerNorm → FeedForward → Residual
        """
        super().__init__()
        # (a) Cross Attention 부분
        self.layer_norm_attn = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        # 추가 residual branch: cross attention 결과에 대해 2-layer MLP 적용
        self.attn_residual_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # (b) FeedForward 부분 (Llama 스타일)
        self.layer_norm_ff = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size)
        )
        self.dropout_ff = nn.Dropout(dropout)
    
    def forward(self, hidden_states, context):
        # (a) Cross Attention Branch
        normed = self.layer_norm_attn(hidden_states)
        attn_out, _ = self.cross_attn(query=normed, key=context, value=context)
        attn_out = self.dropout_attn(attn_out)
        # 기존 residual과 추가 residual branch 결과 합산
        attn_residual = self.attn_residual_branch(attn_out)
        hidden_states = hidden_states + attn_out + attn_residual
        
        # (b) FeedForward Branch
        normed_ff = self.layer_norm_ff(hidden_states)
        ff_out = self.feed_forward(normed_ff)
        ff_out = self.dropout_ff(ff_out)
        hidden_states = hidden_states + ff_out
        
        return hidden_states

#############################################
# 6. Full Model 정의 (aux loss 포함)
#############################################
model_name = "meta-llama/Llama-3.2-3B-Instruct"
config = AutoConfig.from_pretrained(model_name)
config.output_hidden_states = True
config.return_dict_in_generate = True

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    config=config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
# Weight Tying: lm_head와 embed_tokens의 가중치 공유
base_model.lm_head.weight = base_model.model.embed_tokens.weight

class FullModelWithAuxLoss(nn.Module):
    def __init__(self, base_model, company_emb, year_emb, binary_emb, feature_emb, num_cross_layers=3, aux_loss_weight=0.5):
        """
        - 숫자 특성(회사, 연도, 이진, 연속형: 총 48차원)을 텍스트 임베딩 차원으로 투영 후,  
          Advanced Llama 스타일 Cross Attention Block을 여러 개 적용
        - aux_loss_weight: 보조 loss(MSE)의 가중치
        """
        super().__init__()
        self.model = base_model
        self.company_embedding_layer = company_emb
        self.year_embedding_layer = year_emb
        self.binary_embedding_layer = binary_emb
        self.feature_embedding_layer = feature_emb
        # projection: 48차원 -> hidden_size
        self.embedding_projection = nn.Linear(48, base_model.config.hidden_size)
        # 여러 Advanced Llama Cross Attention 블록 적용
        self.cross_attention_layers = nn.ModuleList([
            AdvancedLlamaCrossBlock(base_model.config.hidden_size, num_heads=base_model.config.num_attention_heads)
            for _ in range(num_cross_layers)
        ])
        self.aux_loss_weight = aux_loss_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None, combined_embedding=None):
        # 기본 토큰 임베딩 추출
        token_embeddings = self.model.model.embed_tokens(input_ids)  # [batch, seq_len, hidden_size]
        
        aux_loss = 0.0
        if combined_embedding is not None:
            # 숫자 특성 임베딩 투영
            numeric_emb = self.embedding_projection(combined_embedding)  # [batch, hidden_size]
            # context로 사용하기 위해 차원 확장
            numeric_context = numeric_emb.unsqueeze(1)  # [batch, 1, hidden_size]
            # Advanced Cross Attention Block을 여러 번 적용
            for cross_layer in self.cross_attention_layers:
                token_embeddings = cross_layer(token_embeddings, numeric_context)
            
            # aux loss 예시: 마지막 hidden state의 평균과 numeric_emb 간 MSE
            hidden_state_summary = token_embeddings.mean(dim=1)  # [batch, hidden_size]
            aux_loss = self.mse_loss(numeric_emb, hidden_state_summary)
        
        # 수정된 임베딩을 transformer에 전달
        outputs = self.model(
            inputs_embeds=token_embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        main_loss = outputs.loss
        total_loss = main_loss + self.aux_loss_weight * aux_loss if labels is not None else main_loss
        outputs.loss = total_loss
        return outputs

full_model = FullModelWithAuxLoss(
    base_model,
    company_embedding_layer,
    year_embedding_layer,
    binary_embedding_layer,
    feature_embedding_layer,
    num_cross_layers=3,      # 원하는 cross attention 레이어 수
    aux_loss_weight=0.5      # aux loss 가중치 조절
)

#############################################
# 7. DeepSpeed 초기화
#############################################
model_engine, optimizer, _, scheduler = deepspeed.initialize(
    model=full_model,
    config=ds_config,
    model_parameters=full_model.parameters(),
    dist_init_required=True
)

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<pad>"

#############################################
# 8. Dataset 및 DataLoader 정의
#############################################
def prepare_data_for_llama(df):
    processed_data = []
    stock_ids = []
    years = []
    binary_values = []
    feature_values = []
    
    for _, row in df.iterrows():
        current_year_metrics = ", ".join([f"{col}: {row[col]:.4f}" for col in (continuous_columns + binary_columns)])
        next_year = row["YEAR"] + 1
        next_year_row = df[(df["Stock"] == row["Stock"]) & (df["YEAR"] == next_year)]
        if next_year_row.empty:
            continue
        ny_row = next_year_row.iloc[0]
        next_year_metrics = ", ".join([f"{col}: {ny_row[col]:.4f}" for col in (continuous_columns + binary_columns)])
        
        prompt = (
            "# Instruction:\n"
            "Given the financial statement data below, predict the financial statement for the next year.\n\n"
            "# Input:\n"
            f"Company: {row['Stock']}\n"
            f"Year: {int(row['YEAR'])}\n"
            f"Metrics: {current_year_metrics}\n\n"
            "# Answer:\n"
        )
        output_text = (
            f"Company: {row['Stock']}\n"
            f"Year: {int(next_year)}\n"
            f"Metrics: {next_year_metrics}"
        )
        full_text = prompt + output_text
        tokenized = tokenizer(full_text, truncation=True, max_length=512)
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
    def __init__(self, processed_data, stock_ids, years, binary_values, feature_values, tokenizer,
                 company_embedding_layer, year_embedding_layer, binary_embedding_layer, feature_embedding_layer):
        self.data = processed_data
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
            torch.tensor([self.stock_ids[idx]], dtype=torch.long),
            torch.tensor([self.years[idx]], dtype=torch.long),
            torch.tensor(self.binary_values[idx], dtype=torch.float16),
            torch.tensor(self.feature_values[idx], dtype=torch.float16)
        )

def custom_collate_fn(batch):
    input_ids_list, attention_mask_list, labels_list, stock_list, year_list, binary_list, feature_list = zip(*batch)
    batch_dict = {
        "input_ids": list(input_ids_list),
        "attention_mask": list(attention_mask_list)
    }
    data_collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
    padded = data_collator(batch_dict)
    
    max_len = padded["input_ids"].shape[1]
    padded_labels = []
    for lab in labels_list:
        pad_length = max_len - len(lab)
        padded_lab = torch.tensor(lab.tolist() + [-100] * pad_length, dtype=torch.long)
        padded_labels.append(padded_lab)
    padded_labels = torch.stack(padded_labels)
    
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
    processed_data, stock_ids, years, binary_values, feature_values,
    tokenizer,
    company_embedding_layer, year_embedding_layer, binary_embedding_layer, feature_embedding_layer
)
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=0, collate_fn=custom_collate_fn)

#############################################
# 9. 학습 루프
#############################################
epochs = 15
for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels, stock_tensor, year_tensor, binary_tensor, feature_tensor = batch
        # 추가 임베딩 정보 결합
        stock_emb = company_embedding_layer(stock_tensor)
        year_emb = year_embedding_layer(year_tensor)
        binary_emb = binary_embedding_layer(binary_tensor.unsqueeze(1))
        feature_emb = feature_embedding_layer(feature_tensor.unsqueeze(1))
        combined_embedding = torch.cat([stock_emb, year_emb, binary_emb, feature_emb], dim=-1).squeeze(1)
        
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            combined_embedding=combined_embedding
        )
        loss = outputs.loss.mean()
        model_engine.backward(loss)
        model_engine.step()

        # GPU 캐시를 비워 메모리 단편화를 줄임
        torch.cuda.empty_cache()
        
        if batch_idx % 100 == 0:
            print(f"✅ Epoch {epoch+1}, Step {batch_idx}, Loss: {loss.item()}")


torch.distributed.barrier()
print(f"[Rank {model_engine.local_rank}] All ranks reached pre-checkpoint barrier.")
try:
    model_engine.save_checkpoint("final_llama_model")
    if model_engine.local_rank == 0:
        print("✅ Model saved successfully on Rank 0.")
except Exception as e:
    print(f"❌ Checkpoint save failed on Rank {model_engine.local_rank}: {e}")
torch.distributed.barrier()
print(f"[Rank {model_engine.local_rank}] Passed post-checkpoint barrier.")
