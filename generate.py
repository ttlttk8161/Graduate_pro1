# import os
# import re
# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

# # ✅ 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ 학습된 모델 및 토크나이저 로드
# model_path = "trained_llama_model"
# model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
# tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

# # ✅ 학습된 임베딩 불러오기
# num_total_companies = 1284  
# hidden_dim = model.config.hidden_size  

# # ✅ Feature & Binary 컬럼 정의
# continuous_columns = ["OWN", "FORN", "SIZE", "LEV", "CUR", "GRW", "ROA", "ROE", "CFO", "PPE", "AGE", "INVREC", "MB", "TQ"]
# binary_columns = ["BIG4", "LOSS"]

# # ✅ 임베딩 레이어 정의 및 가중치 로드
# company_embedding_layer = nn.Embedding(num_embeddings=num_total_companies, embedding_dim=hidden_dim).to(device)
# year_embedding_layer = nn.Embedding(num_embeddings=2023 - 2011 + 1, embedding_dim=hidden_dim).to(device)
# binary_embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=hidden_dim).to(device)
# feature_embedding_layer = nn.Linear(len(continuous_columns), hidden_dim, bias=True).to(device)

# company_embedding_layer.load_state_dict(torch.load("trained_llama_model/company_embedding.pth", map_location=device))
# year_embedding_layer.load_state_dict(torch.load("trained_llama_model/year_embedding.pth", map_location=device))
# binary_embedding_layer.load_state_dict(torch.load("trained_llama_model/binary_embedding.pth", map_location=device))
# feature_embedding_layer.load_state_dict(torch.load("trained_llama_model/feature_embedding.pth", map_location=device))

# # ✅ Dropout 추가
# dropout = nn.Dropout(0.1)

# # ✅ MinMaxScaler 로드
# data_path = "data.csv"
# df_original = pd.read_csv(data_path)
# scaler = MinMaxScaler()
# scaler.fit(df_original[continuous_columns])  

# # ✅ 생성할 회사 및 연도 설정
# num_fake_companies = 30  
# years = list(range(2011, 2024))  

# generated_data = []

# # ✅ 데이터 생성 함수
# def generate_financial_data(company, year):
#     """
#     특정 회사와 연도에 대한 재무 데이터를 생성하는 함수
#     """

#     # ✅ 기존 데이터 샘플링 없이 입력을 비워둠
#     input_text = f"Company: {company}, Year: {year}, " + ", ".join([f"{col}:" for col in continuous_columns + binary_columns])
#     inputs = tokenizer(input_text, return_tensors="pt").to(device)

#     # ✅ 모델이 직접 Feature 값을 생성하도록 유도
#     generated_ids = model.generate(
#         inputs["input_ids"],
#         num_return_sequences=1,
#         do_sample=True,
#         temperature=1.0,  # 🔄 샘플링 균형 유지
#         top_k=50,  
#         top_p=0.95,  
#         max_new_tokens=64,  # ✅ 기존 `max_length=128` 제거 후 `max_new_tokens`로 변경
#         pad_token_id=tokenizer.pad_token_id
#     )

#     generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#     print(f"🔍 Debug - Generated Text:\n{generated_text}")  

#     # ✅ 정규 표현식 개선 (공백 허용)
#     matches = re.findall(r"(\w+):\s*(-?\d*\.?\d+)", generated_text)
#     print(f"🔍 Debug - Extracted Matches: {matches}")  

#     if not matches:
#         print("🚨 Error: No matches found!")
#         return None, None  

#     # ✅ 모델이 생성한 데이터를 최종 Feature 값으로 사용
#     feature_values = {key: float(value) for key, value in matches}

#     return generated_text, feature_values


# for company_id in range(1, num_fake_companies + 1):
#     for year in years:
#         generated_text, extracted_values = generate_financial_data(company_id, year)

#         if extracted_values is None:
#             continue  

#         company_embedding = company_embedding_layer(torch.tensor(company_id % num_total_companies, dtype=torch.long, device=device)).unsqueeze(0)
#         company_embedding = dropout(company_embedding + torch.randn_like(company_embedding) * 0.2)  

#         year_embedding = year_embedding_layer(torch.tensor(year - 2011, dtype=torch.long, device=device)).unsqueeze(0)
#         year_embedding = dropout(year_embedding + torch.randn_like(year_embedding) * 0.1)

#         # ✅ 모델이 생성한 Feature 값 사용 (기본값 0.0 처리)
#         feature_values_list = [extracted_values.get(col, 0.0) for col in continuous_columns + binary_columns]

#         combined_embedding = company_embedding + year_embedding

#         generated_data.append([company_id, year] + feature_values_list)

# # ✅ 컬럼 개수 고정하여 DataFrame 생성
# expected_columns = ["Stock", "Year"] + continuous_columns + binary_columns
# df_generated = pd.DataFrame(generated_data, columns=expected_columns)

# # ✅ CSV 파일 저장
# df_generated.to_csv("generated_financial_data.csv", index=False)
# print("✅ 데이터 생성 완료! 저장 위치: generated_financial_data.csv")

# # ✅ 샘플 출력
# print(df_generated.head())

import os
import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 학습된 모델 및 토크나이저 로드
model_path = "trained_llama_model"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

# ✅ Feature & Binary 컬럼 정의
continuous_columns = ["OWN", "FORN", "SIZE", "LEV", "CUR", "GRW", "ROA", "ROE", "CFO", "PPE", "AGE", "INVREC", "MB", "TQ"]
binary_columns = ["BIG4", "LOSS"]

# ✅ 임베딩 레이어 정의 및 가중치 로드
num_total_companies = 1284  
hidden_dim = model.config.hidden_size  

company_embedding_layer = nn.Embedding(num_embeddings=num_total_companies, embedding_dim=hidden_dim).to(device)
year_embedding_layer = nn.Embedding(num_embeddings=2023 - 2011 + 1, embedding_dim=hidden_dim).to(device)
binary_embedding_layer = nn.Embedding(num_embeddings=2, embedding_dim=hidden_dim).to(device)
feature_embedding_layer = nn.Linear(len(continuous_columns), hidden_dim, bias=True).to(device)

company_embedding_layer.load_state_dict(torch.load("trained_llama_model/company_embedding.pth", map_location=device, weights_only=True))
year_embedding_layer.load_state_dict(torch.load("trained_llama_model/year_embedding.pth", map_location=device, weights_only=True))
binary_embedding_layer.load_state_dict(torch.load("trained_llama_model/binary_embedding.pth", map_location=device, weights_only=True))
feature_embedding_layer.load_state_dict(torch.load("trained_llama_model/feature_embedding.pth", map_location=device, weights_only=True))

# ✅ MinMaxScaler 로드 및 데이터 정규화
data_path = "data.csv"
df_original = pd.read_csv(data_path)

# 🔹 컬럼명을 대문자로 변환하여 KeyError 방지
df_original.columns = df_original.columns.str.upper()

# 🔹 YEAR 컬럼 존재 여부 확인
if "YEAR" not in df_original.columns:
    raise KeyError("🚨 'YEAR' 컬럼이 존재하지 않습니다. CSV 파일을 확인하세요!")

# ✅ MinMaxScaler 적용
scaler = MinMaxScaler()
df_original[continuous_columns] = scaler.fit_transform(df_original[continuous_columns])

# ✅ 2011년 평균값 계산 (이진 변수는 0 또는 1이므로 반올림)
avg_2011 = df_original[df_original["YEAR"] == 2011][continuous_columns].mean().to_dict()
binary_avg_2011 = df_original[df_original["YEAR"] == 2011][binary_columns].mode().iloc[0].to_dict()

# ✅ 노이즈 추가 (2011년 초기값에만 적용)
for col in continuous_columns:
    avg_2011[col] += np.random.normal(0, 0.01)

# ✅ 생성할 회사 및 연도 설정
num_fake_companies = 30  
years = list(range(2011, 2024))  
previous_data = {}  

generated_data = []

# ✅ 데이터 생성 함수
def generate_financial_data(company, year):
    """
    특정 회사와 연도에 대한 재무 데이터를 생성하는 함수
    """

    # 🔹 2011년 데이터는 평균값 사용
    if year == 2011:
        prev_values = avg_2011.copy()
        prev_values.update(binary_avg_2011)  
    else:
        prev_values = previous_data.get((company, year - 1), {col: 0 for col in continuous_columns + binary_columns})

    # ✅ 임베딩 적용
    company_tensor = torch.tensor([company], dtype=torch.long, device=device)
    year_tensor = torch.tensor([year - 2011], dtype=torch.long, device=device)

    company_embed = company_embedding_layer(company_tensor).squeeze(0)
    year_embed = year_embedding_layer(year_tensor).squeeze(0)

    binary_features = torch.tensor([prev_values[col] for col in binary_columns], dtype=torch.long, device=device)
    binary_embed = binary_embedding_layer(binary_features).mean(dim=0)

    continuous_features = torch.tensor([prev_values[col] for col in continuous_columns], dtype=torch.float32, device=device)
    feature_embed = feature_embedding_layer(continuous_features).squeeze(0)

    # ✅ 최종 임베딩 벡터
    final_embed = company_embed + year_embed + binary_embed + feature_embed

    prev_text = ", ".join([f"{col}: {prev_values[col]:.4f}" for col in continuous_columns + binary_columns])

    # ✅ 입력 텍스트
    input_text = f"This is the financial statement for this year  Company: {company}, Year: {year}, {prev_text}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # ✅ 모델이 직접 Feature 값을 생성하도록 유도
    generated_ids = model.generate(
        inputs["input_ids"],
        num_return_sequences=1,
        do_sample=True,
        temperature=1.2,  
        top_k=40,  
        top_p=0.90,  
        max_new_tokens=64,  
        pad_token_id=tokenizer.pad_token_id
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"🔍 Debug - Generated Text:\n{generated_text}")  

    # ✅ 정규 표현식 개선
    matches = re.findall(r"(\w+):\s*(-?\d*\.\d+)", generated_text)
    print(f"🔍 Debug - Extracted Matches: {matches}")  

    if not matches:
        print("🚨 Error: No matches found!")
        return None, None  

    # ✅ 모델이 생성한 데이터를 저장하여 다음 연도에 반영
    feature_values = {key: float(value) for key, value in matches}
    feature_values["LOSS"] = round(feature_values.get("LOSS", 0))

    previous_data[(company, year)] = feature_values
    
    return generated_text, feature_values

# ✅ 데이터 생성 및 저장
for company_id in range(1, num_fake_companies + 1):
    for year in years:
        generated_text, extracted_values = generate_financial_data(company_id, year)
        if extracted_values is None:
            continue  
        feature_values_list = [extracted_values.get(col, 0) for col in continuous_columns + binary_columns]
        generated_data.append([company_id, year] + feature_values_list)

df_generated = pd.DataFrame(generated_data, columns=["Stock", "Year"] + continuous_columns + binary_columns)
df_generated[continuous_columns] = scaler.inverse_transform(df_generated[continuous_columns])
df_generated.to_csv("generated_financial_data.csv", index=False)
print("✅ 데이터 생성 완료! 저장 위치: generated_financial_data.csv")
print(df_generated.head())
