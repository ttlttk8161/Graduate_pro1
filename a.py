# test_nccl_multi_gpu.py

import os
import torch
import torch.distributed as dist

def main():
    # 필수: DeepSpeed 또는 torchrun 사용 시 환경변수에서 LOCAL_RANK 사용
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # GPU 할당
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Distributed 초기화
    dist.init_process_group(backend="nccl")

    # 디버깅 정보 출력
    print(f"✅ Rank {dist.get_rank()} / Local Rank {local_rank} / World Size {world_size} / Device {torch.cuda.current_device()}")

    # all_reduce 테스트
    tensor = torch.ones(1).to(device) * (dist.get_rank() + 1)
    print(f"[Before AllReduce] Rank {dist.get_rank()} has tensor {tensor.item()}")
    dist.all_reduce(tensor)
    print(f"[After AllReduce] Rank {dist.get_rank()} has tensor {tensor.item()}")
    print("111111111111111111111111111111111111111111111111")

    # 정리
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
