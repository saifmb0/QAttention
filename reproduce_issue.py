import torch
from eagle.model.ea_model import EaModel
import time

base_model = "/home/202311016/models/Meta-Llama-3-8B-Instruct-W4A16-g128-Rot"
eagle_model = "/home/202311016/models/EAGLE-LLaMA3-Instruct-8B-on-W4A16-Rot"

try:
    model = EaModel.from_pretrained(
        use_eagle3=True,
        base_model_path=base_model,
        ea_model_path=eagle_model,
        total_token=60,
        depth=7,
        top_k=10,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    )
    print("Successfully loaded model")
except Exception as e:
    print(f"Failed to load model: {e}")
    import traceback
    traceback.print_exc()
