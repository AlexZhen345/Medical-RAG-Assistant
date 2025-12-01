import os
import torch

# --- 1. 项目定位 ---
# 获取项目根目录 (即 config.py 所在的目录)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

GRADIO_TEMP_DIR = os.path.join(ROOT_DIR, "gradio_temp")
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

# --- 2. 路径配置 (Path Config) ---
# 数据文件夹
DATA_DIR = os.path.join(ROOT_DIR, "data")
# 模型文件夹
MODELS_DIR = os.path.join(ROOT_DIR, "models")
# 向量索引文件夹
INDEX_DIR = os.path.join(ROOT_DIR, "rag_index_optimized")

# --- 3. 具体文件路径 ---
# 验证集路径 (RAG构建索引要用)
VAL_DATA_PATH = os.path.join(DATA_DIR, "val.jsonl")
# 幻觉检测数据路径
HALLUCINATION_DATA_PATH = os.path.join(DATA_DIR, "hallucination_labeled.xlsx")
HALLUCINATION_PROCESSED_PATH = os.path.join(DATA_DIR, "hallucination_labeled_processed.xlsx")

# --- 4. 模型路径 ---
# Embedding 模型 (注意：根据你的tree结构，它在 models/text2vec)
EMBEDDING_MODEL_PATH = os.path.join(MODELS_DIR, "text2vec")

# LLM 大模型 (注意：这个在根目录 merged_qwen3_medical)
LLM_MODEL_PATH = os.path.join(ROOT_DIR, "merged_qwen3_medical")

# NLI 判别模型 (注意：根据tree结构，它在 models/nli)
NLI_MODEL_PATH = os.path.join(MODELS_DIR, "nli")

# --- 5. 运行参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"