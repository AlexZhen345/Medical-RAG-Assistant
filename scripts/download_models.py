from modelscope.hub.snapshot_download import snapshot_download
import os

print("🚀 开始一键下载所有模型...")

# 1. 下载 RAG 用的 Embedding 模型
# 存到 models/text2vec 文件夹
print("\n⬇️ [1/3] 正在下载 Embedding 模型 (text2vec)...")
snapshot_download(
    "zjwan461/shibing624_text2vec-base-chinese", 
    cache_dir="./models/text2vec"
)

# 2. 下载 Checker 用的 NLI 模型
# 存到 models/nli 文件夹
print("\n⬇️ [2/3] 正在下载 NLI 验证模型...")
snapshot_download(
    "Fengshenbang/Erlangshen-RoBERTa-110M-NLI", 
    cache_dir="./models/nli"
)

# 3. 下载你的微调大模型 (LLM)
# 存到根目录的 merged_qwen3_medical 文件夹
# 这里的 ID 是你刚才在 ModelScope 创建的那个
print("\n⬇️ [3/3] 正在下载微调大模型 (AlexZhen/Medical-Qwen3)...")
snapshot_download(
    "AlexZhen/Medical-Qwen3-4b-2507-Finetuned", 
    cache_dir="./merged_qwen3_medical"
)

print("\n✅ 所有模型下载完成！")
print("请确保 config.py 中的 LLM_MODEL_PATH 指向了 ./merged_qwen3_medical 里的具体模型文件夹")
