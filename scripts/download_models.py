from modelscope.hub.snapshot_download import snapshot_download

# 1. 下载 RAG 用的 Embedding 模型 (存到 models/text2vec 文件夹)
print("⬇️ 正在下载 Embedding 模型...")
snapshot_download("zjwan461/shibing624_text2vec-base-chinese", cache_dir="./models/text2vec")

# 2. 下载 Checker 用的 NLI 模型 (存到 models/nli 文件夹)
print("⬇️ 正在下载 NLI 模型...")
snapshot_download("Fengshenbang/Erlangshen-RoBERTa-110M-NLI", cache_dir="./models/nli")

print("✅ 下载完成！")