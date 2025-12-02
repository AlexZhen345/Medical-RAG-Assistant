Markdown

# 🏥 Medical RAG Assistant (医学智能问答助手)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.0-orange?logo=gradio)](https://gradio.app/)
[![ModelScope](https://img.shields.io/badge/ModelScope-Download-purple)](https://modelscope.cn/models/AlexZhen/Medical-Qwen3-4b-2507-Finetuned)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

一个基于 **RAG (检索增强生成)** 技术的垂直领域医学问答系统。

本项目集成了我在 **40,000+ 条中文医疗选择题与解析数据**上全量微调的 **Qwen3-4B** 大模型，结合 **Faiss** 向量库进行本地知识检索，并内置 **NLI (自然语言推理)** 模块对 AI 回答进行幻觉检测，旨在提供精准、可信的医学咨询服务。

## ✨ 核心亮点 (Key Features)

- **🧠 专属医学大脑**: 搭载 [Medical-Qwen3-4b-2507-Finetuned](https://modelscope.cn/models/AlexZhen/Medical-Qwen3-4b-2507-Finetuned)，经过 4万+ 条专业试题深度微调，具备强大的临床逻辑推理能力。
- **🔍 RAG 精准检索**: 使用 `text2vec` 模型 + Faiss 向量库，利用本地医学知识库（如诊疗指南、百科）修正模型回答。
- **🛡️ 独创幻觉检测**: 内置 Verifier（裁判）模块，利用 NLI 模型实时核查 "AI回答" 与 "检索资料" 的一致性，并给出可信度打分。
- **⚖️ 直观对比模式**: 界面同时展示“RAG 增强回答”与“普通回答”，让你一眼看出知识库修正的效果。
- **⚡ 开箱即用**: 提供全自动模型下载脚本，一键部署所有依赖模型。

## 📂 目录结构 (Directory Structure)

```text
Medical-RAG-Assistant/
├── app.py                  # 🚀 启动入口 (Gradio 界面)
├── config.py               # ⚙️ 全局配置文件 (路径管理)
├── requirements.txt        # 📦 依赖包列表
├── scripts/                # 🛠️ 工具脚本
│   └── download_models.py  # 📥 模型下载脚本
├── src/                    # 🧱 核心源码
│   ├── inference.py        # LLM 推理引擎
│   ├── rag.py              # RAG 检索与索引构建
│   └── verifier.py         # 幻觉检测裁判模块
├── data/                   # 💾 数据文件夹
│   ├── val.jsonl           # 知识库源数据
│   └── hallucination_labeled.xlsx # 裁判训练数据
│
│   # --- 以下文件夹由脚本自动生成或下载 ---
├── merged_qwen3_medical/   # 🧠 [核心大模型] Qwen3 微调版
├── models/                 # 🤖 [基础模型] Embedding 和 NLI 模型
└── rag_index_optimized/    # 🗂️ [索引文件] Faiss 向量索引
🚀 快速开始 (Quick Start)
1. 环境准备
推荐使用 Conda 创建独立环境：

Bash

conda create -n medical_rag python=3.10
conda activate medical_rag

# 安装项目依赖
pip install -r requirements.txt
2. 一键下载模型
本项目依赖三个模型（Embedding, NLI, LLM）。 我已经将微调好的大模型上传至 ModelScope，你可以直接运行脚本一键下载所有模型：

Bash

python scripts/download_models.py
注: 大模型 AlexZhen/Medical-Qwen3-4b-2507-Finetuned 约占用显存 6GB+，请确保你有足够的 GPU 资源。

3. 启动系统
Bash

python app.py
启动成功后，点击终端显示的 URL (如 http://127.0.0.1:7860) 即可在浏览器中使用。首次运行会自动基于 data/val.jsonl 构建向量索引。

🧠 关于微调模型 (About the Model)
本系统使用的核心大模型是我基于 Qwen3-4B-Instruct-2507 进行 SFT 微调的产物。

模型名称: Medical-Qwen3-4b-2507-Finetuned

托管地址: ModelScope 链接

训练数据: 包含执业医师资格考试、考研西医综合等 40k+ 条带详细解析的题目。

训练目标: 强化模型对病例分析、鉴别诊断和医学术语的理解能力。

⚠️ 免责声明 (Disclaimer)
非医疗建议: 本项目生成的回答仅供医学科研、学习和技术交流使用，绝不可作为临床诊断或治疗的依据。如有身体不适，请前往正规医院就诊。

幻觉风险: 尽管经过微调和 RAG 增强，大语言模型仍可能输出错误信息。请参考界面上的“事实一致性评分”谨慎判断。

数据隐私: 请勿在对话中输入真实的患者姓名、身份证号等隐私信息。

🤝 致谢
感谢 Qwen 团队提供的基座模型以及 ModelScope 提供的模型托管服务。
