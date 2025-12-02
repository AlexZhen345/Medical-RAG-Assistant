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

## 📂 项目结构 (Directory Structure)

```text
Medical-RAG-Assistant/
├── app.py                  # 🚀 启动入口 (Gradio 界面)
├── config.py               # ⚙️ 全局配置文件 (路径管理)
├── requirements.txt        # 📦 依赖包列表
├── data/                   # 💾 知识库数据 (jsonl) 与 评测数据 (xlsx)
├── scripts/                # 🛠️ 工具脚本 (模型下载)
├── src/                    # 🧱 核心源码
│   ├── inference.py        # LLM 推理引擎 (适配 Qwen3)
│   ├── rag.py              # RAG 检索与索引构建
│   └── verifier.py         # 幻觉检测裁判模块
└── README.md               # 📖 项目说明书
🚀 快速开始（快速开始）
1.环境准备
推荐使用Conda创建独立环境：

巴什

conda create -n medical_rag python=3.10
conda activate medical_rag

# 安装项目依赖
pip install -r requirements.txt
2.一键下载模型
本项目依赖三个模型（Embedding、NLI、LLM）。我已经将相当好的大模型上传至 ModelScope，你可以直接运行脚本一键下载所有模型：

巴什

python scripts/download_models.py
注：大模型AlexZhen/Medical-Qwen3-4b-2507-Fintuned约占用显存6GB+，请确保您有足够的GPU资源。

3.启动系统
巴什

python app.py
启动成功后，点击终端显示的URL（如http://127.0.0.1:7860）即可在浏览器中使用。

🧠关于模型模型（关于模型）
本使用的核心大模型是我基于系统Qwen3-4B-Instruct-2507进行SFT的产物。

模型名称：Medical-Qwen3-4b-2507-Fintuned

托管地址:模型范围链接

训练数据：包含执业医师资格考试、考研西医综合等40k+条带详细解析的题目。

训练目标：强化模型对病例分析、鉴别诊断和医学术语的理解能力。

⚠️ 免责声明（免责声明）
非医疗建议：本项目生成的答案用于医学科研、学习和技术交流使用，可作为临床诊断或治疗的参考。若身体状况不佳，请前往正规医院就诊。

幻觉风险：尽管经过参数和RAG增强，大语言模型仍可能输出错误信息。请参考界面上的“事实一致性评分”严谨判断。

数据隐私：请勿在对话中输入真实的患者姓名、身份证号等隐私信息。

🤝致谢
谢谢奎文团队提供的基础模型以及模型范围提供模型托管服务。
