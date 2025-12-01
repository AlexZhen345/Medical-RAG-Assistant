import sys
import os
import time

# --- 1. 核心路径配置 (确保能导入 src 和 config) ---
# 获取当前文件所在的目录 (项目根目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录加入到 Python 的搜索路径中，这样才能导入 config.py 和 src 包
sys.path.append(current_dir)

# 导入配置文件 (一定要有 config.py)
try:
    import config
except ImportError:
    raise ImportError("❌ 找不到 config.py！请确保你已经按照教程创建了 config.py 文件。")

import gradio as gr

# 从 src 包中导入模块
from src.rag import MedicalRAG
from src.inference import get_medical_answer, get_normal_answer
from src.verifier import HallucinationVerifier, load_or_process_data, train_verifier

# --- 2. 业务逻辑管道 ---
def rag_pipeline(question, top_k=3):
    """
    处理用户请求的主管道：检索 -> 生成 -> 验证
    """
    # 记录开始时间
    start_time = time.time()

    # 1. 检索相关知识 (RAG)
    # 调用 RAG 模块，找回最相关的 top_k 条知识
    response = embedding_model.rag_retrieve(question, top_k=top_k, return_full_answer=True)
    
    if response['success']:
        knowledge_contexts = [result.get('answer') for result in response['results']]
        # 格式化参考资料文本，用于展示
        contexts_text = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(knowledge_contexts)])
    else:
        knowledge_contexts = []
        contexts_text = "未检索到相关知识。"

    # 2. 生成回答 (LLM)
    context_str = "\n".join(knowledge_contexts) # 拼接待会儿要喂给模型的上下文
    
    # 试卷A：医学回答 (基于RAG)
    answer1 = get_medical_answer(question, context_str)
    # 试卷B：普通回答 (裸奔)
    answer2 = get_normal_answer(question, "")

    # 3. 事实一致性检查 (Verifier)
    # 只有当有参考资料时，检查才有意义
    if knowledge_contexts:
        consistency_result = verifier.verify(question, answer1, context_str)
        # 提取分数和标签
        score_val = consistency_result.get('fact_consistency_score', 0)
        consistency_msg = f"{score_val:.2f} ({consistency_result.get('label', 'unknown')})"
        
        # 如果分数太低，加个警告
        if score_val < 0.5:
            consistency_msg += " ⚠️ 警告：可信度低！"
    else:
        consistency_msg = "N/A (无参考资料)"

    # 计算总耗时
    total_time = round(time.time() - start_time, 2)

    # 4. 最终结果拼装
    final_output = (
        f"### 💊 医学回答 (RAG增强)\n{answer1}\n\n"
        f"---\n"
        f"### 🤖 普通回答 (无知识库)\n{answer2}\n\n"
        f"---\n"
        f"### 📚 检索到的知识片段\n{contexts_text}\n\n"
        f"---\n"
        f"### ⚖️ 事实一致性评分\n{consistency_msg}\n\n"
        f"⏱️ 总处理时间: {total_time}秒"
    )
    
    return final_output


# --- 3. 程序入口 ---
if __name__ == "__main__":
    print("\n🚀 正在启动 Medical RAG System...")
    print(f"📂 项目根目录: {config.ROOT_DIR}")

    # --- 初始化组件 A: RAG 检索模块 ---
    print("正在加载 RAG 模块...")
    
    # 智能检查：如果索引文件夹不存在，或者里面是空的，就先构建索引
    if not os.path.exists(config.INDEX_DIR) or not os.listdir(config.INDEX_DIR):
        print(f"⚠️ 未检测到有效索引，正在从 {config.VAL_DATA_PATH} 构建...")
        if os.path.exists(config.VAL_DATA_PATH):
            temp_rag = MedicalRAG(model_name=config.EMBEDDING_MODEL_PATH)
            temp_rag.load_knowledge_from_jsonl(config.VAL_DATA_PATH)
            temp_rag.build_index(save_path=config.INDEX_DIR)
            print("✅ 索引构建完成！")
        else:
            print(f"❌ 错误：找不到数据文件 {config.VAL_DATA_PATH}，无法构建索引。")
    
    # 正式加载 RAG
    embedding_model = MedicalRAG(
        model_name=config.EMBEDDING_MODEL_PATH,
        index_path=config.INDEX_DIR
    )

    # --- 初始化组件 B: Verifier 验证模块 ---
    print("正在加载 Verifier 模块...")
    
    # 加载并训练裁判数据
    if os.path.exists(config.HALLUCINATION_DATA_PATH):
        df = load_or_process_data(
            data_path=config.HALLUCINATION_DATA_PATH,
            processed_path=config.HALLUCINATION_PROCESSED_PATH
        )
        train_verifier(df)
        
        # 实例化裁判
        verifier = HallucinationVerifier(
            embed_model=config.EMBEDDING_MODEL_PATH,
            nli_model=config.NLI_MODEL_PATH
        )
    else:
        print(f"⚠️ 警告：找不到 {config.HALLUCINATION_DATA_PATH}，验证功能可能不可用。")
        verifier = None # 防止报错

    print("✅ 系统初始化完成！正在启动网页界面...")

    # --- 4. Gradio 界面搭建 ---
    with gr.Blocks(title="医学智能问答系统") as demo:
        gr.Markdown("# 🏥 Medical RAG - 医学智能问答助手")
        gr.Markdown("本项目基于 RAG 技术 + Qwen 大模型，提供准确的医学知识问答，并内置幻觉检测机制。")

        with gr.Row():
            # 左侧输入区
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="👩‍⚕️ 请输入你的医学问题",
                    placeholder="例如：感冒了能吃阿司匹林吗？糖尿病有什么忌口？",
                    lines=3
                )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=5, value=3, step=1, 
                        label="检索知识条数 (Top-K)"
                    )
                    
                submit_btn = gr.Button("🔍 开始分析", variant="primary", size="lg")
                
                # 示例问题
                gr.Examples(
                    examples=[
                        ["糖尿病的症状包括发烧吗？"],
                        ["肺结核是由什么病原体导致的？"],
                        ["高血压患者应该注意饮食吗？"],
                        ["感冒了吃什么药好得快？"]
                    ],
                    inputs=question_input
                )

            # 右侧输出区
            with gr.Column(scale=6):
                answer_output = gr.Markdown(label="📋 分析报告")

        # 绑定事件
        submit_btn.click(
            fn=rag_pipeline,
            inputs=[question_input, top_k_slider],
            outputs=answer_output
        )
        
        # 支持回车提交
        question_input.submit(
            fn=rag_pipeline,
            inputs=[question_input, top_k_slider],
            outputs=answer_output
        )

    # 启动服务
    demo.launch(server_name="127.0.0.1", share=False, inbrowser=True)