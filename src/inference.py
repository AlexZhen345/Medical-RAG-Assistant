import sys
import os

# --- 1. 路径设置 (确保能导入根目录的 config.py) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import config  # 导入配置文件
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 2. 加载模型与分词器 ---
MERGED_MODEL_PATH = config.LLM_MODEL_PATH

print(f"🔄 [Inference] 正在加载大模型: {MERGED_MODEL_PATH} ...")

if not os.path.exists(MERGED_MODEL_PATH):
    # 如果找不到，尝试使用 config 里定义的原始路径，或者是 HuggingFace ID
    print(f"⚠️ 警告: 路径 {MERGED_MODEL_PATH} 不存在。")
    print("请检查 config.py 配置，或确认模型文件已放入指定文件夹。")

try:
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MERGED_MODEL_PATH,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        torch_dtype=torch.bfloat16,  # 显存优化
        device_map=config.DEVICE,    # 使用 config 中的设备 (cuda/cpu)
        trust_remote_code=True
    )
    model.eval() # 设置为评估模式
    print("✅ [Inference] 大模型加载完成！")

except Exception as e:
    print(f"❌ [Inference] 模型加载失败: {e}")
    # 这里不抛出异常，防止整个 app 崩溃，但在实际调用时会报错
    model = None
    tokenizer = None


# --- 3. 核心功能函数 ---

def _build_prompt_messages(user_question: str, rag_context: str) -> list:
    """构建 RAG 模式的提示词 (System Prompt)"""
    system_content = f"""
你是一个专业医学助手。你的任务是基于提供的信息，对用户的问题给出准确、相关且简洁的回答，你的语气应让你显得专业且有同理心。请遵循以下指南：
1. 请使用【参考材料】中的信息来回答问题，如果【参考材料】中的信息能直接回答问题，则尽量使用原文语段，并且之后应当补充上你所知的其他信息。
2. 如果【参考材料】中的信息不足以回答问题，则你应该回答“我不知道”。
3. 在你生成回答时，应保持回答的连贯性和逻辑性。
4. 回答中绝对不能包括“请给出正确答案并说明理由”这段话。
---
【参考资料】
{rag_context}
---
"""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_question}
    ]
    return messages


def _build_normal_messages(user_question: str, rag_context: str) -> list:
    """构建普通模式的提示词"""
    system_content = f"""你是一个专业的医学助手。请回答问题。"""
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_question}
    ]
    return messages


def get_medical_answer(user_question: str, rag_context: str) -> str:
    """获取医学回答 (RAG增强版)"""
    if model is None: return "❌ 模型未成功加载，无法回答。"
    
    try:
        messages = _build_prompt_messages(user_question, rag_context)
        
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device) 
        
        generate_ids = model.generate(
            inputs,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        input_length = inputs.shape[1]
        response = tokenizer.decode(
            generate_ids[0][input_length:], 
            skip_special_tokens=True
        )
        return response

    except Exception as e:
        print(f"推理时发生错误: {e}")
        return "抱歉，模型在回答时遇到了一个内部错误。"


def get_normal_answer(user_question: str, rag_context: str) -> str:
    """获取普通回答 (裸奔版)"""
    if model is None: return "❌ 模型未成功加载，无法回答。"

    try:
        # 普通回答不需要 rag_context，传空字符串或忽略即可
        messages = _build_normal_messages(user_question, "")

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        generate_ids = model.generate(
            inputs,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        input_length = inputs.shape[1]
        response = tokenizer.decode(
            generate_ids[0][input_length:],
            skip_special_tokens=True
        )
        return response

    except Exception as e:
        print(f"推理时发生错误: {e}")
        return "抱歉，模型在回答时遇到了一个内部错误。"


# --- 4. 自我测试 (当直接运行此文件时) ---
if __name__ == "__main__":
    print("\n--- 正在执行自我测试... ---")
    
    dummy_context = """
    阿司匹林（Aspirin）是一种水杨酸盐药物，常用于治疗疼痛、发热和炎症。
    它还可以通过抑制血小板聚集来预防心脏病发作和中风。
    常见副作用包括胃肠道不适、恶心和出血风险增加。
    """
    dummy_question = "阿司匹林有什么用？它有什么副作用？"
    
    print(f"测试问题: {dummy_question}")
    
    if model:
        answer = get_medical_answer(dummy_question, dummy_context)
        print("\n--- 模型的回答 ---")
        print(answer)
        print("------------------")
    else:
        print("❌ 模型未加载，跳过推理测试。")
    
    print("\n--- 自我测试完成 ---")