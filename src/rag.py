"""
医学知识检索与RAG模块 - 优化版
作者: lyy (知识检索与RAG负责人)
功能: 专门针对医学问答数据集优化的检索模块
"""

import json
import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime
import re


class MedicalRAG:
    """
    医学知识检索类 - 针对问答数据集优化
    
    数据格式：
    {
        "instruction": "问题+选项",
        "input": "",
        "output": "答案+解析"
    }
    
    优化点：
    1. 智能提取问题核心内容
    2. 结合问题和答案构建知识库
    3. 针对医学术语优化检索
    """
    
    def __init__(self,  model_name='./models/text2vec', index_path: Optional[str] = None):
        """
        初始化RAG模块
        
        参数:
            model_name: 向量化模型
                - 默认: all-MiniLM-L6-v2 (轻量级，适合快速测试)
                - 医学专用: pr  ·1itamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
                - 中文医学: shibing624/text2vec-base-chinese-paraphrase
        """
        print(f"正在加载向量化模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        self.knowledge_base = []  # 存储知识片段
        self.embeddings = None    # 向量矩阵
        self.index = None         # FAISS索引

        # 新增：如果提供了索引路径，直接加载
        if index_path and os.path.exists(index_path):
            print(f"检测到预构建索引路径: {index_path}")
            self.load_index(index_path)

        print("RAG模块初始化完成！")
    
    def _extract_question_core(self, instruction: str) -> str:
        """
        从instruction中提取问题核心内容
        
        为什么需要这个函数：
        - instruction包含问题+选项，内容较长
        - 提取核心问题可以提高检索准确性
        
        示例：
        输入: "请回答以下医学问题：\n上消化道出血可单纯表现为...\n选项如下：\nA... B..."
        输出: "上消化道出血可单纯表现为..."
        """
        # 移除"请回答以下医学问题："等前缀
        text = re.sub(r'请回答以下医学问题：\s*', '', instruction)
        
        # 提取问题部分（在"选项如下"之前）
        if '选项如下' in text:
            question = text.split('选项如下')[0].strip()
        elif '选项：' in text:
            question = text.split('选项：')[0].strip()
        else:
            question = text.strip()
        
        # 移除"请给出正确答案"等后缀
        question = re.sub(r'请给出.*', '', question).strip()
        
        return question
    
    def _extract_answer_key(self, output: str) -> str:
        """
        从output中提取关键答案信息
        
        示例：
        输入: "正确答案是：C。\n解析：上消化道出血表现为..."
        输出: "上消化道出血表现为..."
        """
        # 移除"正确答案是：X"
        text = re.sub(r'正确答案[是为][:：]\s*[A-E][\s。，,]*', '', output)
        
        # 提取解析部分
        if '解析' in text:
            text = text.split('解析')[1].strip()
            text = re.sub(r'^[:：]\s*', '', text)
        
        return text.strip()
    
    def load_knowledge_from_jsonl(self, jsonl_path: str, max_samples: Optional[int] = None):
        """
        从医学问答数据集加载知识
        
        参数:
            jsonl_path: D同学提供的.jsonl文件路径
            max_samples: 最多加载多少条（用于测试，None表示全部）
        
        数据格式:
            {
                "instruction": "问题+选项",
                "input": "",
                "output": "答案+解析"
            }
        
        优化策略:
            1. 提取问题核心内容
            2. 提取答案关键解析
            3. 构建"问题-答案"对，便于检索
        """
        print(f"\n开始从 {jsonl_path} 加载知识库...")
        
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"找不到数据文件: {jsonl_path}")
        
        count = 0
        skipped = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # 提取字段
                    instruction = data.get('instruction', '')
                    output = data.get('output', '')
                    
                    if not instruction or not output:
                        skipped += 1
                        continue
                    
                    # 提取问题核心
                    question = self._extract_question_core(instruction)
                    
                    # 提取答案解析
                    answer_explanation = self._extract_answer_key(output)
                    
                    # 构建知识条目
                    # 策略1: 组合文本 - 同时包含问题和答案，检索时更准确
                    combined_text = f"{question} {answer_explanation}"
                    
                    knowledge_item = {
                        'id': count,
                        'text': combined_text,  # 用于向量化和检索的文本
                        'question': question,   # 原始问题
                        'answer': output,       # 完整答案
                        'source': '医学问答数据集',
                        'metadata': {
                            'instruction': instruction,
                            'output': output,
                            'question_core': question,
                            'answer_key': answer_explanation
                        }
                    }
                    
                    self.knowledge_base.append(knowledge_item)
                    count += 1
                    
                    if max_samples and count >= max_samples:
                        break
                        
                except json.JSONDecodeError as e:
                    skipped += 1
                    continue
        
        print(f"✓ 成功加载 {len(self.knowledge_base)} 条知识片段")
        if skipped > 0:
            print(f"  (跳过 {skipped} 条无效数据)")
        
        return len(self.knowledge_base)
    
    def build_index(self, save_path: Optional[str] = None):
        """
        构建FAISS向量索引
        
        参数:
            save_path: 索引保存路径（可选）
        """
        if not self.knowledge_base:
            raise ValueError("知识库为空！请先使用 load_knowledge_from_jsonl() 加载数据")
        
        print("\n开始构建向量索引...")
        
        # 提取所有文本（使用组合后的text字段）
        texts = [item['text'] for item in self.knowledge_base]
        
        # 转换为向量
        print(f"正在将 {len(texts)} 条知识转换为向量...")
        self.embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32  # 批量处理，提高速度
        )
        
        # 构建FAISS索引
        dimension = self.embeddings.shape[1]
        print(f"向量维度: {dimension}")
        
        # 使用IndexFlatIP（内积相似度）
        self.index = faiss.IndexFlatIP(dimension)
        
        # 归一化向量（使内积等价于余弦相似度）
        faiss.normalize_L2(self.embeddings)
        
        # 添加到索引
        self.index.add(self.embeddings)
        
        print(f"✓ 索引构建完成！索引中共有 {self.index.ntotal} 个向量")
        
        # 保存索引
        if save_path:
            self.save_index(save_path)
        
        return True
    
    def save_index(self, save_path: str):
        """保存知识库和索引"""
        print(f"\n保存索引到 {save_path}...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(save_path, 'faiss.index'))
        
        # 保存知识库
        with open(os.path.join(save_path, 'knowledge_base.pkl'), 'wb') as f:
            pickle.dump(self.knowledge_base, f)
        
        # 保存embeddings
        np.save(os.path.join(save_path, 'embeddings.npy'), self.embeddings)

        config = {
            'knowledge_count': len(self.knowledge_base),
            'created_time': datetime.now().isoformat()
        }
        with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        print("✓ 保存完成！")

    def is_index_loaded(self) -> bool:
        """检查索引是否已加载"""
        return self.index is not None and len(self.knowledge_base) > 0

    def load_index(self, load_path: str):
        """从本地加载索引"""
        print(f"\n从 {load_path} 加载索引...")

        required_files = ['faiss.index', 'knowledge_base.pkl', 'embeddings.npy']
        for file in required_files:
            file_path = os.path.join(load_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"索引文件缺失: {file_path}")
        
        # 加载FAISS索引
        self.index = faiss.read_index(os.path.join(load_path, 'faiss.index'))
        
        # 加载知识库
        with open(os.path.join(load_path, 'knowledge_base.pkl'), 'rb') as f:
            self.knowledge_base = pickle.load(f)
        
        # 加载embeddings
        self.embeddings = np.load(os.path.join(load_path, 'embeddings.npy'))
        
        print(f"✓ 加载完成！知识库包含 {len(self.knowledge_base)} 条记录")
    
    def retrieve(self, query: str, top_k: int = 5, return_full_answer: bool = True) -> List[Dict[str, Any]]:
        """
        检索相关知识片段
        
        参数:
            query: 查询问题
            top_k: 返回最相关的K个结果
            return_full_answer: 是否返回完整答案（True）还是只返回核心解析（False）
        
        返回:
            知识片段列表，每个包含：
            - text: 检索用的组合文本
            - question: 原始问题
            - answer: 完整答案（如果return_full_answer=True）
            - answer_key: 核心解析（如果return_full_answer=False）
            - score: 相似度分数
        """
        if not self.is_index_loaded():
            raise ValueError("索引未加载！请先构建索引或加载预构建索引")

        if self.index is None:
            raise ValueError("索引未构建！请先使用 build_index() 或 load_index()")
        
        # 将查询转换为向量
        query_vector = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        
        # 检索
        scores, indices = self.index.search(query_vector, top_k)
        
        # 整理结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                knowledge_item = self.knowledge_base[idx]
                
                result = {
                    'text': knowledge_item['text'],
                    'question': knowledge_item['question'],
                    'score': float(score),
                    'source': knowledge_item['source'],
                    'id': knowledge_item['id']
                }
                
                # 根据参数决定返回完整答案还是核心解析
                if return_full_answer:
                    result['answer'] = knowledge_item['answer']
                else:
                    result['answer_key'] = knowledge_item['metadata']['answer_key']
                
                # 包含完整metadata（供后续使用）
                result['metadata'] = knowledge_item['metadata']
                
                results.append(result)
        
        return results
    
    def rag_retrieve(self, query: str, top_k: int = 5, return_full_answer: bool = True) -> Dict[str, Any]:
        """
        标准化检索接口（供其他模块调用）
        
        参数:
            query: 医学问题
            top_k: 返回的知识片段数量
            return_full_answer: 是否返回完整答案
        
        返回:
            {
                'success': bool,
                'query': str,
                'results': list,
                'retrieved_count': int,
                'timestamp': str
            }
        """
        try:
            results = self.retrieve(query, top_k, return_full_answer)
            
            response = {
                'success': True,
                'query': query,
                'top_k': top_k,
                'retrieved_count': len(results),
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            if not results:
                response['success'] = False
                response['message'] = "未检索到相关知识，请检查查询关键词"
            
            return response
            
        except Exception as e:
            return {
                'success': False,
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def main():
    """
    使用示例 - 基于实际的医学问答数据集
    """
    print("=" * 60)
    print("医学知识检索与RAG模块 - 优化版")
    print("专门针对医学问答数据集优化")
    print("=" * 60)

    # 预构建索引路径
    prebuilt_index_path = "./rag_index_optimized"

    # 检查是否存在预构建索引
    if os.path.exists(prebuilt_index_path):
        print(f"检测到预构建索引: {prebuilt_index_path}")
        # 直接加载预构建索引
        rag = MedicalRAG(index_path=prebuilt_index_path)
    else:
        print("未找到预构建索引，需要重新构建...")
        # 初始化并构建索引
        rag = MedicalRAG()

        # 加载D同学的数据集
        jsonl_path = "val.jsonl"

        if os.path.exists(jsonl_path):
            print(f"\n发现数据集: {jsonl_path}")
            rag.load_knowledge_from_jsonl(jsonl_path)
            # 构建索引
            rag.build_index(save_path='./rag_index_optimized')
        else:
            print(f"\n✗✗ 找不到数据文件: {jsonl_path}")
            print("请确保D同学的数据文件已上传")
            return

    # 测试检索
    print("\n" + "=" * 60)
    print("测试检索功能")
    print("=" * 60)
    
    test_queries = [
        "上消化道出血的表现是什么？",
        "夜间阵发性呼吸困难见于哪些疾病？",
        "嘶哑样咳嗽的病因是什么？",
        "高血压的诊断标准",
        "糖尿病有什么症状"
    ]
    
    for query in test_queries:
        print(f"\n问题: {query}")
        response = rag.rag_retrieve(query, top_k=3, return_full_answer=True)
        
        if response['success']:
            print(f"检索到 {response['retrieved_count']} 条相关知识:")
            for i, result in enumerate(response['results'], 1):
                print(f"\n  [{i}] 相似度: {result['score']:.4f}")
                print(f"      相关问题: {result['question'][:60]}...")
                print(f"      答案: {result['answer'][:100]}...")
        else:
            print(f"检索失败: {response.get('message', response.get('error'))}")
    
    print("\n" + "=" * 60)
    print("测试完成！索引已保存到 ./rag_index_optimized")
    print("=" * 60)


if __name__ == "__main__":
    main()