import sys
import os

# --- 1. 路径与配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import config
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import torch
import joblib

# 定义模型保存路径 (保存在根目录)
VERIFIER_MODEL_PATH = os.path.join(config.ROOT_DIR, "verifier_model.pkl")

# 1️⃣ 加载数据
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到数据文件: {path}")
    df = pd.read_excel(path)
    print(f"✅ [Verifier] Loaded {len(df)} samples from {path}")
    return df

# 2️⃣ 计算语义特征
def compute_semantic_features(df, model_name=config.EMBEDDING_MODEL_PATH):
    print(f"⚙️ [Verifier] 计算语义特征 (使用模型: {model_name})...")
    model = SentenceTransformer(model_name)

    sims_gt_ma, sims_q_ma, sims_q_gt = [], [], []
    for _, row in df.iterrows():
        gt, ma, q = str(row["GroundTruth"]), str(row["ModelAnswer"]), str(row["Question"])
        
        emb_gt = model.encode(gt, convert_to_tensor=True)
        emb_ma = model.encode(ma, convert_to_tensor=True)
        emb_q = model.encode(q, convert_to_tensor=True)

        sims_gt_ma.append(float(util.cos_sim(emb_gt, emb_ma)))
        sims_q_ma.append(float(util.cos_sim(emb_q, emb_ma)))
        sims_q_gt.append(float(util.cos_sim(emb_q, emb_gt)))

    df["sim_gt_ma"] = sims_gt_ma
    df["sim_q_ma"] = sims_q_ma
    df["sim_q_gt"] = sims_q_gt
    df["len_diff"] = df["ModelAnswer"].apply(lambda x: len(str(x))) - df["GroundTruth"].apply(lambda x: len(str(x)))
    print("✅ [Verifier] 语义特征计算完成。")
    return df

# 3️⃣ 中文 Roberta NLI 特征
def compute_nli_features(df, model_name=config.NLI_MODEL_PATH, max_samples=None):
    print(f"⚙️ [Verifier] 计算 NLI 逻辑特征 (使用模型: {model_name})...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ NLI 模型加载失败: {e}")
        print("请检查 config.py 中的 NLI_MODEL_PATH 是否正确，或运行 scripts/download_models.py")
        raise e
        
    model.eval()

    entail_probs, contra_probs = [], []

    for i, row in df.iterrows():
        if max_samples and i >= max_samples:
            entail_probs.append(0.0)
            contra_probs.append(0.0)
            continue

        premise = str(row["GroundTruth"])
        hypothesis = str(row["ModelAnswer"])
        
        inputs = tokenizer(premise, hypothesis, return_tensors='pt',
                           truncation=True, max_length=512, padding='max_length')
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0].numpy()

        # Roberta-NLI 的类别顺序一般为 [entailment, neutral, contradiction]
        # 注意：不同模型的输出顺序可能不同，这里假设符合该顺序
        entail_probs.append(float(probs[0]))
        contra_probs.append(float(probs[2]))

        if i % 20 == 0:
            print(f"  Processed {i+1}/{len(df)} samples")

    df["nli_entail"] = entail_probs
    df["nli_contra"] = contra_probs
    print("✅ [Verifier] NLI 特征计算完成。")
    return df

# 4️⃣ 训练分类器
def train_verifier(df):
    features = ["sim_gt_ma", "sim_q_ma", "sim_q_gt", "len_diff", "nli_entail", "nli_contra"]
    
    # 简单的空值处理
    df = df.dropna(subset=features + ["HallucinationLabel"])
    
    X = df[features].values
    y = df["HallucinationLabel"].astype(int)

    print("⚙️ [Verifier] 正在训练逻辑回归分类器...")
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    y_score = clf.predict_proba(X)[:, 1]
    
    try:
        auc = roc_auc_score(y, y_score)
        print(f"✅ AUC: {auc:.3f}")
    except:
        print("⚠️ 样本太少或单一，无法计算 AUC")
        
    print(classification_report(y, y_pred, digits=3))

    joblib.dump(clf, VERIFIER_MODEL_PATH)
    print(f"✅ [Verifier] 模型已保存至: {VERIFIER_MODEL_PATH}")
    return clf

# 5️⃣ 核心验证类
class HallucinationVerifier:
    def __init__(self,
                 embed_model=config.EMBEDDING_MODEL_PATH,
                 nli_model=config.NLI_MODEL_PATH):
        
        print("⚙️ [Verifier] 初始化验证器...")
        self.embedder = SentenceTransformer(embed_model)
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        
        if os.path.exists(VERIFIER_MODEL_PATH):
            self.classifier = joblib.load(VERIFIER_MODEL_PATH)
            print("✅ [Verifier] 已加载预训练的分类器。")
        else:
            print("⚠️ [Verifier] 未找到预训练分类器 (verifier_model.pkl)。请先运行 train_verifier()。")
            self.classifier = None

    def verify(self, question, model_answer, groundtruth):
        """输入问题、AI回答、正确答案 → 输出事实一致性分数"""
        if self.classifier is None:
            return {"fact_consistency_score": 0.0, "label": "unknown (no model)", "notes": []}

        # 计算语义特征
        emb_gt = self.embedder.encode(str(groundtruth), convert_to_tensor=True)
        emb_ma = self.embedder.encode(str(model_answer), convert_to_tensor=True)
        emb_q = self.embedder.encode(str(question), convert_to_tensor=True)

        sim_gt_ma = float(util.cos_sim(emb_gt, emb_ma))
        sim_q_ma = float(util.cos_sim(emb_q, emb_ma))
        sim_q_gt = float(util.cos_sim(emb_q, emb_gt))
        len_diff = len(str(model_answer)) - len(str(groundtruth))

        # 计算 NLI 特征
        inputs = self.tokenizer(str(groundtruth), str(model_answer), return_tensors='pt',
                                truncation=True, max_length=512, padding='max_length')
        with torch.no_grad():
            probs = torch.softmax(self.nli_model(**inputs).logits, dim=1)[0].numpy()
        nli_entail, nli_contra = float(probs[0]), float(probs[2])

        # 预测
        X = np.array([[sim_gt_ma, sim_q_ma, sim_q_gt, len_diff, nli_entail, nli_contra]])
        score = self.classifier.predict_proba(X)[0, 1]
        
        # 定义阈值，分数越高越consistent (非幻觉)
        label = "consistent" if score >= 0.5 else "hallucination"

        return {
            "fact_consistency_score": round(score, 3),
            "label": label,
            "notes": [
                f"sim(gt,ma)={sim_gt_ma:.2f}",
                f"nli_entail={nli_entail:.2f}",
                f"nli_contra={nli_contra:.2f}"
            ]
        }

def load_or_process_data(data_path, processed_path):
    """加载或重新处理数据"""
    if os.path.exists(processed_path):
        print(f"📁 [Verifier] 发现已处理数据: {processed_path}")
        return pd.read_excel(processed_path)
    else:
        print("🔄 [Verifier] 未找到缓存，开始处理原始数据...")
        df = load_data(data_path)
        df = compute_semantic_features(df)
        df = compute_nli_features(df, max_samples=None)
        df.to_excel(processed_path, index=False)
        print(f"💾 [Verifier] 处理完成并保存至: {processed_path}")
        return df

# 6️⃣ 单元测试
if __name__ == "__main__":
    # 使用 config 中的路径进行测试
    data_path = config.HALLUCINATION_DATA_PATH
    processed_path = config.HALLUCINATION_PROCESSED_PATH

    print("\n--- Verifier 单元测试 ---")
    
    if os.path.exists(data_path):
        # 1. 准备数据
        df = load_or_process_data(data_path, processed_path)
        
        # 2. 训练模型
        train_verifier(df)

        # 3. 验证单条
        verifier = HallucinationVerifier()
        result = verifier.verify(
            "糖尿病患者适合吃什么主食？",
            "糖尿病患者应多吃糯米和红薯。",
            "糖尿病患者应避免糯米等高升糖食物。"
        )
        print("\n✅ 测试结果：", result)
    else:
        print(f"❌ 找不到数据文件 {data_path}，无法进行测试。")
