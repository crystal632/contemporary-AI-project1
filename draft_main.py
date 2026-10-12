import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("--- 1. 正在加载数据 ---")
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data_unlabeled.csv')

X_train_full = train_df['text'].astype(str).tolist()
y_train_full = train_df['target'].values
X_test_unlabeled = test_df['text'].astype(str).tolist()

# --- 2. 划分验证集 (80% 训练, 20% 验证) ---
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# --- 3. 特征提取 (TF-IDF) ---
print("--- 3. 正在进行 TF-IDF 特征提取 ---")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# --- 4. 模型一：逻辑回归 ---
print("--- 4. 训练模型一：逻辑回归 ---")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_val_tfidf)
lr_acc = accuracy_score(y_val, lr_preds)
print(f"逻辑回归验证集准确率: {lr_acc:.4f}")

# --- 5. 模型二：支持向量机 (SVM) ---
print("--- 5. 训练模型二：支持向量机 (SVM) ---")
svm_model = SVC(C=1.0, random_state=42)
svm_model.fit(X_train_tfidf, y_train)
svm_preds = svm_model.predict(X_val_tfidf)
svm_acc = accuracy_score(y_val, svm_preds)
print(f"SVM 验证集准确率: {svm_acc:.4f}")

# --- 6. 使用全部训练数据重新训练并预测测试集 ---
print("--- 6. 生成最终测试集预测结果 ---")
best_vectorizer = TfidfVectorizer(max_features=5000)
X_full_tfidf = best_vectorizer.fit_transform(X_train_full)

# 根据上面准确率选择更好的模型（通常SVM或逻辑回归效果都不错）
final_model = SVC(C=1.0, random_state=42)
final_model.fit(X_full_tfidf, y_train_full)

predictions = final_model.predict(best_vectorizer.transform(X_test_unlabeled))
pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)
print("恭喜！预测结果已成功保存到 predictions.csv 中！")