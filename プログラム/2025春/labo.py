import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os

import pandas as pd

# 📂 データ読み込み
df1 = pd.read_csv("belbin_converted_roles_labeled_with_unknown.csv")
df2 = pd.read_csv("belbin_converted_roles_test_with_unknown.csv")

# 🧩 データ結合（行方向にマージ）
df_combined = pd.concat([df1, df2], ignore_index=True)

# 💾 結合データの保存（必要なら）
df_combined.to_csv("belbin_combined_dataset.csv", index=False)
print("✅ 2つのCSVを統合しました。")


# 🔧 パス設定
DATA_PATH = "belbin_combined_dataset.csv"
MODEL_DIR = "./fine_tuned_belbin"
os.makedirs(MODEL_DIR, exist_ok=True)

# 🧠 Sentence-BERTの読み込み（事前学習済みモデル）
model = SentenceTransformer("cl-tohoku/bert-base-japanese-v2")


# 🧾 データ読み込み
df = pd.read_csv(DATA_PATH)

# 🏷️ 学習対象の3系統列
role_columns = ["思考系_役割", "対人関係系_役割", "行動系_役割"]

# 🧠 文埋め込みの事前計算
utterances = df["発話"].fillna("").astype(str).tolist()
embeddings = model.encode(utterances, convert_to_numpy=True)

# 🔁 各分類器の学習
label_maps = {}
for col in role_columns:
    print(f"\n🔄 ファインチューニング中: {col}")
    
    labels = df[col].fillna("不明").tolist()
    label2id = {label: i for i, label in enumerate(sorted(set(labels)))}
    id2label = {i: label for label, i in label2id.items()}
    y = np.array([label2id[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

    base_clf = LogisticRegression(max_iter=1000, random_state=42)
    clf = CalibratedClassifierCV(estimator=base_clf, cv=3)
    clf.fit(X_train, y_train)

    # 📊 評価
    y_pred = clf.predict(X_test)
    print(f"📊 {col} の分類結果:")
    print(classification_report(y_test, y_pred, target_names=[id2label[i] for i in sorted(id2label)]))

    # 💾 保存
    with open(f"classifier_{col}.pkl", "wb") as f:
        pickle.dump(clf, f)

    label_maps[col] = (label2id, id2label)

# 🔧 ラベルマップの保存
with open("label_maps.pkl", "wb") as f:
    pickle.dump(label_maps, f)

print("✅ 全分類器とラベルマップを保存しました。")
