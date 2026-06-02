# ============================
# Belbin分類器の訓練スクリプト
# ============================
# 本スクリプトでは、グループ内での発言データに基づき、
# Belbin理論の3系統（思考系・対人関係系・行動系）それぞれに対する分類器を学習します。
# Sentence-BERT によって各発言をベクトル化し、Logistic Regression + Calibrated Classifier により分類器を構築します。
# 学習された分類器は .pkl 形式で保存され、別スクリプト（例: Streamlitアプリ）での推論に利用されます。

import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =======================
# 1. データ読み込み
# =======================
# 'output_belbin_multidim.csv' は、各発言とそれぞれのBelbin系統に対応するラベルを含むCSVです。
# 各列の意味:
# - utterance: 会話発言テキスト
# - 思考系_役割 / 対人関係系_役割 / 行動系_役割: 各系統の役割ラベル（例: プラント, コーディネーターなど）
df = pd.read_csv("output_belbin_multidim.csv", usecols=[1, 2, 4, 6], header=1,
                 names=["utterance", "思考系_役割", "対人関係系_役割", "行動系_役割"])

# =======================
# 2. Sentence-BERTモデル読み込み
# =======================
# 事前に fine-tune 済みの Sentence-BERT モデルを使用して発言をベクトル化します。
# モデルディレクトリには config.json や pytorch_model.bin などが含まれています。
model = SentenceTransformer("./fine_tuned_belbin")

# =======================
# 3. 役割ごとの分類器の訓練
# =======================
# 3系統（思考系 / 対人関係系 / 行動系）それぞれについて分類器を構築・学習し、
# ロールのラベルマップも一緒に辞書に格納して後で保存します。
output_dir = "./"
label_maps = {}  # 後ほど全体を label_maps.pkl に保存

# 系統ごとに繰り返し処理
for col in ["思考系_役割", "対人関係系_役割", "行動系_役割"]:
    print(f"🔄 ファインチューニング中: {col}")

    # 該当系統のデータのみ抽出
    df_role = df[[col, "utterance"]].dropna()
    labels = df_role[col].tolist()
    sentences = df_role["utterance"].tolist()

    # =======================
    # ラベルエンコーディング
    # =======================
    # 文字ラベル（例: プラント）を数値IDに変換
    label2id = {label: i for i, label in enumerate(sorted(set(labels)))}
    id2label = {i: label for label, i in label2id.items()}
    y = [label2id[label] for label in labels]

    # =======================
    # 文ベクトル化
    # =======================
    # 各発言をSentence-BERTでベクトル化（高次元な意味特徴ベクトルを生成）
    X = model.encode(sentences, convert_to_numpy=True)

    # =======================
    # 分類器の学習
    # =======================
    # ベースモデルにロジスティック回帰を用い、分類確率の較正（CalibratedClassifier）を実施
    base_clf = LogisticRegression(max_iter=1000)
    clf = CalibratedClassifierCV(estimator=base_clf, cv=3)

    clf.fit(X, y)

    # =======================
    # 分類器の保存
    # =======================
    with open(f"{output_dir}/classifier_{col}.pkl", "wb") as f:
        pickle.dump(clf, f)

    # ラベルマップを保存用辞書に追加
    label_maps[col] = (label2id, id2label)

    # =======================
    # 評価（分類精度の確認）
    # =======================
    y_pred = clf.predict(X)
    print(f"📊 {col} の分類結果:")
    print(classification_report(y, y_pred, target_names=list(label2id.keys())))

# =======================
# 4. ラベルマップの保存
# =======================
# 推論時に使用するため、label2id / id2label マップを保存しておく。
# これは Streamlit アプリ側で読み込まれ、分類結果を人間可読に表示するのに使われます。
with open(f"{output_dir}/label_maps.pkl", "wb") as f:
    pickle.dump(label_maps, f)

print("✅ 全分類器とラベルマップを保存しました。")

# =======================
# 🔁 関連スクリプトとの関係（補足）
# =======================
# - generate_label_maps.py: ラベルマップの単体生成に特化したスクリプト（このコードでも統合している）
# - application.py（非提出）: 本分類器を用いてリアルタイムに発話を分類するStreamlitアプリ
# - belbin_combined_dataset.csv 等: 元の学習用データファイル
