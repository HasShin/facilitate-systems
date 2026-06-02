import pandas as pd
import pickle

# 正しく列番号と名前を対応
df = pd.read_csv(
    "output_belbin_multidim.csv",
    usecols=[1, 2, 4, 6],  # B, C, E, G列
    names=["utterance", "思考系_役割", "対人関係系_役割", "行動系_役割"],
    header=1  # 1行目がヘッダーならこれでOK（0が最初の行）
)

# 対象の3列
role_columns = ["思考系_役割", "対人関係系_役割", "行動系_役割"]

label_maps = {}

for col in role_columns:
    labels = df[col].dropna().unique().tolist()
    labels = sorted([label for label in labels if label != "不明"])  # 不明は除く
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    label_maps[col] = (label2id, id2label)

with open("label_maps.pkl", "wb") as f:
    pickle.dump(label_maps, f)

print("✅ label_maps.pkl を正常に生成しました。")
