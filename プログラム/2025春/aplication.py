import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re

st.set_page_config(page_title="Belbinロール分類", layout="wide")
st.title("Belbinチームロール分類アプリ（不明なし / 部分分割判定）")

# 一時的に学習済みモデルによる推定を無効化
ENABLE_MODEL_INFERENCE = False

@st.cache_resource
def load_components():
    model = SentenceTransformer('./fine_tuned_belbin')
    with open("label_maps.pkl", "rb") as f:
        label_maps = pickle.load(f)
    classifiers = {}
    for col in ["思考系_役割", "対人関係系_役割", "行動系_役割"]:
        with open(f"classifier_{col}.pkl", "rb") as f:
            classifiers[col] = pickle.load(f)
    return model, classifiers, label_maps

if ENABLE_MODEL_INFERENCE:
    model, classifiers, label_maps = load_components()

if "processed_sentences" not in st.session_state:
    st.session_state.processed_sentences = set()
if "user_scores" not in st.session_state:
    st.session_state.user_scores = defaultdict(lambda: defaultdict(float))

def split_sentence(text, max_len=30):
    # 句点または読点、スペースで分割（細かい制御も可能）
    parts = re.split(r'[。、「」、]', text)
    parts = [p.strip() for p in parts if p.strip()]
    # 長い文はさらに分割
    result = []
    for p in parts:
        while len(p) > max_len:
            result.append(p[:max_len])
            p = p[max_len:]
        if p:
            result.append(p)
    return result

def predict_and_update_scores(sentences_with_speakers, alert_threshold=2.5):
    if not ENABLE_MODEL_INFERENCE:
        st.info("現在、学習済みモデルによる推定は停止中です。")
        return
    new_sentences = [(spk, sent) for spk, sent in sentences_with_speakers if sent not in st.session_state.processed_sentences]
    if not new_sentences:
        st.info("🔁 新しい未処理の発話はありません。")
        return

    for speaker, sent in new_sentences:
        st.markdown(f"**👤 {speaker}**：{sent}")
        st.session_state.processed_sentences.add(sent)

        # 部分的に分割
        partials = split_sentence(sent)
        if not partials:
            continue

        partial_embeddings = model.encode(partials, convert_to_numpy=True)

        for col in ["思考系_役割", "対人関係系_役割", "行動系_役割"]:
            clf = classifiers[col]
            label2id, id2label = label_maps[col]

            # 各部分の予測確率を平均化
            probs_list = [clf.predict_proba([emb])[0] for emb in partial_embeddings]
            mean_probs = np.mean(probs_list, axis=0)

            # 不明は無視（ただし不明がない場合も考慮）
            valid_ids = [i for i in range(len(mean_probs)) if id2label[i] != "不明"]
            valid_probs = {i: mean_probs[i] for i in valid_ids}

            # 最大スコアのラベルを出力
            max_id = max(valid_probs, key=valid_probs.get)
            pred_label = id2label[max_id]
            max_score = valid_probs[max_id]

            st.markdown(f"　➡ **{col}**: {pred_label}（スコア: {max_score:.2f}）")
            st.session_state.user_scores[speaker][pred_label] += max_score

            if st.session_state.user_scores[speaker][pred_label] >= alert_threshold:
                st.error(f"⚠️ {speaker} の {pred_label} スコアが閾値を超えました！リセットします。")
                st.session_state.user_scores[speaker][pred_label] = 0.0

# 📥 入力フォーム
with st.form(key="input_form"):
    user_input = st.text_area("💬 発言を入力（形式：発言者: 発言内容）。複数行可", height=200)
    submit = st.form_submit_button("🔍 分析")

if submit and user_input.strip():
    lines = user_input.strip().split('\n')
    parsed = []
    for line in lines:
        match = re.match(r"(.+?)[:：]\s*(.+)", line)
        if match:
            speaker, utterance = match.groups()
            parsed.append((speaker.strip(), utterance.strip()))
        else:
            st.warning(f"形式エラー：{line}")
    if parsed:
        predict_and_update_scores(parsed)

# 🔄 リセットボタン
if st.button("🔄 スコアをリセット"):
    st.session_state.processed_sentences = set()
    st.session_state.user_scores = defaultdict(lambda: defaultdict(float))
    st.success("リセットが完了しました。")
