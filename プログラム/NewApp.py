import streamlit as st
import re
from mecanism import analyze_utterance  # 先ほど作ったmecanism.py

# Streamlit設定
st.set_page_config(page_title="Belbinロール分類（αβγ版）", layout="wide")
st.title("Belbinチームロール分類アプリ（αβγスコアベース）")

# セッションステート初期化
if "processed_sentences" not in st.session_state:
    st.session_state.processed_sentences = set()

# 入力フォーム
with st.form(key="input_form"):
    user_input = st.text_area(
        "💬 発言を入力（形式：発言者: 発言内容）。複数行可", height=200
    )
    submit = st.form_submit_button("🔍 分析")

if submit and user_input.strip():
    lines = user_input.strip().split("\n")
    parsed = []
    for line in lines:
        match = re.match(r"(.+?)[:：]\s*(.+)", line)
        if match:
            speaker, utterance = match.groups()
            parsed.append((speaker.strip(), utterance.strip()))
        else:
            st.warning(f"形式エラー：{line}")

    for idx, (speaker, utterance) in enumerate(parsed):
        if utterance in st.session_state.processed_sentences:
            st.info(f"🔁 {speaker} の発言は既に処理済みです")
            continue

        st.session_state.processed_sentences.add(utterance)

        # 発話の分析（αβγ + 総合スコア + ロール判定）
        result = analyze_utterance(speaker, utterance)

        # 発言表示
        st.markdown(f"**👤 {speaker}**：{utterance}")

        # ロール判定は常に表示
        roles = result["roles"]
        st.markdown(
            f"ロール判定: 思考系={roles['思考系_役割']}, "
            f"対人関係系={roles['対人関係系_役割']}, "
            f"行動系={roles['行動系_役割']}"
        )

        # αβγスコアは expander で表示
        with st.expander("スコアを表示"):
            st.markdown(
                f"総合スコア: {result['total_score']:.2f} "
                f"(α={result['alpha']:.2f}, β={result['beta']:.2f}, γ={result['gamma']:.2f})"
            )

# 🔄 リセットボタン
if st.button("🔄 発話履歴をリセット"):
    st.session_state.processed_sentences = set()
    st.success("発話履歴をリセットしました")
