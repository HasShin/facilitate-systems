from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import re
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_TFIDF = True
except ImportError:
    HAS_TFIDF = False

model = SentenceTransformer('cl-tohoku/bert-base-japanese-v2')

processed_sentences = set()
processed_embeddings = []
processed_texts = []
utterance_count = defaultdict(int)
char_count = defaultdict(int)
proposal_count = defaultdict(int)
coord_count = defaultdict(int)
first_speaker = None
idea_first_speaker = None


coordination_keywords = [
    "まとめ", "確認", "調整", "なるほど", "じゃあ", "どうですか", "どう思",
    "ありますか", "必要", "注目", "みなさん", "皆さん", "みんな","?","？"
]

threshold = 0.5
initial_penalty = 0.0
decay_rate = 0.05      
overlap_threshold = 0.55
plant_first_bonus = 0.0
monitor_overlap_bonus = 0.15
specialist_bonus = 0.34
specialist_length_scale = 200
idea_beta_threshold = 0.6
min_utterance_length = 10
coordination_boost = 1.1
tfidf_specialist_threshold = 0.7
plant_tfidf_bonus = 0.3
specialist_tfidf_bonus = 0.12
plant_first_person_bonus = 0.35
first_person_keywords = ["私", "僕", "俺", "自分", "わたし", "ぼく", "おれ", "個人的には"]
monitor_requires_beta = True
monitor_beta_floor = 0.2
plant_overlap_penalty = 0.15
callout_keywords = ["皆さん", "みなさん", "みんな"]
callout_bonus = 0.5
date_keywords = ["日付", "日", "月", "月日", "締切", "期限", "週", "来週", "今週", "今月", "来月", "日曜", "月曜", "火曜", "水曜", "木曜", "金曜", "土曜"]
completer_date_bonus = 1.1
action_first_person_bonus = 0.4
action_first_person_schedule_bonus = 0.5
completer_date_only_bonus = 0.4
shaper_callout_bonus = 0.55
shaper_callout_keywords = ["さん", "くん", "ちゃん", "氏", "君"]
coordinator_other_bonus = 0.2
coordinator_other_keywords = ["さん", "くん", "ちゃん", "氏", "君", "あなた", "皆さん", "みなさん", "みんな", "他の人"]
idea_keywords = [
    "案", "提案", "アイデア", "こうしたら", "こうすると", "すれば", "すると",
    "必要", "改善", "工夫", "方針", "方策", "解決"
]


def compute_alpha(speaker, utterance):
    utterance_count[speaker] += 1
    char_count[speaker] += len(utterance)
    total_count = sum(utterance_count.values())
    total_chars = sum(char_count.values())
    char_ratio = char_count[speaker] / total_chars if total_chars > 0 else 0
    return char_ratio

# ==============================
# β：新規提案率（逆減衰＋初期抑制）
# ==============================
def _tfidf_max_similarity(new_text):
    if not HAS_TFIDF or not processed_texts:
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts + [new_text])
    new_vec = tfidf_matrix[-1]
    prev_vecs = tfidf_matrix[:-1]
    sims = (prev_vecs @ new_vec.T).toarray().ravel()
    return float(sims.max()) if sims.size else 0.0

def _tfidf_peak_score(new_text):
    if not HAS_TFIDF:
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts + [new_text])
    new_vec = tfidf_matrix[-1].toarray().ravel()
    return float(new_vec.max()) if new_vec.size else 0.0

def _max_similarity_to_previous(new_emb, new_text):
    if not processed_embeddings:
        return 0.0
    cos_sims = [np.dot(new_emb, emb)/(np.linalg.norm(new_emb)*np.linalg.norm(emb)+1e-8)
                for emb in processed_embeddings]
    bert_max = max(cos_sims)
    if HAS_TFIDF:
        tfidf_max = _tfidf_max_similarity(new_text)
        return max(bert_max, tfidf_max)
    return bert_max

def _technical_term_score(text):
    matches = re.findall(r"[A-Za-z0-9]{2,}|[ァ-ヶー]{3,}", text)
    return min(1.0, len(matches) / 3)

def _specialist_score(text):
    tech_score = _technical_term_score(text)
    length_score = min(1.0, len(text) / specialist_length_scale)
    return (tech_score + length_score) / 2

def _is_idea_utterance(text, beta):
    if beta >= idea_beta_threshold:
        return True
    return any(k in text for k in idea_keywords)

def _has_first_person(text):
    return any(k in text for k in first_person_keywords)

def is_new_utterance(new_emb, new_text):
    if not processed_embeddings:
        return True
    cos_sims = [np.dot(new_emb, emb)/(np.linalg.norm(new_emb)*np.linalg.norm(emb)+1e-8)
                for emb in processed_embeddings]
    bert_max = max(cos_sims)
    if HAS_TFIDF:
        tfidf_max = _tfidf_max_similarity(new_text)
        combined_sim = (bert_max + tfidf_max) / 2
        return combined_sim < threshold
    return bert_max < threshold

def compute_beta(speaker, utterance, emb=None):
    if emb is None:
        emb = model.encode(utterance)
    new_flag = is_new_utterance(emb, utterance)

    # 過去発話に追加
    processed_sentences.add(utterance)
    processed_embeddings.append(emb)
    processed_texts.append(utterance)

    # 新規発言ならカウント
    if new_flag:
        proposal_count[speaker] += 1

    # 通常のβ（新規提案率）を計算
    total = utterance_count[speaker] if utterance_count[speaker] > 0 else 1
    beta = proposal_count[speaker] / total

    # 初期抑制 + 発言回数に応じて徐々に補正（逆減衰）
    beta_adjusted = max(0, beta - initial_penalty + decay_rate * (utterance_count[speaker]-1))

    return beta_adjusted

# ==============================
# γ：調整発言比率
# ==============================
def coordination_weight(utterance):
    if not any(k in utterance for k in coordination_keywords):
        return 0.0
    if any(k in utterance for k in callout_keywords):
        return 1.0 + callout_bonus
    if "?" in utterance or "？" in utterance:
        return 0.5
    return 1.0

def compute_gamma(speaker, utterance):
    weight = coordination_weight(utterance)
    if weight > 0:
        coord_count[speaker] += weight + coordination_boost * weight
    total = utterance_count[speaker] if utterance_count[speaker] > 0 else 1
    return coord_count[speaker] / total

# ==============================
# ロール判定（αβγ加重・最大値）
# ==============================
def _mark_first_speaker(speaker):
    global first_speaker
    if first_speaker is None:
        first_speaker = speaker

def _mark_idea_first_speaker(speaker, utterance, beta):
    global idea_first_speaker
    if idea_first_speaker is None and _is_idea_utterance(utterance, beta):
        idea_first_speaker = speaker

def _is_idea_first_speaker(speaker):
    return speaker == idea_first_speaker

def _think_role_bonus(speaker, utterance, overlap_sim, tfidf_peak):
    bonuses = {"プラント": 0.0, "モニター評価者": 0.0, "スペシャリスト": 0.0}
    specialist_score = _specialist_score(utterance)
    if _is_idea_first_speaker(speaker):
        bonuses["プラント"] += plant_first_bonus
    if _has_first_person(utterance):
        bonuses["プラント"] += plant_first_person_bonus
    if overlap_sim >= overlap_threshold:
        monitor_bonus = monitor_overlap_bonus
        if specialist_score >= 0.6:
            monitor_bonus *= 0.5
        bonuses["モニター評価者"] += monitor_bonus
        bonuses["プラント"] -= plant_overlap_penalty
    bonuses["スペシャリスト"] += specialist_bonus * specialist_score
    if HAS_TFIDF:
        if tfidf_peak >= tfidf_specialist_threshold:
            bonuses["スペシャリスト"] += specialist_tfidf_bonus
        else:
            bonuses["プラント"] += plant_tfidf_bonus
    return bonuses

def _has_date_reference(text):
    return any(k in text for k in date_keywords)

def _has_callout(text):
    return any(k in text for k in shaper_callout_keywords)

def _has_other_reference(text):
    return any(k in text for k in coordinator_other_keywords)

def predict_roles_by_scores(alpha, beta, gamma, speaker, utterance, overlap_sim, tfidf_peak):
    bonuses = _think_role_bonus(speaker, utterance, overlap_sim, tfidf_peak)
    # 思考系
    think_scores = {
        "プラント": beta + bonuses["プラント"],
        "モニター評価者": gamma + bonuses["モニター評価者"],
        "スペシャリスト": alpha + bonuses["スペシャリスト"]
    }
    if monitor_requires_beta and beta < monitor_beta_floor:
        think_scores["モニター評価者"] *= 0.5
    think_role = max(think_scores, key=think_scores.get)

    # 対人関係系
    interpersonal_scores = {
        "チームワーカー": gamma,
        "コーディネーター": alpha,
        "リソース探究者": beta
    }
    if _has_other_reference(utterance):
        interpersonal_scores["コーディネーター"] += coordinator_other_bonus
    interpersonal_role = max(interpersonal_scores, key=interpersonal_scores.get)

    # 行動系
    action_scores = {
        "完遂者": alpha,
        "実行者": beta * 0.6,
        "シェイパー": gamma
    }
    has_date = _has_date_reference(utterance)
    has_first_person = _has_first_person(utterance)
    if has_date:
        action_scores["完遂者"] += completer_date_bonus
        if not has_first_person:
            action_scores["完遂者"] += completer_date_only_bonus
    if has_first_person:
        action_scores["実行者"] += action_first_person_bonus
    if has_date and has_first_person:
        action_scores["実行者"] += action_first_person_schedule_bonus
    if _has_callout(utterance):
        action_scores["シェイパー"] += shaper_callout_bonus
    action_role = max(action_scores, key=action_scores.get)

    return {
        "思考系_役割": think_role,
        "対人関係系_役割": interpersonal_role,
        "行動系_役割": action_role
    }

# ==============================
# 総合スコア＋ロール判定
# ==============================
def analyze_utterance(speaker, utterance, alpha_weight=0.4, beta_weight=0.3, gamma_weight=0.3):
    if len(utterance) < min_utterance_length:
        return {
            "total_score": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
            "roles": {
                "思考系_役割": "未判定",
                "対人関係系_役割": "未判定",
                "行動系_役割": "未判定"
            }
        }
    emb = model.encode(utterance)
    overlap_sim = _max_similarity_to_previous(emb, utterance)
    tfidf_peak = _tfidf_peak_score(utterance)
    alpha = compute_alpha(speaker, utterance)
    beta = compute_beta(speaker, utterance, emb=emb)
    gamma = compute_gamma(speaker, utterance)
    _mark_first_speaker(speaker)
    _mark_idea_first_speaker(speaker, utterance, beta)
    total_score = alpha_weight * alpha + beta_weight * beta + gamma_weight * gamma
    roles = predict_roles_by_scores(alpha, beta, gamma, speaker, utterance, overlap_sim, tfidf_peak)
    return {
        "total_score": total_score,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "roles": roles
    }
