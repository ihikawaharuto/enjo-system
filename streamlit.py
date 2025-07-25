import streamlit as st
import torch
from transformers import BertJapaneseTokenizer, BertModel
import logging
import numpy as np  #環境問題(Pythonのバージョンを3.11に設定・変更したため解決)
import base64
import os

logging.basicConfig(level=logging.ERROR)

# --- 定数定義 ---
FIRE_GIF_FILENAME = "fire.gif"
SAFE_TXT_FILENAME = "safe.txt"
OUT_TXT_FILENAME = "out.txt"
DIFF_SIM_TXT_FILENAME = "different_sim.txt"

# --- 関数定義 (元のコードから構造を維持) ---

# モデルとトークナイザーの読み込み（アプリ起動時に一度だけ実行される）
@st.cache_resource
def load_model_and_tokenizer():
    """BERTモデルとトークナイザーを読み込み、キャッシュします。"""
    # TODO: モデルのバージョンを定数化することも検討
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    return tokenizer, model

# safe,outファイルを読み込み
def load_file(filepath):
    """テキストとスコアをファイルから読み込みます。"""
    texts, scores = [], []
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    texts.append(parts[0])
                    scores.append(parts[1])
    except FileNotFoundError:
        st.error(f"エラー: データファイル '{filepath}' が見つかりません。")
    return texts, scores

# テキストのベクトル化
def text_vec(text, tokenizer, model):
    """単一のテキストをベクトル化します。"""
    tmp = tokenizer.encode_plus(text, truncation=True, padding=False, return_tensors='pt')
    outputs = model(**tmp)
    return outputs.pooler_output.detach().numpy()[0]

# ファイルのベクトル化
def list_vec(texts_list, scores_list, label, tokenizer, model):
    """テキストのリストをベクトル化します。"""
    vectors, sources = [], []
    for text, score in zip(texts_list, scores_list):
        vectors.append(text_vec(text, tokenizer, model))
        sources.append((text, label, score))
    return vectors, sources

# コサイン類似度を求める
def comp_sim(qvec, tvec):
    """2つのベクトル間のコサイン類似度を計算します。"""
    return np.dot(qvec, tvec) / (np.linalg.norm(qvec) * np.linalg.norm(tvec))

# listの平均値算出
def average_file(filepath):
    """ファイル内の数値の平均を計算します。"""
    number = []
    try:
        with open(filepath, 'r', encoding="utf-8") as file:
            for line in file:
                try:
                    number.append(float(line.strip()))
                except (ValueError, IndexError):
                    continue  # 数値に変換できない行はスキップ
    except FileNotFoundError:
        return 0.75  # ファイルがない場合はデフォルト値
    return float(sum(number) / len(number)) if len(number) != 0 else 0.75

# データをロードしてベクトル化する処理をキャッシュ
@st.cache_data
def load_and_vectorize_data(_tokenizer, _model, safe_path, out_path):
    """safe,outファイルを読み込みベクトル化します。"""
    text_safe, score_safe = load_file(safe_path)
    text_out, score_out = load_file(out_path)

    vec_safe, sources_safe = list_vec(text_safe, score_safe, "safe", _tokenizer, _model)
    vec_out, sources_out = list_vec(text_out, score_out, "out", _tokenizer, _model)

    vec = vec_safe + vec_out
    text_sources = sources_safe + sources_out
    return vec, text_sources

# 動画ファイルをBase64にエンコードする関数（キャッシュあり）
@st.cache_data
def get_file_as_base64(path):
    """ファイルを読み込み、Base64エンコードされた文字列を返す。"""
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"ファイルが見つかりません: {path}")
        return None
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return None
    
# --- Streamlit アプリケーションのUIとロジック ---

st.title("炎上判定システム")

# --- ファイルパスの準備 ---
# スクリプトファイルがあるディレクトリの絶対パスを取得
script_dir = os.path.dirname(os.path.abspath(__file__))
# GIFファイルのパスを構築
gif_path = os.path.join(script_dir, FIRE_GIF_FILENAME)
# データファイルのパスを構築
safe_txt_path = os.path.join(script_dir, SAFE_TXT_FILENAME)
out_txt_path = os.path.join(script_dir, OUT_TXT_FILENAME)
diff_sim_txt_path = os.path.join(script_dir, DIFF_SIM_TXT_FILENAME)

# GIFを事前に読み込んでおく
fire_gif_base64 = get_file_as_base64(gif_path)

# モデルとデータの準備
tokenizer, model = load_model_and_tokenizer()
vec, text_sources = load_and_vectorize_data(tokenizer, model, safe_txt_path, out_txt_path)

# ユーザー入力
text_x_input = st.text_area('判定したいテキストを入力して下さい：')
# session_stateをフォロワー数入力に利用
if 'follower_count' not in st.session_state:
    st.session_state.follower_count = 100  # デフォルト値

def set_follower_count(count):
    """ボタンクリックでフォロワー数を設定するコールバック関数"""
    st.session_state.follower_count = count

# フォロワー数入力UI
st.number_input("フォロワー数：", min_value=0, key='follower_count')

st.write("または、おおよその数を選択:")
follower_options = {
    "1,000": 1000,
    "10,000": 10000,
    "100,000": 100000,
    "1,000,000": 1000000,
}
cols = st.columns(len(follower_options))
for i, (label, count) in enumerate(follower_options.items()):
    cols[i].button(
        label, on_click=set_follower_count, args=(count,), use_container_width=True
    )

# 判定実行ボタンと炎のGIFを横並びに表示
col1, col2 = st.columns([3, 1], vertical_alignment="bottom")

with col1:
    # ボタンのクリック状態を変数で受け取る
    run_button = st.button("判定実行", use_container_width=True)

with col2:
    pass  # col2 の初期表示を空にする

if run_button:  # run_button の外から判定ロジックを移動
        if not text_x_input.strip():
            st.warning("テキストを入力してください。")
        elif not vec:
            st.error("比較用データが見つかりません。safe.txt, out.txt を確認してください。")
        else:
            # text_xを受け取りベクトル化、類似度を測り、判定を出力
            vec_x = text_vec(text_x_input, tokenizer, model)

            similarity_score = []
            for tvec in vec:
                similarities = comp_sim(vec_x, tvec)
                similarity_score.append(similarities)

            most_similar_index = np.argmax(similarity_score)
            most_similar_text, source_file, B = text_sources[most_similar_index]
            most_similar_score = similarity_score[most_similar_index]

            # IRLの計算
            F = st.session_state.follower_count
            if source_file == "safe":
                P = 0
            elif source_file == "out":
                P = 1

            I = int(F * 0.3 + F ** 0.1 * (1 + 210970 * (int(B) / 100) ** 3.2 * (1 + 0.5 * (int(B) / 100) ** 5 * P)))
            R = int(I * 0.01 * (1 + 2 * (int(B) / 100) ** 2) * (1 + P))
            L = int(I * 0.03 * (1 + 0.5 * (int(B) / 100) ** 0.7) * (1 + 0.1 * P))

            # 結果の表示
            st.write("---")
            st.write(f"似ている文：{most_similar_text}、類似度：{most_similar_score:.4f}")
            st.write(f"インプレッション数：{I:,}、リポスト数：{R:,}、いいね数：{L:,}")

            if most_similar_score < average_file(diff_sim_txt_path):  # 卍要検討卍
                st.warning("判定不可")
            else:
                if "safe" in source_file:
                    st.success(f"判定：SAFE、バズスコア：{B}")
                elif "out" in source_file:
                    st.error(f"判定：OUT、バズスコア：{B}")
                    # --- ここから動画再生のロジック ---
                    with col2:  # 判定とGIF表示を col2 内で行う
                        if fire_gif_base64:
                            # 画面中央にGIFを表示するためのHTMLとCSS
                            gif_html = f"""
                                    <div style="text-align: center;">
                                        <img src="data:image/gif;base64,{fire_gif_base64}" alt="炎上GIF" style="max-width: 50%; max-height: 50%; width: auto; height: auto;">
                                    </div>
                            """
                            st.markdown(gif_html, unsafe_allow_html=True)

            # フィードバックのために結果を保存
            st.session_state.last_result = {
                'source_file': source_file,
                'similarity_score': most_similar_score
            }

# フィードバックセクション
if 'last_result' in st.session_state:
    st.write("---")
    check = st.radio('あなたの判定は？(safe/out):', ('safe', 'out'), key='feedback_radio')
    
    if st.button("フィードバックを送信"):
        last_result = st.session_state.last_result
        if check != last_result['source_file']:
            try:
                with open(diff_sim_txt_path, 'a', encoding="utf-8") as file:
                    # スコアの後に改行を追加する方が一般的で安全です
                    file.write(f"{last_result['similarity_score']:.4f}\n")
                st.toast("フィードバックを記録しました。ありがとうございます。")
            except Exception as e:
                st.error(f"フィードバックの保存中にエラーが発生しました: {e}")
        else:
            st.toast("フィードバックありがとうございます！")
