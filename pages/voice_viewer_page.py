import streamlit as st
import pandas as pd
import os
from pathlib import Path

is_streamlit_cloud =True

# ページ全体のレイアウトを広げる設定
st.set_page_config(layout="wide")


csv_file_path = Path(__file__).parent / "../data/processed/features_VOICEACTRESS100_001.csv"

# CSVデータを読み込む
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    st.error("ファイルが見つかりません: ../data/processed/features_VOICEACTRESS100_001.csv")
# タイトルを表示
st.title("JVS 音声再生アプリ")

# サイドバーで性別を選択
gender = st.sidebar.radio("性別を選択してください", ("男性 (Male)", "女性 (Female)"))

# 性別に応じてデータをフィルタリング
if gender == "男性 (Male)":
    filtered_df = df[df["Male_or_Female"] == "M"]
else:
    filtered_df = df[df["Male_or_Female"] == "F"]

# サイドバーでソートする特徴量を選択 
sort_column = st.sidebar.selectbox(
    "ソートする特徴量を選択してください",
    options=filtered_df.columns[4:],  # 特徴量列を選択肢として表示
    index=0
)

# ソート順を選択
sort_order = st.sidebar.radio("ソート順を選択してください", ("降順 (大きい順)", "昇順 (小さい順)"))
ascending = sort_order == "昇順 (小さい順)"

# データをソート
sorted_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

# 再生ボタンをリストで表示する関数
def display_audio_list(dataframe):
    """
    再生ボタンをリスト形式で表示する関数。
    """
    for _, row in dataframe.iterrows():
        st.write(f"**{row['jvs_id']}**")

        if is_streamlit_cloud:
            # Streamlit Cloud上でのファイルパス
            ori_path = Path(row['filepath'])
            relative_path = ori_path.relative_to("../data/raw")  # jvs_ver1/jvs017/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav
            audio_path = f"https://raw.githubusercontent.com/libra2924/QA4U3_G46_app/main/data/raw_extracted/" + str(relative_path)
            print(audio_path)
        else:
            # ローカル環境でのファイルパス
            audio_path = Path(__file__).parent / row["filepath"]
        
        st.audio(audio_path, format="audio/wav")

# 特徴量の表を作成する関数
def create_styled_table(dataframe):
    """
    特徴量の表を作成し、小数点第2位まで表示し、色分けを適用する関数。
    """
    # 小数点第2位まで表示
    dataframe = dataframe.copy()
    dataframe = dataframe.style.format(precision=3).background_gradient(
        axis=0, cmap="RdYlGn", subset=dataframe.columns[4:]  # 特徴量列を色分け
    )
    return dataframe

# レイアウトを2列に分割
col1, col2 = st.columns([1, 7])

# 左側に再生ボタンを表示
with col1:
    st.subheader("再生ボタン")
    display_audio_list(sorted_df)

# 右側に特徴量の表を表示
with col2:
    st.subheader("特徴量一覧")
    styled_table = create_styled_table(sorted_df.drop(columns=["filepath"]))  # filepath列は削除
    st.dataframe(styled_table, use_container_width=True, height=1000)  # 高さを指定して表示