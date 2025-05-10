import streamlit as st
import pandas as pd
import numpy as np
import openjij as oj
import json
from datetime import datetime
from backend.qbooster_qubo import build_qboost_qubo  # 自作関数を読み込む
from backend.recomend import recomend_voice  # 自作関数を読み込む

# --- セッション初期化 ---
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "gender" not in st.session_state:
    st.session_state.gender = ""
if "recomend_ids" not in st.session_state:
    st.session_state.recomend_ids = []
if "recomend2_ids" not in st.session_state:
    st.session_state.recomend2_ids = []
if "selected_names" not in st.session_state:
    st.session_state.selected_names = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}
if "feedback2" not in st.session_state:
    st.session_state.feedback2 = {}
if "step" not in st.session_state:
    st.session_state.step = 1

# --- ユーザー入力 ---
st.title("音声推薦システム")

with st.form("user_info"):
    st.session_state.user_id = st.text_input("ユーザーIDを入力", value=st.session_state.user_id)
    st.session_state.gender = st.radio("推薦する声の性別を選択", ("男性 (Male)", "女性 (Female)"), index=0)
    submitted = st.form_submit_button("送信")

user_id = st.session_state.user_id
# --- データ読み込み ---
df = pd.read_csv("data/processed/samples.csv")
H_df = pd.read_csv("data/processed/H_matrix_with_descriptions.csv")
samples_df = df.merge(H_df, on="sample_id", how="left")

# 入力完了時のみ推薦表示
if st.session_state.step == 1 and user_id:
    st.markdown("---")
    st.header("Step 1: フィードバック")
    st.write("音声を試聴して、フィードバックをお願いします。")
    st.write("できるだけ2つ以上「いいね」をしてください")

    
    gender_key = "M" if "男性" in st.session_state.gender else "F"
    samples_df = samples_df[samples_df["Male_or_Female"] == gender_key]

    # ランダムなサンプルを1回だけ選ぶ（セッション中は保持）
    if "sample_ids" not in st.session_state:
        st.session_state.sample_ids = samples_df.sample(n=10)["sample_id"].tolist()


    for sid in st.session_state.sample_ids:
        row = samples_df[samples_df["sample_id"] == sid].iloc[0]
        # st.audio(row["filepath"])
        # st.session_state.feedback[sid] = st.radio(
        #     f"{sid} の評価：",
        #     ("好き", "普通", "苦手"),
        #     key=f"feedback_{sid}",
        #     index=1  # 「普通」が2番目なので index=1
        # )
        # 未評価の初期値を「普通」に設定
        if sid not in st.session_state.feedback:
            st.session_state.feedback[sid] = "普通"

        with st.container():
            cols = st.columns([2, 1])
            with cols[0]:
                st.audio(row["filepath"])
            with cols[1]:
                liked = st.button(f"👍 いいね！", key=f"feedback_{sid}")
                if liked:
                    st.session_state.feedback[sid] = "好き"
                st.write("評価:", st.session_state.feedback[sid])

            st.markdown("---")


    # --- フィードバック送信 ---
    if st.button("フィードバック送信"):
        feedback_df = pd.DataFrame([
            {"user_id": st.session_state.user_id, "sample_id": sid, "liked": fb, "gender": st.session_state.gender}
            for sid, fb in st.session_state.feedback.items()
        ])
        st.session_state.step = 2
        st.rerun()

# --- Step 2: 解析ステップ ---
elif st.session_state.step == 2 and user_id:
    st.header("Step 2: 解析結果")
    # ユーザーフィードバック
    feedback_df =  pd.DataFrame([
        {"sample_id": sid, "liked": fb}
        for sid, fb in st.session_state.feedback.items()
    ])

    Y = [1 if l == "好き" else 0 for l in feedback_df['liked'].values]  # likedが"好き"なら1、そうでなければ0
    # 弱識別器の出力をNumpy配列 (N x D) 形式で並べたもの
    h_list = feedback_df.merge(H_df, on="sample_id", how="left")
    H = h_list.drop(columns=["sample_id", "liked"])
    
    # # パラメータ（重み）
    lambda_reg = 0.01  # 正則化項の係数
    # QUBO行列の作成（特徴量ごとの重要度を学習する）
    QUBO = build_qboost_qubo(Y, np.array(H).T, lambda_reg)
    # #サンプラー
    sampler = oj.SASampler()
    num_reads = 100  # サンプリング回数
    # # QUBO行列を解く
    result = sampler.sample_qubo(QUBO, num_reads=num_reads)
    # st.write(result)
    
    # # 結果の取得
    best_solution = result.first.sample

    # H_df.columns から、選ばれた弱識別器の名前を取得
    selected_indices = [i for i, v in best_solution.items() if v == 1]
    selected_names = [H_df.columns[i+1] for i in selected_indices] #  # +1 するのは、最初の列が sample_id のため
    st.session_state.selected_names = selected_names

    st.write("選ばれた好みに影響するパラメータの名前一覧：")
    for name in st.session_state.selected_names:
        st.write(name)
    st.write("以上の条件に近い音声を抽出します。次へをクリックしてください。")

    K = 10
    gender_key = "M" if "男性" in st.session_state.gender else "F"
    recomend = recomend_voice(feedback_df, st.session_state.selected_names, gender_key, K)
    # --- おすすめ音声推薦 ---
    if st.button("次へ"):
        st.session_state.recomend_ids = recomend
        st.session_state.step = 3
        st.rerun()

# --- Step 3: おすすめ音声 ---
elif st.session_state.step == 3 and user_id:
    st.header("Step 3: おすすめ音声再確認")
    st.write("あなたの好みに近い音声を抽出しました。再度フィードバックをお願いします。")
    for sid in st.session_state.recomend_ids:
        row = samples_df[samples_df["sample_id"] == sid].iloc[0]
        # 未評価の初期値を「普通」に設定
        if sid not in st.session_state.feedback2:
            st.session_state.feedback2[sid] = "普通"

        with st.container():
            cols = st.columns([2, 1])
            with cols[0]:
                st.audio(row["filepath"])
            with cols[1]:
                liked = st.button(f"👍 いいね！", key=f"feedback2_{sid}")
                if liked:
                    st.session_state.feedback2[sid] = "好き"
                st.write("評価:", st.session_state.feedback2[sid])

            st.markdown("---")

    # --- フィードバック送信 ---
    if st.button("フィードバック送信"):
        feedback2_df = pd.DataFrame([
            {"user_id": st.session_state.user_id, "sample_id": sid, "liked": fb, "gender": st.session_state.gender}
            for sid, fb in st.session_state.feedback2.items()
        ])
        st.session_state.step = 4
        st.rerun()

# --- Step 4: おすすめ音声 ---
elif st.session_state.step == 4 and user_id:
    st.header("Step 4: 最終おすすめ音声")
    K = 3
    gender_key = "M" if "男性" in st.session_state.gender else "F"

    df1 = pd.DataFrame([
        {"sample_id": sid, "liked": fb}
        for sid, fb in st.session_state.feedback.items()
    ])
    df2 = pd.DataFrame([
        {"sample_id": sid, "liked": fb}
        for sid, fb in st.session_state.feedback2.items()
    ])
    feedback_df = pd.concat([df1, df2], ignore_index=True)

    recomend2 = recomend_voice(feedback_df, st.session_state.selected_names, gender_key, K)
    st.write("おすすめの音声を抽出しました。")
    st.write("最終おすすめ音声一覧：")
    for sid in recomend2:
        row = samples_df[samples_df["sample_id"] == sid].iloc[0]
        with st.container():
            cols = st.columns([1, 1])
            with cols[0]:
                st.audio(row["filepath"])
            with cols[1]:
                st.write("jvs_id:", row["jvs_id"])

            st.markdown("---")


    # 保存する情報をまとめる（DataFrameはdictに変換）
    save_data = {
        "user_id": user_id,
        "saved_at": datetime.now().isoformat(),
        "selected_names": st.session_state.selected_names,
        "feedback": feedback_df.to_dict(orient="records"),  # list of dicts
        "recommended_sample_ids": recomend2
    }
    # 保存（ファイル名は例として user_id_日時.json）
    filename = f"data/feedback/{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    st.write("おすすめの音声は好みにマッチしていましたか？")

