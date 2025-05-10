import pandas as pd
import numpy as np
import openjij as oj

# ---------------------------------------------
# データの準備
# ---------------------------------------------

def recomend_voice(feedback_df, selected_names, gender, K, lambda_penalty=10.0):
    """
    アプローチ1に基づくQUBO行列を自作で構築する。
    
    Parameters:
    - feedback_df: # ユーザーフィードバックデータ
    - selected_names: # 選ばれた弱識別器のカラム名
    - gender:   # 性別選択
    - K: 選ぶサンプル数
    - lambda_penalty: 制約のペナルティ係数
    
    Returns:
    - sample_ids: # 選ばれたサンプルのID
    """

    # --- データ読み込み ---
    samples_df = pd.read_csv("data/processed/samples.csv")
    samples_df = samples_df[samples_df["Male_or_Female"]==gender]  # 性別でフィルタリング

    # 好き
    liked_df = feedback_df[feedback_df["liked"] == "好き"]
    
    # それ以外の sample_id を含むデータフレーム（例：samples_df から分けたい場合）
    used_sample_ids = feedback_df["sample_id"].unique()
    other_df = samples_df[~samples_df["sample_id"].isin(used_sample_ids)]

    cls_df = pd.read_csv("data/processed/H_matrix_with_descriptions.csv")
    #読み込んだcsvから特定のカラムだけを取得
    cls_df = cls_df[["sample_id"] + selected_names]

    liked_df = liked_df.merge(cls_df, on="sample_id", how="left")
    liked_df = liked_df.drop(columns=["sample_id","liked"])  # 列を削除
    teachers = np.array(liked_df)  # 形状：(D, N) = (サンプル数, 特徴量次元（選ばれた弱分類器の数）)

    features_df = other_df.merge(cls_df, on="sample_id", how="left")
    features_df_drop = features_df.drop(columns=["sample_id","jvs_id","Male_or_Female","filepath"])  # 列を削除
    features = np.array(features_df_drop)  # 形状：(D, N) = (サンプル数, 特徴量次元（選ばれた弱分類器の数）)

    N = features.shape[0]

    # 教師ベクトルの平均（教師データを抽出）
    teacher_avg = np.mean(teachers, axis=0)

    # ---------------------------------------------
    # 教師データとのスコア作成
    # ---------------------------------------------
    # 単項係数 h_i の計算
    Q = np.zeros((N, N))
    
    # 単項係数 h_i の計算（対角要素）
    for i in range(N):
        similarity = np.dot(features[i], teacher_avg)
        Q[i, i] = -similarity + lambda_penalty * (1 - 2 * K)
    
    # 二次項係数 J_{ij} の計算（非対角要素）
    for i in range(N):
        for j in range(i + 1, N):
            Q[i, j] = lambda_penalty
            Q[j, i] = lambda_penalty  # 対称行列
    
    sampler = oj.SASampler()
    result = sampler.sample_qubo(Q, num_reads=100)
    best_solution = result.first.sample
    # 値が1のインデックスだけ取り出す
    selected_indices = [index for index, flag in best_solution.items() if flag == 1]

    # 対応する行だけ抽出
    selected_rows = features_df.loc[selected_indices]
    # sample_id のみを抽出
    sample_ids = selected_rows["sample_id"].tolist()
    # sample_ids を返す
    return sample_ids




if __name__ == "__main__":
    # --- データ読み込み ---
    data = [
        {"sample_id": 78, "liked": "好き"},
        {"sample_id": 23, "liked": "好き"},
        {"sample_id": 81, "liked": "普通"},
        {"sample_id": 6,  "liked": "普通"},
        {"sample_id": 50, "liked": "普通"},
        {"sample_id": 73, "liked": "普通"},
        {"sample_id": 70, "liked": "好き"},
        {"sample_id": 86, "liked": "普通"},
        {"sample_id": 74, "liked": "好き"},
        {"sample_id": 68, "liked": "普通"},
    ]
    feedback_df = pd.DataFrame(data) # ユーザーフィードバックデータ
    selected_names = ["mean_f0 > 269.185","min_f0 > 143.387","f0_range > 325.109","hnr > 20.330","mean_f1 > 819.711","rms_mean > 0.028","rms_mean > 0.037","rms_std > 0.035","odd_ratio_1 > 57.520","odd_ratio_1 > 82.044","even_ratio_1 > 30.218","odd_ratio_2 > 55.719","odd_ratio_2 > 83.471"]  # 選ばれた弱識別器のカラム名（引数として受け取る）
    K = 10
    gender = "M"  # 男性のデータを選ぶ場合は "M"、女性のデータを選ぶ場合は "F"
    recomend = recomend_voice(feedback_df, selected_names, gender, K)
    print("推薦されたサンプルid:")
    print(recomend)