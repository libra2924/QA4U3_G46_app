from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
from scipy.spatial import ConvexHull
import pandas.io.formats.style

from backend.voice_clustering import *

# --- セッション初期化 ---
if 'clust_use_clumnus' not in st.session_state:
    st.session_state.clust_use_clumnus = ['min_f0']
if "clust_step" not in st.session_state:
    st.session_state.clust_step = 1


# ページ全体のレイアウトを広げる設定
st.set_page_config(layout="wide")

# CSVファイルのパス
csv_file_path = "./data/processed/features_VOICEACTRESS100_001.csv"

# CSVデータを読み込む
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    st.error("ファイルが見つかりません: ./data/processed/features_VOICEACTRESS100_001.csv")

# タイトルを表示
st.title("音声クラスタリング")


# ステップ1: クラスタリング設定/特徴量の選択

# 1. クラスタリング設定
st.header("1. クラスタリング設定")
gender = st.radio("対象の性別を選択してください", ("男性", "女性"))
gender_filter = "M" if gender == "男性" else "F"

# 2. データの前処理
st.header("2. データの前処理と表示")

# 男性/女性のみを抽出
df = df[df["Male_or_Female"] == gender_filter].reset_index(drop=True)

# メタ情報の列を定義
meta_cols = ['jvs_id', 'Male_or_Female','filepath']

# 特徴量のみを抽出
df_X = df.drop(columns=meta_cols)
# odd_ratio_xのみを抽出 (※odd_ratioとeven_ratioは相関しているため、odd_ratioのみを使用)
df_X = df_X.drop(columns=['even_ratio_1', 'even_ratio_2'])


st.subheader("特徴データ一覧")
styled_table = create_styled_table(df.drop(columns=["filepath"]))  # filepath列は削除
st.dataframe(styled_table, use_container_width=True, height=500)  # 高さを指定して表示

# 標準化
scaler = StandardScaler()
df_X_scaled = scaler.fit_transform(df_X)


# 3. VIFの計算と表示
st.header("3. VIFの計算と表示(多重共線性の確認)")

# VIFの概要説明を表示
st.subheader("VIFとは？")
st.markdown("""
VIF（分散膨張係数）は、回帰モデルにおける多重共線性を評価するための指標です。
多重共線性とは、複数の独立変数が強い相関を持つ状態を指し、この状態では回帰係数の推定値が不安定になり、モデルの信頼性が低下する可能性があります。
VIFの値が高いほど、多重共線性の影響が大きいことを示します。一般的に、VIFが10を超える場合は注意が必要とされています。
""")

# 標準化された特徴量をDataFrameに変換
X_df = pd.DataFrame(df_X_scaled, columns=df_X.columns)

# 特徴量フィルタリング（VIFが閾値以下）
vif_initial = calculate_vif(X_df)

# VIFの可視化
fig = plot_vif(vif_initial, threshold=10)
st.subheader("VIFの可視化")
st.pyplot(fig)


# 4.使用する特徴量の設定
st.header("4. 使用する特徴量の設定")
defalut_features = [
    "min_f0", "mean_f0", "max_f0", "hnr", "mean_f1", "mean_f2",
    "rms_mean", "rms_min", "odd_ratio_1"
]

# 特徴量の概要
# 使用する特徴量の選択
st.subheader("使用する特徴量の選択")
selected_features = st.multiselect(
    "使用する特徴量を選択してください",
    options=df_X.columns.tolist(),
    default=defalut_features
)
st.write("デフォルトでは、主にVIFが10以下の特徴量が選択されています。")

st.subheader("特徴量の概要")
st.markdown("""
- **min_f0[Hz]**: 話者のメタデータに基づく基本周波数(最小、最大) ※話者の属性データであることに注意
- **X_f0**: 音声データの分析に基づく基本周波数(最小、最大、平均)        
- **hnr**: 音声のハーモニックノイズ比
- **mean_f1**: 平均F1周波数
- **mean_f2**: 平均F2周波数
- **rms_X**: RMSエネルギー (平均、最小、最大、標準偏差)
- **rms_dynamic_range**: RMSエネルギーのダイナミックレンジ
- **odd_ratio_X**: 音声の偶数時倍音比率 (算出法1/2) ※odd_ratioとeven_ratioは相関あるため、odd_ratioのみを使用可能
""")

if st.button("使用特徴量を決定してクラスタリングを実行"):
    # 選択された特徴量をセッションステートに保存
    st.session_state.clust_use_clumnus = selected_features
    st.session_state.clust_step = 2



# step2:クラスタリングの実行
if st.session_state.clust_step >= 2:
    st.markdown(f"使用する特徴量:{st.session_state.clust_use_clumnus}")

    # 5.クラスタリングの実行(階層的クラスタリング)
    st.header("5. クラスタリングの実行(階層的クラスタリング)")
    # 階層的クラスタリングの実行

    # 選択された特徴量でデータをフィルタリング
    df_X_filtered = df_X[selected_features]
    df_X_scaled_filtered = scaler.fit_transform(df_X_filtered)

    # 距離行列とリンク情報を作成（Ward法）
    Z = sch.linkage(df_X_scaled_filtered, method='ward')

    # サンプル名として jvs_id を使う（見やすくするため）
    jvs_ids = df['jvs_id'].values

    # デンドログラムの描画
    st.subheader("デンドログラム")
    max_d = st.slider("デンドログラムのカットオフ距離を選択してください", min_value=0.0, max_value=10.0, value=5.6, step=0.1)
    fig = plot_dendrogram(Z, jvs_ids, max_d)
    st.pyplot(fig)

    # カットオフ距離に基づいてクラスタを作成
    cluster_labels = fcluster(Z, t=max_d, criterion='distance')
    current_cluster_count = len(set(cluster_labels))
    st.write(f"現在のクラスタ数: {current_cluster_count}")

    # クラスタラベルをデータフレームに追加
    df['hierarchical_cluster'] = cluster_labels

    # クラスタリング結果の表示
    st.subheader("クラスタリング結果")
    st.write(f"カットオフ距離: {max_d}")
    st.write(f"クラスタ数: {current_cluster_count}")

    # クラスタごとのデータを表示
    st.dataframe(df[['jvs_id', 'hierarchical_cluster']].sort_values(by='hierarchical_cluster'))

    # クラスタ数の決定
    st.subheader("クラスタ数の決定")
    if st.button("現在のクラスタ数で分析を行います"):
        # 選択された特徴量をセッションステートに保存
        st.session_state.clust_step = 3
        st.session_state.clust_use_count = current_cluster_count
    


# step3:クラスタの特徴の分析
if st.session_state.clust_step >= 3:

    st.markdown(f"使用するクラスタ数:{st.session_state.clust_use_count}")
    # クラスタリング結果の表示
    st.header("6. クラスタリング結果の表示")

    # クラスタごとの特徴量の平均と標準偏差を計算
    df['hierarchical_cluster'] = cluster_labels
    df_scaled = pd.DataFrame(df_X_scaled_filtered, columns=selected_features)
    df_scaled['hierarchical_cluster'] = cluster_labels
    cluster_stats_scaled = df_scaled.groupby('hierarchical_cluster').mean()

    # ヒートマップを描画
    st.subheader("クラスタごとの特徴量の平均値 (スケール後)")
    fig = plot_cluster_feature_means(cluster_stats_scaled)
    st.pyplot(fig)

    # クラスタごとの平均値の絶対値を計算
    absolute_means = cluster_stats_scaled.abs()

    # 各クラスタで平均値が大きい特徴量を取得
    top_features_per_cluster = {
        cluster: absolute_means.loc[cluster].nlargest(5)
        for cluster in absolute_means.index
    }

    # 結果をデータフレームとして保存（必要に応じて）
    top_features_df = pd.DataFrame(top_features_per_cluster).T

    # クラスタの選択
    st.subheader("クラスタの選択")
    selected_cluster = st.selectbox("クラスタを選択してください", options=sorted(set(cluster_labels)))
    st.write(f"選択されたクラスタ: {selected_cluster}")

    # 選択されたクラスタのデータをフィルタリング
    selected_cluster_data = df[df['hierarchical_cluster'] == selected_cluster]
    st.write(f"選択されたクラスタのデータ数: {len(selected_cluster_data)}")
    
    
    # 選択されたクラスタの特徴を表示
    st.subheader("選択されたクラスタの特徴")

    # feature_table = generate_feature_table(df_X_filtered, thresholds)
    # 選択されたクラスタの特徴を表で表示

    # 選択されたクラスタのデータをスケール前のデータフレームから取得
    selected_cluster_original_data = df_X_filtered[df['hierarchical_cluster'] == selected_cluster]

    # 全体の平均値と選択されたクラスタの平均値を計算（スケール前）
    overall_means_original = df_X_filtered.mean()
    cluster_means_original = selected_cluster_original_data.mean()

    # 全体の平均値と選択されたクラスタの平均値を計算（スケール後）
    overall_means_scaled = df_scaled.mean()
    cluster_means_scaled = df_scaled[df_scaled['hierarchical_cluster'] == selected_cluster].mean()

    # 比較用のデータフレームを作成（スケール前とスケール後）
    comparison_df = pd.DataFrame({
        "Feature": cluster_means_original.index,
        "Cluster Mean (Scaled)": cluster_means_scaled.values[:-1],
        "Cluster Mean (Original)": cluster_means_original.values,
        "Overall Mean (Original)": overall_means_original.values,
        })
    
    

    styled_comparison_df = comparison_df.style.applymap(
    lambda val: highlight_scaled_values(val, vmin=-2, vmax=2), subset=["Cluster Mean (Scaled)"]
    )
    st.dataframe(styled_comparison_df, use_container_width=True)

    # 全サンプルを使用して各特徴量ごとにしきい値を算出(5段階)
    thresholds = compute_thresholds(df_X_filtered)

    # 選択されたクラスタの特徴を自然言語で説明
    st.subheader("クラスタの特徴を自然言語で説明(絶対値上位3つ)")
    # 上位3つの特徴量を絶対値で選択して説明
    top_features = absolute_means.loc[selected_cluster].nlargest(3).index.tolist()
    descriptions = []
    for feature in top_features:
        mean_val = cluster_means_original[feature]
        # Use the function to determine the region
        region = determine_region(feature, mean_val, thresholds)

        description = generate_feature_description(feature, region)
        descriptions.append(description)
    st.write("選択されたクラスタの特徴(5段階):")
    for desc in descriptions:
        st.write(f"- **{desc}**")

    # 説明文を表示

    st.subheader("7. クラスタごとの個別の音声の特徴と視聴")

    # 選択されたクラスタのデータをソート
    sorted_df = selected_cluster_data.sort_values(by="jvs_id")

    print(selected_cluster_data)

    # レイアウトを2列に分割
    col1, col2 = st.columns([1, 7])

    # 左側に再生ボタンを表示
    with col1:
        st.subheader("再生ボタン")
        display_audio_list(sorted_df)
    # 右側に特徴量の表を表示
    with col2:
        st.subheader("特徴量一覧")
        # 既存のDataFrameから対象列の最小・最大を取得
        col_min_max = get_columns_min_max(df, subset_columns=selected_features)
        # 別のDataFrame (other_df) に対して、同じカラーマップの色付けを適用
        styled_table_cluster = apply_colormap_using_ref_params(sorted_df, col_min_max, cmap_name="coolwarm", subset_columns=selected_features)
        st.dataframe(styled_table_cluster, use_container_width=True, height=400)


    # Streamlitで改行を挿入
    st.markdown("---------")
    st.write("\n")
    st.write("\n")

    # おまけ:k-means++クラスタリングをクリックで表示
    with st.expander("Appendix: K-means++クラスタリングと階層型クラスタリングの比較"):
        # k-means++ クラスタリングの実行
        k = st.session_state.clust_use_count
        cluster_labels_kmeans = perform_kmeans_clustering(df_X_scaled_filtered, n_clusters=k)

        # クラスタラベルをデータフレームに追加
        df['kmeans_cluster'] = cluster_labels_kmeans

        # t-SNE の実行
        X_tsne = perform_tsne(df_X_scaled_filtered)

        # 階層クラスタリングの結果をプロット
        tsne_df_hierarchical = create_tsne_dataframe(X_tsne, cluster_labels, jvs_ids)
        fig_hierarchical = px.scatter(
            tsne_df_hierarchical,
            x='Dim 1',
            y='Dim 2',
            color='Cluster',
            hover_data=['Sample'],
            title="t-SNE with Hierarchical Clustering Labels",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_hierarchical = add_convex_hull(fig_hierarchical, tsne_df_hierarchical, 'Cluster', 'Dim 1', 'Dim 2')

        # k-means++ クラスタリングの結果をプロット
        tsne_df_kmeans = create_tsne_dataframe(X_tsne, cluster_labels_kmeans, jvs_ids)
        fig_kmeans = px.scatter(
            tsne_df_kmeans,
            x='Dim 1',
            y='Dim 2',
            color='Cluster',
            hover_data=['Sample'],
            title="t-SNE with K-means++ Clustering Labels",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_kmeans = add_convex_hull(fig_kmeans, tsne_df_kmeans, 'Cluster', 'Dim 1', 'Dim 2')

        st.subheader("クラスタリング結果の比較 (t-SNE 可視化)")
        st.subheader("t-SNEとは？")
        st.markdown("""
        t-SNE（t-Distributed Stochastic Neighbor Embedding）は、高次元データを低次元（通常2次元または3次元）に
        可視化するための非線形次元削減手法です。この手法は、データ間の局所的な類似性を保ちながら、
        クラスタやデータの分布を視覚的に捉えやすくすることを目的としています。
        注意点:
        t-SNEは近いデータ同士の関係（局所構造）はよく反映しますが、大域的な構造（全体の相対的な位置関係）
        は必ずしも正確に表現されないため、どのグループが本当に離れているのかは注意が必要。
        """)

        # プロットを左右に並べる
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_hierarchical, use_container_width=True)
        with col2:
            st.plotly_chart(fig_kmeans, use_container_width=True)
    
