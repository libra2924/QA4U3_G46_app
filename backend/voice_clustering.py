import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
import plotly.express as px
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.spatial import ConvexHull
import scipy.cluster.hierarchy as sch
import pandas.io.formats.style

is_streamlit_cloud = True

def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    各特徴量のVIFを計算してDataFrameで返す関数
    """
    vif_data = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_data


def plot_vif(vif: pd.DataFrame, threshold: float = 10):
    """
    VIFを棒グラフで描画する関数
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(vif["Feature"], vif["VIF"], color="skyblue")
    ax.axhline(y=threshold, color="red", linestyle="--", label=f"VIF = {threshold} (Threshold)")
    ax.set_xticklabels(vif["Feature"], rotation=90)
    ax.set_title("Variance Inflation Factor (VIF) for Features")
    ax.set_xlabel("Features")
    ax.set_ylabel("VIF")
    ax.legend()
    return fig


# 特徴量の表を作成する関数
def create_styled_table(dataframe):
    """
    特徴量の表を作成し、小数点第2位まで表示し、色分けを適用する関数。
    """
    # 小数点第2位まで表示
    dataframe = dataframe.copy()
    dataframe = dataframe.style.format(precision=3).background_gradient(
        axis=0, cmap="coolwarm", subset=dataframe.columns[4:]  # 特徴量列を色分け
    )
    return dataframe


def get_columns_min_max(ref_df: pd.DataFrame, subset_columns=None) -> dict:
    """
    参照用 DataFrame の各列（subset_columns が指定されていればその列）の最小値、最大値を取得します。
    
    戻り値は {列名: (min, max)} の形式。
    """
    if subset_columns is None:
        subset_columns = ref_df.columns
    min_max = {}
    for col in subset_columns:
        min_max[col] = (ref_df[col].min(), ref_df[col].max())
    return min_max

def apply_colormap_using_ref_params(target_df: pd.DataFrame, col_min_max: dict, cmap_name: str = "coolwarm", subset_columns=None) -> pd.io.formats.style.Styler:
    """
    取得済みの各列の最小値・最大値(col_min_max)を利用して、
    target_df の対象列にカラーマップを適用したスタイラーを返します。
    
    各セルは、対象列ごとに個別に正規化された値に基づいて色付けされます。
    """
    if subset_columns is None:
        subset_columns = target_df.columns

    cmap = plt.get_cmap(cmap_name)
    
    def style_column(series: pd.Series):
        col = series.name
        col_min, col_max = col_min_max.get(col, (series.min(), series.max()))
        norm = mcolors.Normalize(vmin=col_min, vmax=col_max)
        # 各セルの値に対し、カラーマップに基づいた背景色を返すスタイルリストを生成
        return ['background-color: ' + mcolors.to_hex(cmap(norm(val))) for val in series]
    
    return target_df.style.apply(style_column, subset=subset_columns)

# 階層型クラスタリング関連

def plot_dendrogram(Z, jvs_ids, max_d):
    """
    デンドログラムを描画する関数
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    sch.dendrogram(Z, labels=jvs_ids, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.axhline(y=max_d, c='red', linestyle='--', label=f'Distance = {max_d}')
    ax.legend()
    ax.set_title('Hierarchical Clustering Dendrogram with Cut-Off Line')
    ax.set_xlabel('Sample (jvs_id)')
    ax.set_ylabel('Distance')
    return fig

# クラスタリングの結果の描画関連

# 平均値だけを取り出してヒートマップを描画する関数
def plot_cluster_feature_means(cluster_stats_scaled):
    """
    クラスタごとの特徴量の平均値をヒートマップで描画する関数
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_stats_scaled, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
    ax.set_title("Cluster-wise Feature Means (Scaled)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    return fig


def compute_thresholds(df: pd.DataFrame, quantiles: list = [0.2, 0.4, 0.6, 0.8]) -> dict:
    """
    各特徴量ごとに、指定された分位点に基づいてしきい値を算出する
    """
    thresholds = {}
    for feature in df.columns:
        quantile_values = df[feature].quantile(quantiles).values
        thresholds[feature] = {
            'very_low': quantile_values[0],
            'low': quantile_values[1],
            'average': quantile_values[2],
            'high': quantile_values[3]
        }
    return thresholds

def determine_region(feature, mean_val, thresholds):
    """
    Determine the region based on thresholds for a given feature and mean value.
    """
    if mean_val <= thresholds[feature]['very_low']:
        return "非常に低い"
    elif mean_val <= thresholds[feature]['low']:
        return "低い"
    elif mean_val <= thresholds[feature]['average']:
        return "平均的"
    elif mean_val <= thresholds[feature]['high']:
        return "高い"
    else:
        return "非常に高い"


def generate_feature_description(feature: str, region: str) -> str:
    """
    指定された領域に応じた自然言語での説明文を作成する
    """
    if region == "非常に高い":
        return f"{feature} は非常に高いです"
    elif region == "高い":
        return f"{feature} は高いです"
    elif region == "平均的":
        return f"{feature} は平均的な値です"
    elif region == "低い":
        return f"{feature} は低いです"
    else:
        return f"{feature} は非常に低いです"
    

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

# スタイルを適用してデータフレームを表示
def highlight_scaled_values(val, vmin=-2, vmax=2):
    """
    スケールされた値に基づいて背景色を設定する関数
    """
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    color = cmap(norm(val)) if pd.notnull(val) else (1, 1, 1, 0)  # NaNの場合は透明
    return f'background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]})'


# k-means法との比較関連

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def perform_kmeans_clustering(data, n_clusters, random_state=42):
    """
    k-means++ クラスタリングを実行する関数
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels + 1  # クラスタラベルを1始まりに変更

def perform_tsne(data, n_components=2, perplexity=30, random_state=42):
    """
    t-SNE を実行する関数
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(data)

def create_tsne_dataframe(tsne_result, cluster_labels, sample_ids):
    """
    t-SNE の結果をデータフレームに変換する関数
    """
    return pd.DataFrame({
        'Dim 1': tsne_result[:, 0],
        'Dim 2': tsne_result[:, 1],
        'Cluster': cluster_labels.astype(str),  # 凡例をカテゴリカルにするため文字列に変換
        'Sample': sample_ids
    })

def add_convex_hull(fig, data, cluster_column, x_col, y_col):
    """
    クラスタごとに凸包を描画する関数
    """
    for cluster in data[cluster_column].unique():
        cluster_points = data[data[cluster_column] == cluster]
        if len(cluster_points) < 3:
            continue  # 点数が3未満の場合はスキップ
        hull = ConvexHull(cluster_points[[x_col, y_col]].values)
        hull_points = cluster_points.iloc[hull.vertices]
        hull_points = pd.concat([hull_points, hull_points.iloc[[0]]])  # 閉じた形にする
        fig.add_trace(go.Scatter(
            x=hull_points[x_col],
            y=hull_points[y_col],
            mode='lines',
            fill='toself',
            name=f"Cluster {cluster} Area",
            opacity=0.2,
            line=dict(color='rgba(0,0,0,0)')  # 境界線を非表示
        ))
    return fig