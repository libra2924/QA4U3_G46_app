import numpy as np
import pandas as pd
import openjij as oj


def generate_weak_classifiers_variable_thresholds(features, feature_names, threshold_dict):
    """
    特徴量としきい値の組み合わせから、弱識別器を自動生成する。
    
    Parameters:
    - X: 特徴量の2次元配列（NumPy配列またはpandasのvalues）
    - feature_names: 特徴量名のリスト
    - bins_per_feature: 各特徴量ごとにいくつのしきい値を作るか

    Returns:
    - classifiers: [(関数, 説明)] のリスト
    """

    classifiers = []
    descriptions = []
    n_features = features.shape[1]

    for i in range(n_features):
        feature_values = features[:, i]
        feature_name = feature_names[i]
        num_thresholds = threshold_dict.get(feature_name, 3)  # デフォルトは3

        thresholds = np.linspace(feature_values.min(), feature_values.max(), num=num_thresholds + 2)[1:-1]

        for threshold in thresholds:
            def make_classifier(index=i, th=threshold):
                return lambda sample: 1 if sample[index] > th else -1

            clf = make_classifier()
            description = f"{feature_name} > {threshold:.3f}"
            classifiers.append((clf, description))

    return classifiers


# 特徴量データ（音声の16個の特徴、例：ピッチや音量など）
df = pd.read_csv("data/processed/analyze_and_normalize.csv")

# 2. 特徴量と列名を抽出
feature_names = [
    "mean_f0", "max_f0", "min_f0", "f0_range", "hnr",
    "mean_f1", "mean_f2",
    "rms_mean", "rms_std", "rms_min", "rms_max", "rms_dynamic_range",
    "odd_ratio_1", "even_ratio_1", "odd_ratio_2", "even_ratio_2"
]
features = df[feature_names].values

classifiers = []
descriptions = []
threshold_dict = {
    "min_f0": 5,
    "mean_f0": 5,
    "max_f0": 5,
    "f0_range": 5,
    "hnr": 5,
    "mean_f1": 3,
    "mean_f2": 3,
    "rms_mean": 3,
    "rms_std": 3,
    "rms_min": 3,
    "rms_max": 3,
    "rms_dynamic_range": 3,
    "odd_ratio_1": 3,
    "even_ratio_1": 3,
    "odd_ratio_2": 3,
    "even_ratio_2": 3,
}

classifiers = generate_weak_classifiers_variable_thresholds(features, feature_names, threshold_dict)


# 1つ目の識別器を使ってみる
sample = features[0]  # 最初のサンプル
clf, desc = classifiers[1]
print(f"識別器: {desc} -> 出力: {clf(sample)}")


def compute_classifier_outputs(features, classifiers):
    """
    各識別器に対して X の各サンプルを適用し、
    H[i][j] = i番目の識別器がj番目のサンプルに対して出力した結果
    となる 2次元 numpy 配列を返す。
    """
    M = len(classifiers)   # 識別器の数
    N = len(features)      # サンプルの数
    H = np.zeros((M, N))
    D = []                  #説明リスト

    for i, (h, desc) in enumerate(classifiers):
        H[i] = np.array([h(x) for x in features])
        D.append(desc)

    return H, D


H, descs = compute_classifier_outputs(features, classifiers)
# DataFrame化
H_df = pd.DataFrame(H.T, columns=descs)
H_df.to_csv("data/processed/H_matrix_with_descriptions.csv", index=False)