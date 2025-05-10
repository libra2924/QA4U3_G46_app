import numpy as np


def build_qboost_qubo(y, h_list, lambda_reg=0.01):
    """
    QBoost用のQUBO行列をディクショナリ形式で構築する（for文のみ使用）。
    
    Parameters:
    - y: ラベルベクトル (D,)
    - h_list: 弱識別器の出力をリスト形式で並べたもの（N x D）
    - lambda_reg: 正則化項の係数
    
    Returns:
    - QUBO: 辞書形式 {(i, j): value}
    """
    D = len(y)  # データ点数 (D)
    N = len(h_list)  # 弱学習器数 (N)
    H = h_list  # N x D のリスト（H[i][d] が c_i(x^(d))）

    QUBO = {}

    # 二次項 Q_{ij} と一次項 Q_{ii} を計算
    for i in range(N):
        for j in range(i, N):
            # シグマ ∑_d c_i(x^(d)) c_j(x^(d)) を計算
            sum_ci_cj = 0
            for d in range(D):
                sum_ci_cj += H[i][d] * H[j][d]

            if i == j:
                # 対角項 Q_{ii} = (D/N^2) - (2/N) ∑_d c_i(x^(d)) y^(d) + λ
                # H[i][d] ∈ {-1, +1} なら ∑_d (H[i][d])^2 = N
                sum_ci_y = 0
                for d in range(D):
                    sum_ci_y += H[i][d] * y[d]
                QUBO[(i, i)] = (D / (D * D)) - (2 / D) * sum_ci_y + lambda_reg
            else:
                # 非対角項 Q_{ij} = (2/N^2) ∑_d c_i(x^(d)) c_j(x^(d))
                QUBO[(i, j)] = (2 / (D * D)) * sum_ci_cj

    return QUBO