# -*- coding: utf-8 -*-
import librosa
import numpy as np
import pyworld as pw
import parselmouth
import pandas as pd
from parselmouth.praat import call
from scipy.signal import spectrogram, argrelmax

def calculate_stonemask_f0_statistics(x, sr):
    """
    StoneMask法で基本周波数（F0）の統計量を計算。
    """
    _f0_dio, t = pw.dio(x, sr)
    f0_stonemask = pw.stonemask(x, _f0_dio, t, sr)
    voiced_f0 = f0_stonemask[f0_stonemask > 0]
    
    if len(voiced_f0) > 0:
        min_f0 = float(np.min(voiced_f0))
        mean_f0 = float(np.mean(voiced_f0))
        max_f0 = float(np.max(voiced_f0))
    else:
        min_f0, mean_f0, max_f0 = 0.0, 0.0, 0.0
    
    return min_f0, mean_f0, max_f0

def measure_hnr(sound):
    """
    HNR（Harmonics-to-Noise Ratio）を計算。
    """
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return float(hnr) if not np.isnan(hnr) else 0.0

def measure_formants(sound):
    """
    フォルマント（F1, F2）を計算。
    """
    formant = call(sound, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
    mean_f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
    mean_f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
    return float(mean_f1) if not np.isnan(mean_f1) else 0.0, float(mean_f2) if not np.isnan(mean_f2) else 0.0

def calculate_rms(signal, frame_length=4096, hop_length=512, threshold=0.01):
    """
    RMSエネルギーの統計量を計算。
    """
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_min = float(np.min(rms[rms > threshold]) if np.any(rms > threshold) else 0.0)
    rms_max = float(np.max(rms))
    dynamic_range = float(20 * np.log10(rms_max / rms_min)) if rms_min > 0 else 0.0
    
    return {
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_min": rms_min,
        "rms_max": rms_max,
        "rms_dynamic_range": dynamic_range
    }

def calculate_harmonic_energy_ratio(signal, sr, f0):
    """
    音声信号全体で奇数次・偶数次倍音のエネルギー比率を計算。
    """
    spectrum = np.abs(np.fft.fft(signal))[: len(signal) // 2]
    freqs = np.fft.fftfreq(len(signal), d=1.0 / sr)[: len(signal) // 2]
    s_power = spectrum ** 2
    peaks = argrelmax(s_power, order=70)[0]
    
    odd_energy = 0.0
    even_energy = 0.0
    
    for peak in peaks:
        harmonic_number = round(freqs[peak] / f0)
        if harmonic_number % 2 == 0:
            even_energy += s_power[peak]
        else:
            odd_energy += s_power[peak]
    
    total_energy = odd_energy + even_energy
    if total_energy == 0:
        odd_ratio = 0.0
        even_ratio = 0.0
    else:
        odd_ratio = (odd_energy / total_energy) * 100
        even_ratio = (even_energy / total_energy) * 100
    
    return odd_ratio, even_ratio

def calculate_harmonic_energy_ratios(f, t, Sxx_power, f0, step_index, silence_threshold):
    """
    特定の時間ステップで奇数次・偶数次倍音のエネルギー比率を計算。
    """
    total_spectrum_energy = np.sum(Sxx_power[:, step_index])
    if total_spectrum_energy < silence_threshold:
        return {"odd_ratio": 0.0, "even_ratio": 0.0}
    
    f0_at_step = f0[step_index]
    if np.isnan(f0_at_step):
        return {"odd_ratio": 0.0, "even_ratio": 0.0}
    
    harmonics = [f0_at_step * n for n in range(1, int(f[-1] / f0_at_step) + 1)]
    odd_energy_list = []
    even_energy_list = []
    
    for i, harmonic in enumerate(harmonics):
        closest_idx = np.argmin(np.abs(f - harmonic))
        energy = Sxx_power[closest_idx, step_index]
        if (i + 1) % 2 == 0:
            even_energy_list.append(energy)
        else:
            odd_energy_list.append(energy)
    
    odd_energy = sum(odd_energy_list)
    even_energy = sum(even_energy_list)
    total_energy = odd_energy + even_energy
    
    if total_energy == 0:
        odd_ratio = 0.0
        even_ratio = 0.0
    else:
        odd_ratio = (odd_energy / total_energy) * 100
        even_ratio = (even_energy / total_energy) * 100
    
    return {"odd_ratio": odd_ratio, "even_ratio": even_ratio}

def calculate_ratios_for_all_steps(signal, sr, silence_threshold=1e-10):
    """
    すべての時間ステップで奇数次・偶数次倍音の割合を計算し、平均を返す。
    """
    f, t, Sxx = spectrogram(signal, sr, nperseg=1024, noverlap=512)
    Sxx_power = Sxx ** 2
    f0, voiced_flag, _ = librosa.pyin(signal, fmin=50, fmax=500, sr=sr)
    
    odd_ratios = []
    even_ratios = []
    
    for step_index in range(len(t)):
        result = calculate_harmonic_energy_ratios(f, t, Sxx_power, f0, step_index, silence_threshold)
        odd_ratios.append(result["odd_ratio"])
        even_ratios.append(result["even_ratio"])
    
    non_zero_odd_ratios = [ratio for ratio in odd_ratios if ratio > 0]
    non_zero_even_ratios = [ratio for ratio in even_ratios if ratio > 0]
    
    average_odd_ratio = float(np.mean(non_zero_odd_ratios)) if non_zero_odd_ratios else 0.0
    average_even_ratio = float(np.mean(non_zero_even_ratios)) if non_zero_even_ratios else 0.0
    
    return average_odd_ratio, average_even_ratio

def analyze_voice(filepath):
    """
    音声ファイルを解析し、特徴量（生データ）を返す。
    minf0[Hz], maxf0[Hz] は除外。

    Parameters:
        filepath (str): 音声ファイルのパス。

    Returns:
        dict: 解析結果（min_f0, mean_f0, max_f0, f0_range, hnr, mean_f1, mean_f2,
                      rms_mean, rms_std, rms_min, rms_max, rms_dynamic_range,
                      odd_ratio_1, even_ratio_1, odd_ratio_2, even_ratio_2）。
    """
    # 音声データの読み込み
    x, sr = librosa.load(filepath, sr=22050, mono=True)
    x = x.astype(np.float64)
    sound = parselmouth.Sound(filepath)
    
    # 基本周波数（F0）
    min_f0, mean_f0, max_f0 = calculate_stonemask_f0_statistics(x, sr)
    f0_range = max_f0 - min_f0 if max_f0 > 0 and min_f0 > 0 else 0.0
    
    # HNR
    hnr = measure_hnr(sound)
    
    # フォルマント
    mean_f1, mean_f2 = measure_formants(sound)
    
    # RMS
    rms_stats = calculate_rms(x)
    
    # 倍音比率（方法1：全体）
    odd_ratio_1, even_ratio_1 = calculate_harmonic_energy_ratio(x, sr, mean_f0 if mean_f0 > 0 else 100.0)
    
    # 倍音比率（方法2：時間ステップごとの平均）
    odd_ratio_2, even_ratio_2 = calculate_ratios_for_all_steps(x, sr)
    
    # 結果を辞書にまとめる
    result = {
        'min_f0': min_f0,
        'mean_f0': mean_f0,
        'max_f0': max_f0,
        'f0_range': f0_range,
        'hnr': hnr,
        'mean_f1': mean_f1,
        'mean_f2': mean_f2,
        'rms_mean': rms_stats['rms_mean'],
        'rms_std': rms_stats['rms_std'],
        'rms_min': rms_stats['rms_min'],
        'rms_max': rms_stats['rms_max'],
        'rms_dynamic_range': rms_stats['rms_dynamic_range'],
        'odd_ratio_1': odd_ratio_1,
        'even_ratio_1': even_ratio_1,
        'odd_ratio_2': odd_ratio_2,
        'even_ratio_2': even_ratio_2
    }
    
    return result

def normalize_features(features):
    """
    解析結果を生物学的基準値に基づいて0～3の範囲で正規化。

    Parameters:
        features (dict): analyze_voiceの出力（生データ）。

    Returns:
        dict: 生データと正規化済みデータを含む辞書。
    """
    # 生物学的基準値
    reference_ranges = {
        'min_f0': (60, 200),
        'mean_f0': (80, 350),
        'max_f0': (150, 450),
        'f0_range': (90, 400),
        'hnr': (5, 30),
        'mean_f1': (300, 1000),
        'mean_f2': (800, 2500),
        'rms_mean': (0.01, 0.2),
        'rms_std': (0.01, 0.1),
        'rms_min': (0.005, 0.05),
        'rms_max': (0.05, 0.5),
        'rms_dynamic_range': (10, 40),
        'odd_ratio_1': (1, 99),
        'even_ratio_1': (1, 99),
        'odd_ratio_2': (1, 99),
        'even_ratio_2': (1, 99)
    }
    
    # 結果をコピー
    normalized_features = features.copy()
    
    # 各項目を正規化
    for key in features:
        if key in reference_ranges:
            min_val, max_val = reference_ranges[key]
            value = features[key]
            
            # Min-Max正規化（0～3）
            if max_val != min_val:
                normalized_value = 3 * (value - min_val) / (max_val - min_val)
                # クリッピング
                normalized_value = max(0.0, min(3.0, normalized_value))
            else:
                normalized_value = 0.0
            
            normalized_features[f'{key}_normalized'] = normalized_value
        else:
            print(f"警告: {key} の基準値が定義されていません。")
    
    return normalized_features

# 直接利用時
if __name__ == "__main__":
    csv_path_main = "../data/processed/features_VOICEACTRESS100_001.csv"
    df = pd.read_csv(csv_path_main)

    results = []

    for _, row in df.iterrows():
        filepath = row['filepath']
        gender = row['Male_or_Female']  # 例：1つ目の追加カラム
        speaker_id = row['jvs_id']  # 例：2つ目の追加カラム

        features = analyze_voice(filepath)
        normalized = normalize_features(features)

        # 辞書に元データのカラムも追加
        normalized['jvs_id'] = speaker_id
        normalized['Male_or_Female'] = gender
        normalized['filepath'] = filepath

        results.append(normalized)

    output_df = pd.DataFrame(results)
    output_df.to_csv('analyze_and_normalize.csv', index=False)