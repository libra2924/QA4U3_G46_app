# Streamlit マルチページアプリ

このプロジェクトは、Streamlitを使用したマルチページ構成のウェブアプリケーションです。メインページ、3Dボクセル可視化ページ、音声マッチングページの3つのページで構成されています。`data/` ディレクトリに格納されたCSVデータを読み込み、データ処理や可視化を行います。

## 目次
- [プロジェクト概要](#プロジェクト概要)
- [機能](#機能)
- [ディレクトリ構造](#ディレクトリ構造)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [データ](#データ)
- [依存関係](#依存関係)

## プロジェクト概要
このアプリケーションは、以下の3つのページといくつかのサンプルページを持つStreamlitベースのウェブアプリです：
1. **メインページ**: アプリの概要を紹介  
2. **3Dボクセルページ**: 3Dボクセルデータの可視化  
3. **音声提案ページ**: サンプル音声の再生用  
4. **音声サンプルページ**: サンプル音声の再生用  
5. **バックエンド処理（`backend/` ディレクトリ）**: データ処理・音声解析などの共通処理を関数として分離しています。UIと処理ロジックを明確に分けることで保守性を高めています。

アプリは、`data/` ディレクトリ内のCSVデータを使用します。生データは `data/raw/` に、前処理済みのデータ（例: 特徴量）は `data/processed/` に格納されます。

## 機能
- Streamlitの `pages/` ディレクトリを使用したマルチページナビゲーション。
- 3Dボクセルデータの可視化。
- 音声の好みのアンケートとサンプル音声の提案。
- バックエンド処理との連携による再利用可能な音声解析・特徴量抽出。

## ディレクトリ構造
```
project/
├── app.py                    # Streamlitアプリ　メインページ
├── backend/                  # バックエンド処理（関数など）
│   ├── __init__.py
│   └── analyzer.py           # 音声解析・特徴量処理などの共通処理
├── data/                     # データ格納ディレクトリ
│   ├── raw/                  # 生データ（未処理）
│   │   ├── jvs_ver1/         # サンプル生データ
│   ├── processed/            # 前処理済みデータ（例: 特徴量）
│   │   ├── features.csv      # サンプル特徴量データ
│   ├── feedback/             # フィードバックデータ
│   │   ├── username.json     # ユーザーからのフィードバックデータ
├── images/                   # 自作画像など（読み込み用）
│   └── logo.png              
├── pages/                    # 各ページのスクリプト
│   ├── voxel_page.py         # 3Dボクセル可視化ページ
│   ├── audio_page.py         # 音声提案ページ
│   └── voice_viewer_page.py  # 音声サンプルページ
├── requirements.txt          # Python依存関係
├── README.md                 # このファイル
```

## インストール
1. **リポジトリをクローン**:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **仮想環境の作成**（推奨）:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
   ```

3. **依存関係のインストール**:
   ```bash
   pip install -r requirements.txt
   ```

4. **データファイルの準備**:
   - 生データ（例: `jvs_ver1/`）を `data/raw/` に配置。
   - 必要に応じて、前処理済みデータ（例: `features.csv`）を `data/processed/` に配置。

## 使用方法
1. **Streamlitアプリの起動**:
   ```bash
   streamlit run app.py
   ```

2. **アプリへのアクセス**:
   - ブラウザで `http://localhost:8501` を開く。
   - サイドバーからページを選択：
     - **メインページ**: アプリの概要を表示。
     - **3Dボクセルページ**: 3Dボクセル可視化とCSVデータの表示。
     - **音声サンプルページ**: サンプル音声の再生、または録音（ローカル環境推奨）。

3. **バックエンド処理の利用**:
   - 各ページや`app.py`から、共通処理を呼び出す際は以下のようにインポートします：
     ```python
     from backend.analyzer import analyze_voice, normalize_features
     ```

## データ
- **生データ（raw）**: `data/raw/` に格納（例: `sample.csv`）。ユーザーやセンサーから収集した未処理のデータ。
- **前処理済みデータ**: `data/processed/` に格納（例: `features.csv`）。クリーニングや特徴量エンジニアリング済みのデータ。
- **サンプルデータ**: テスト用に、`sample.csv` を以下のような形式で作成可能：
  ```csv
  id,value1,value2
  1,10,20
  2,15,25
  ```

## 依存関係
`requirements.txt` に記載された主なパッケージ：
- `streamlit`: ウェブアプリのフレームワーク。
- `pandas`: CSVデータの処理。
- `numpy`: 数値計算。
- `matplotlib`: プロットライブラリ。
- `openjij`: QUBO最適化ライブラリ。

インストール方法：
```bash
pip install -r requirements.txt
```

---

*最終更新: 2025年4月29日*

