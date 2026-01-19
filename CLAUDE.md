# Q-Storm EDA Distribution Analyzer Agent

## プロジェクト概要

Claude Agent SDKを使用した確率分布分析エージェント。
恵比寿店・横浜元町店の売上データからヒストグラムを作成し、
確率分布（正規分布/ポアソン分布/負の二項分布）を判定して自然言語で解説する。

## 技術スタック

- **Runtime**: Python 3.13+
- **Agent Framework**: Claude Agent SDK (claude-agent-sdk)
- **Web Framework**: FastAPI + Uvicorn
- **Data Analysis**: NumPy, SciPy, Pandas
- **Deployment**: Render

## ファイル構成

```
├── app.py                    # FastAPI Web API
├── store_histogram_agent.py  # メインエージェント（CLI + MCPツール）
├── eda_distribution_agent.py # 基本分布分析エージェント
├── test_eda_agent.py         # テストスイート
├── render.yaml               # Render デプロイ設定
├── pyproject.toml            # 依存関係
└── README.md                 # ドキュメント
```

## MCPカスタムツール

| ツール名 | 説明 |
|---------|------|
| `load_store_data` | Excelデータ読み込み |
| `create_histogram` | ヒストグラム作成 |
| `analyze_distribution` | 確率分布判定 |
| `compare_shops` | 店舗間比較 |
| `list_columns` | カラム一覧 |

## 確率分布判定ロジック

| データ特性 | 分散/平均比 | 判定結果 |
|-----------|------------|---------|
| 連続値 | - | 正規分布 |
| 離散カウント | ≈ 1.0 | ポアソン分布 |
| 離散カウント | > 1.0 | 負の二項分布 |

## 開発コマンド

```bash
# 依存関係インストール
uv sync

# CLI実行
uv run python store_histogram_agent.py

# Web API起動
uv run uvicorn app:app --reload --port 8000

# テスト実行
uv run python test_eda_agent.py --quick
```

## デプロイ

- **GitHub**: https://github.com/TakashiTakenouchi/qstorm-eda-agent
- **Render**: Web Service経由でデプロイ
- **環境変数**: `ANTHROPIC_API_KEY` が必要

## データソース

デフォルトデータパス:
```
C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\
Ensuring consistency between tabular data and time series forecast data\
fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx
```

## 注意事項

- HTMLテンプレート内で`%`フォーマットは使用しない（CSS競合防止）
- `.format()`またはf-stringを使用すること
- エージェントのmax_turnsは20に設定
