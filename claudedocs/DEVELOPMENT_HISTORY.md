# Q-Storm EDA Distribution Analyzer Agent - Development History

**Project**: qstorm-eda-agent
**Date**: 2026-01-19
**Framework**: Claude Agent SDK

---

## Executive Summary

Claude Agent SDKを使用した確率分布分析エージェントの開発記録。恵比寿店・横浜元町店の売上データからヒストグラムを作成し、確率分布（正規分布/ポアソン分布/負の二項分布）を判定して自然言語で解説する。

---

## Development Timeline

### Phase 1: Core Agent Development (04:36)

**Commit**: `ffa3de1`
**Title**: feat: Q-Storm EDA Distribution Analyzer Agent

#### Files Created

| File | Size | Description |
|------|------|-------------|
| `store_histogram_agent.py` | 39KB | メインエージェント（CLI + MCPツール） |
| `eda_distribution_agent.py` | 30KB | 基本分布分析エージェント |
| `test_eda_agent.py` | 9.5KB | テストスイート |
| `websearch_agent.py` | 5KB | Web検索エージェント例 |

#### Custom MCP Tools Implemented

```python
# store_stats MCP Server Tools
├── load_store_data      # Excelデータ読み込み
├── create_histogram     # ヒストグラム作成
├── analyze_distribution # 確率分布判定
├── compare_shops        # 店舗間比較
└── list_columns         # カラム一覧
```

#### Distribution Classification Logic

| Data Type | Variance/Mean Ratio | Classification |
|-----------|---------------------|----------------|
| Continuous | - | Normal Distribution |
| Discrete Count | ≈ 1.0 | Poisson Distribution |
| Discrete Count | > 1.0 (overdispersed) | Negative Binomial |

#### Statistical Tests Implemented

- **Normality Tests**: Shapiro-Wilk, Kolmogorov-Smirnov
- **Comparison Tests**: t-test, Mann-Whitney U
- **Model Selection**: AIC-based comparison

---

### Phase 2: Web API Development (04:42)

**Commit**: `115002e`
**Title**: feat: Add FastAPI Web API and Render deployment

#### Files Created

| File | Description |
|------|-------------|
| `app.py` | FastAPI Web Application with HTML UI |
| `render.yaml` | Render deployment configuration |
| `Procfile` | Alternative deployment configuration |

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI with question input form |
| `/api/query` | POST | Natural language Q&A |
| `/api/analyze` | POST | Column distribution analysis |
| `/api/compare` | POST | Shop comparison |
| `/api/columns` | GET | Available columns list |
| `/api/examples` | GET | Example questions |
| `/health` | GET | Health check |

#### HTML UI Features

- Gradient purple theme with card-based layout
- Real-time question input with loading spinner
- Example question buttons for quick access
- Column tags with click-to-analyze functionality
- Cost tracking display

---

### Phase 3: CSS Formatting Fix (05:03)

**Commit**: `26f0c21`
**Title**: fix: Replace % formatting with .format() in HTML template

#### Problem

```
ValueError: % characters in CSS (e.g., width: 100%)
conflicting with Python's old-style % string formatting
```

#### Solution Applied

Changed `%s` placeholders to `{columns_json}` with `.format()` method.

---

### Phase 4: Brace Escape Fix

**Commit**: `d3b21cc`
**Title**: fix: Replace .format() with replace() to avoid CSS brace conflicts

**Issue**: `KeyError: ' box-sizing'`

#### Root Cause

The `.format()` method in Python interprets `{}` braces as format placeholders. The CSS contains:

```css
* { box-sizing: border-box; margin: 0; padding: 0; }
```

This was incorrectly parsed as a format placeholder.

#### Solution Implemented

```python
# Before (Line 421-423)
""".format(
    columns_json=str([...])
)

# After
"""
columns_data = [
    {"name": c, "name_ja": COLUMN_DEFINITIONS.get(c, c)}
    for c in TARGET_COLUMNS
    if c not in ["shop", "shop_code"]
]
html_content = html_content.replace("{columns_json}", json.dumps(columns_data, ensure_ascii=False))
```

#### Changes Made

1. Added `import json` (line 18)
2. Replaced `.format()` with `replace()` + `json.dumps()` (lines 424-430)

---

### Phase 5: Streamlit Hybrid App Migration

**Commit**: `9013445`
**Title**: feat: Hybrid Streamlit app with rule-based NLP (no API required)

#### Background

- **問題**: Claude Agent SDK版（`app.py`）がRenderデプロイ時に `Invalid API key` エラー
- **原因**: `ANTHROPIC_API_KEY` 環境変数の設定が必要
- **解決策**: APIキー不要のハイブリッドStreamlitアプリへ移行

#### 設計決定

ユーザー承認済みアプローチ:
> "API新規登録不要であればハイブリッド案を承認する"

採用設計:
1. **メニュー選択式UI** - メイン操作方法
2. **ルールベースNLP** - キーワードマッチングによる自然言語入力サポート
3. **オプショナルAPI** - 将来的なClaude API統合の余地を残す

#### Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `streamlit_app.py` | 新規 | ハイブリッドStreamlitアプリ（779行） |
| `requirements.txt` | 新規 | Render用依存関係ファイル |
| `pyproject.toml` | 更新 | Claude Agent SDKをオプショナルに |
| `render.yaml` | 更新 | Streamlitデプロイ設定 |

#### RuleBasedNLP Engine

```python
class RuleBasedNLP:
    INTENT_PATTERNS = {
        "histogram": ["ヒストグラム", "分布", "ばらつき", "散らばり", "グラフ"],
        "analyze": ["分析", "判定", "判別", "特定", "調べ"],
        "compare": ["比較", "違い", "差", "対比", "vs"],
        "normality": ["正規", "正規性", "ガウス", "検定"],
        "summary": ["概要", "サマリー", "一覧", "まとめ"],
    }
```

---

### Phase 6: Render Deployment Error Fix

**Commit**: `0893e29`
**Title**: fix: Update Procfile to use Streamlit instead of uvicorn

#### Problem

```
error: Failed to spawn: 'uvicorn'
```

Renderデプロイが失敗。

#### Root Cause Analysis

1. `Procfile` が `render.yaml` を上書きしていた
2. `Procfile`: `web: uvicorn app:app --host 0.0.0.0 --port $PORT`
3. `uvicorn` は `requirements.txt` に含まれていない（optional-dependenciesのみ）
4. Render は Procfile を優先して読み込む

#### Solution

**Procfile 修正**:

```diff
- web: uvicorn app:app --host 0.0.0.0 --port $PORT
+ web: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

#### Verification

- Render が自動デプロイを開始
- デプロイログでエラーなし
- Streamlit UI が正常に表示

---

## Render デプロイ設定

### render.yaml

```yaml
services:
  - type: web
    name: qstorm-eda-agent
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
    autoDeploy: true
```

### Procfile

```
web: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

### 環境変数

| 変数 | 必須 | 説明 |
|------|------|------|
| `PYTHON_VERSION` | Yes | Python 3.11 |
| `PORT` | Yes (自動) | Renderが自動設定 |
| `ANTHROPIC_API_KEY` | **No** | ハイブリッド版では不要 |

### Render vs Procfile の優先順位

| 設定ファイル | 優先度 | 用途 |
|-------------|--------|------|
| `Procfile` | **高** | Heroku互換、Renderも読み込む |
| `render.yaml` | 中 | Render Blueprint固有設定 |

**重要**: 両方存在する場合、`Procfile` の `startCommand` が優先される。

---

## テスト経緯

### テストスイート概要

`test_eda_agent.py` - EDA Distribution Agent テストスイート

#### テストケース定義

| テスト名 | 分布タイプ | パラメータ | 説明 |
|---------|-----------|-----------|------|
| `normal_standard` | 正規分布 | μ=50, σ=15 | 標準的な正規分布 |
| `normal_narrow` | 正規分布 | μ=100, σ=5 | 狭い正規分布 |
| `poisson_low_lambda` | ポアソン分布 | λ=3 | 低い発生率 |
| `poisson_high_lambda` | ポアソン分布 | λ=10 | 高い発生率 |
| `nbinom_overdispersed` | 負の二項分布 | r=5, p=0.3 | 過分散データ |
| `nbinom_mild` | 負の二項分布 | r=10, p=0.5 | 軽度の過分散 |

#### テスト実行コマンド

```bash
# 全テスト実行
uv run python test_eda_agent.py

# クイックテスト（1件のみ）
uv run python test_eda_agent.py --quick

# 特定の分布タイプのみ
uv run python test_eda_agent.py --type normal
uv run python test_eda_agent.py --type poisson
uv run python test_eda_agent.py --type negative_binomial
```

#### テストデータ生成

```python
# 正規分布
np.random.normal(mean, std, n)

# ポアソン分布
np.random.poisson(lam, n)

# 負の二項分布
np.random.negative_binomial(r, p, n)
```

#### 判定精度検証

| 分布タイプ | 期待される判定 | 検証ポイント |
|-----------|--------------|-------------|
| 連続値データ | 正規分布 | Shapiro-Wilk, KS検定 |
| 離散カウント (分散/平均≈1) | ポアソン分布 | 分散/平均比の閾値 |
| 離散カウント (分散/平均>1) | 負の二項分布 | 過分散の検出 |

### ローカルテスト手順

```bash
# Streamlitアプリ起動
cd /mnt/c/PyCharm/ClaudeAgent\ SDK_Try
streamlit run streamlit_app.py

# ブラウザで確認
# http://localhost:8501
```

### デプロイ前チェックリスト

- [x] `streamlit_app.py` がエラーなく起動
- [x] サンプルデータで分析機能が動作
- [x] ヒストグラムが正常に表示
- [x] 店舗間比較が正常に動作
- [x] 日本語フォントが正常に表示
- [x] `requirements.txt` に必要なパッケージがすべて含まれている
- [x] `Procfile` が Streamlit コマンドを指定

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Q-Storm EDA Agent                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastAPI Web Application                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │   │
│  │  │   /     │  │/api/*   │  │/health  │              │   │
│  │  │ HTML UI │  │REST API │  │ Check   │              │   │
│  │  └────┬────┘  └────┬────┘  └─────────┘              │   │
│  └───────┼────────────┼──────────────────────────────────┘   │
│          │            │                                      │
│  ┌───────▼────────────▼──────────────────────────────────┐   │
│  │           StoreHistogramAgent                          │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │          ClaudeSDKClient                         │  │   │
│  │  │  - system_prompt: 日本語統計解説                  │  │   │
│  │  │  - permission_mode: bypassPermissions            │  │   │
│  │  │  - max_turns: 20                                 │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                          │                             │   │
│  │  ┌───────────────────────▼──────────────────────────┐  │   │
│  │  │         MCP Server: "store_stats"                │  │   │
│  │  │  ┌────────────────────────────────────────────┐  │  │   │
│  │  │  │ Tools:                                     │  │  │   │
│  │  │  │  • load_store_data   → Excel読み込み       │  │  │   │
│  │  │  │  • create_histogram  → ヒストグラム作成    │  │  │   │
│  │  │  │  • analyze_distribution → 分布判定        │  │  │   │
│  │  │  │  • compare_shops     → 店舗間比較         │  │  │   │
│  │  │  │  • list_columns      → カラム一覧         │  │  │   │
│  │  │  └────────────────────────────────────────────┘  │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Data Layer                          │   │
│  │  ┌─────────────────┐  ┌─────────────────────────┐    │   │
│  │  │   DataStore     │  │  Excel Data Source       │    │   │
│  │  │  (Singleton)    │←─│  恵比寿店・横浜元町店    │    │   │
│  │  └─────────────────┘  └─────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
ClaudeAgent SDK_Try/
├── streamlit_app.py            # Hybrid Streamlit App (779行) ★メイン
├── app.py                      # FastAPI Web API (18KB) - 非推奨
├── store_histogram_agent.py    # Main Agent + MCP Tools (40KB)
├── eda_distribution_agent.py   # Core Distribution Analysis (30KB)
├── test_eda_agent.py           # Test Suite (9.5KB)
├── websearch_agent.py          # Web Search Example (5KB)
├── main.py                     # Entry Point (97B)
├── test_agent.py               # Basic Agent Test (456B)
├── test_agent2.py              # Agent Test v2 (967B)
├── pyproject.toml              # Dependencies (Streamlit + Optional API)
├── requirements.txt            # Render用依存関係 (APIキー不要)
├── render.yaml                 # Render Deployment (Streamlit)
├── Procfile                    # Heroku互換 (Streamlit)
├── README.md                   # Documentation
├── CLAUDE.md                   # Project Instructions
└── claudedocs/
    ├── DEVELOPMENT_HISTORY.md  # This File
    └── HANDOVER.md             # 開発引き継ぎドキュメント
```

---

## Dependencies

```toml
[project]
requires-python = ">=3.13"
dependencies = [
    "claude-agent-sdk>=0.1.20",
    "fastapi>=0.128.0",
    "matplotlib>=3.10.8",
    "numpy>=2.0.0",
    "openpyxl>=3.1.5",
    "pandas>=2.3.3",
    "python-multipart>=0.0.21",
    "scipy>=1.14.0",
    "uvicorn>=0.40.0",
]
```

---

## Claude Agent SDK Usage

### Agent Configuration

```python
from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    create_sdk_mcp_server,
    tool,
)

# Custom Tool Definition
@tool(
    "tool_name",
    "Tool description",
    {"type": "object", "properties": {...}}
)
async def tool_function(args: dict) -> dict:
    return {"content": [{"type": "text", "text": "result"}]}

# MCP Server Creation
stats_server = create_sdk_mcp_server(
    name="store_stats",
    version="1.0.0",
    tools=[tool1, tool2, ...]
)

# Agent Options
options = ClaudeAgentOptions(
    system_prompt="...",
    mcp_servers={"stats": stats_server},
    allowed_tools=["mcp__stats__tool_name"],
    permission_mode="bypassPermissions",
    max_turns=20
)

# Query Execution
async with ClaudeSDKClient(options=options) as client:
    await client.query(question)
    async for message in client.receive_response():
        # Process messages
```

---

## Data Source

**Default Path**:
```
C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\
Ensuring consistency between tabular data and time series forecast data\
fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx
```

**Target Shops**: 恵比寿店, 横浜元町店

**Analysis Columns** (24 total):
- Financial: Total_Sales, gross_profit, Operating_profit, etc.
- Product Categories: Mens_*, WOMEN'S_*
- Operational: Inventory, Months_of_inventory, BEP
- Customer: Number_of_guests, Price_per_customer
- External: Average_Temperature

---

## Git Commit History

```
0893e29 2026-01-19 fix: Update Procfile to use Streamlit instead of uvicorn
9013445 2026-01-19 feat: Hybrid Streamlit app with rule-based NLP (no API required)
d3b21cc 2026-01-19 fix: Replace .format() with replace() to avoid CSS brace conflicts
26f0c21 2026-01-19 fix: Replace % formatting with .format() in HTML template
115002e 2026-01-19 feat: Add FastAPI Web API and Render deployment
ffa3de1 2026-01-19 feat: Q-Storm EDA Distribution Analyzer Agent
```

---

## Deployment

### Local Development

```bash
# CLI Mode
uv run python store_histogram_agent.py

# Web API
uv run uvicorn app:app --reload --port 8000
```

### Render Deployment

1. Create GitHub repo: `gh repo create qstorm-eda-agent --public`
2. Connect to Render Dashboard
3. Set environment variable: `ANTHROPIC_API_KEY`
4. Deploy via render.yaml Blueprint

---

## Key Learnings

### 1. HTML Template String Formatting

**Problem**: Python's `.format()` conflicts with CSS `{}` braces

**Solution**: Use `replace()` + `json.dumps()` for safe JSON injection

```python
# Safe pattern
html = html.replace("{placeholder}", json.dumps(data, ensure_ascii=False))
```

### 2. MCP Tool Design

- Return structured `{"content": [{"type": "text", "text": "..."}]}` format
- Use `is_error: True` for error responses
- Include Japanese descriptions for user-facing output

### 3. Agent System Prompt

- Define clear analysis steps
- Include domain context (店舗データ、確率分布)
- Specify output requirements (日本語、ビジネス解釈)

---

## Future Enhancements

- [ ] Add more distribution types (exponential, gamma, beta)
- [ ] Implement time series analysis
- [ ] Add visualization export (PNG, PDF)
- [ ] Integrate with cloud storage for data sources
- [ ] Add authentication for production deployment

---

*Generated by Claude Opus 4.5 - 2026-01-19*
*Last Updated: 2026-01-19 (Phase 6: Render Deployment Error Fix)*
