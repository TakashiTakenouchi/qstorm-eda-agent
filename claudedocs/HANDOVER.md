# Q-Storm EDA Distribution Analyzer - 開発引き継ぎドキュメント

**作成日**: 2026-01-19
**プロジェクト**: qstorm-eda-agent
**現在のステータス**: ハイブリッドStreamlitアプリ デプロイ完了

---

## 1. プロジェクト概要

### 1.1 背景
- **元の問題**: Claude Agent SDK版（`app.py`）がRenderデプロイ時に `Invalid API key` エラーで動作せず
- **原因**: `ANTHROPIC_API_KEY` 環境変数が未設定
- **解決策**: APIキー不要のハイブリッドStreamlitアプリへ移行

### 1.2 設計決定
ユーザー承認済みアプローチ:
> "API新規登録不要であればハイブリッド案を承認する"

採用した設計:
1. **メニュー選択式UI** - メイン操作方法
2. **ルールベースNLP** - キーワードマッチングによる自然言語入力サポート
3. **オプショナルAPI** - 将来的なClaude API統合の余地を残す

---

## 2. ファイル構成と変更状況

### 2.1 新規作成ファイル

| ファイル | 状態 | 説明 |
|---------|------|------|
| `streamlit_app.py` | ✅ 完成 | ハイブリッドStreamlitアプリ（779行） |
| `requirements.txt` | ✅ 完成 | Render用依存関係ファイル |
| `CLAUDE.md` | ✅ 完成 | プロジェクト指示ファイル |
| `claudedocs/DEVELOPMENT_HISTORY.md` | ✅ 完成 | 開発履歴ドキュメント |

### 2.2 変更済みファイル

| ファイル | 変更内容 |
|---------|----------|
| `pyproject.toml` | Claude Agent SDKをオプショナル依存に変更、Streamlitをメインに |
| `render.yaml` | FastAPI → Streamlitデプロイ設定に変更 |

### 2.3 Git状態（未コミット）

```bash
modified:   pyproject.toml
modified:   render.yaml
untracked:  CLAUDE.md
untracked:  claudedocs/
untracked:  requirements.txt
untracked:  streamlit_app.py
```

---

## 3. 技術アーキテクチャ

### 3.1 ハイブリッドアプリ構成

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI                             │
├─────────────────┬──────────────────┬────────────────────────┤
│  Tab1: NLP入力  │  Tab2: メニュー  │  Tab3: 全体概要        │
├─────────────────┴──────────────────┴────────────────────────┤
│                  RuleBasedNLP Engine                        │
│  - キーワードマッチング                                      │
│  - 意図検出 (histogram/analyze/compare/normality/summary)   │
│  - エンティティ抽出 (column/shop)                           │
├─────────────────────────────────────────────────────────────┤
│              Statistical Analysis (SciPy)                   │
│  - 分布判定 (正規/ポアソン/負の二項)                         │
│  - 正規性検定 (Shapiro-Wilk, KS)                            │
│  - 店舗間比較 (t検定, Mann-Whitney U)                       │
├─────────────────────────────────────────────────────────────┤
│                 Data Layer (Pandas)                         │
│  - Excelファイル読み込み                                    │
│  - サンプルデータ生成                                       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 RuleBasedNLPクラス

```python
class RuleBasedNLP:
    INTENT_PATTERNS = {
        "histogram": ["ヒストグラム", "分布", "ばらつき", "散らばり", "グラフ", "可視化"],
        "analyze": ["分析", "判定", "判別", "特定", "調べ", "確認"],
        "compare": ["比較", "違い", "差", "対比", "vs", "ＶＳ"],
        "normality": ["正規", "正規性", "ガウス", "検定", "テスト"],
        "summary": ["概要", "サマリー", "一覧", "まとめ", "全体"],
    }

    SHOP_PATTERNS = {
        "恵比寿": ["恵比寿", "えびす", "エビス", "ebisu"],
        "横浜元町": ["横浜", "元町", "よこはま", "ヨコハマ", "yokohama"],
    }

    @classmethod
    def parse_query(cls, query: str) -> dict:
        # Returns: {"intent", "column", "shop", "confidence"}
```

### 3.3 分布判定ロジック

| データ特性 | 分散/平均比 | 判定結果 |
|-----------|------------|---------|
| 連続値 | - | 正規分布 |
| 離散カウント | 0.8〜1.2 | ポアソン分布 |
| 離散カウント | > 1.2 | 負の二項分布 |

---

## 4. 依存関係

### 4.1 requirements.txt（Render用）

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
openpyxl>=3.1.0
```

### 4.2 pyproject.toml

```toml
[project]
name = "qstorm-eda-agent"
version = "2.0.0"
dependencies = [
    "streamlit>=1.28.0",
    "matplotlib>=3.7.0",
    "numpy>=1.24.0",
    "openpyxl>=3.1.0",
    "pandas>=2.0.0",
    "scipy>=1.11.0",
]

[project.optional-dependencies]
api = [
    "claude-agent-sdk>=0.1.20",
    "fastapi>=0.100.0",
    "python-multipart>=0.0.6",
    "uvicorn>=0.23.0",
]
```

---

## 5. デプロイ設定

### 5.1 render.yaml

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

### 5.2 環境変数

| 変数 | 必須 | 説明 |
|------|------|------|
| `PYTHON_VERSION` | Yes | Python 3.11 |
| `PORT` | Yes (自動) | Renderが自動設定 |
| `ANTHROPIC_API_KEY` | **No** | ハイブリッド版では不要 |

---

## 6. 完了済みタスク

### 6.1 デプロイ完了

- [x] **ローカルテスト**: `streamlit run streamlit_app.py`
- [x] **Gitコミット**: すべての変更をコミット
- [x] **GitHubプッシュ**: `git push origin master`
- [x] **Procfile修正**: uvicorn → Streamlit に変更 (commit: `0893e29`)
- [x] **Renderデプロイ確認**: 自動デプロイ成功

### 6.2 デプロイ修正履歴

**問題**: `error: Failed to spawn: 'uvicorn'`

**原因**: Procfileが古いuvicornコマンドを参照していた

**解決**: Procfileを修正
```diff
- web: uvicorn app:app --host 0.0.0.0 --port $PORT
+ web: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

### 6.3 参考コマンド

```bash
# ローカルテスト
cd /mnt/c/PyCharm/ClaudeAgent\ SDK_Try
streamlit run streamlit_app.py

# Git操作
git status
git log --oneline -5
```

---

## 7. 過去の修正履歴

### 7.1 CSS波括弧エスケープ修正

**問題**: `app.py` 421行目で `KeyError: ' box-sizing'`

**原因**: `.format()` がCSS内の `{ box-sizing: ... }` をPythonプレースホルダーとして解釈

**解決**: `.format()` を `json.dumps()` + `replace()` に変更

```python
# Before (エラー発生)
""".format(columns_json=str([...]))

# After (修正済み)
columns_data = [{"name": c, "name_ja": COLUMN_DEFINITIONS.get(c, c)}
                for c in TARGET_COLUMNS if c not in ["shop", "shop_code"]]
html_content = html_content.replace("{columns_json}",
    json.dumps(columns_data, ensure_ascii=False))
```

**コミット**: `d3b21cc fix: Replace .format() with replace() to avoid CSS brace conflicts`

---

## 8. アプリ機能一覧

### 8.1 Tab1: 自然言語入力

- テキストボックスに日本語で質問を入力
- RuleBasedNLPが意図・カラム・店舗を自動検出
- 信頼度スコアを表示
- 質問例ボタンで入力補助

### 8.2 Tab2: メニュー選択

- ドロップダウンでカラム選択
- 分析タイプ選択（ヒストグラム/比較/検定）
- 店舗フィルタ選択

### 8.3 Tab3: 全体概要

- 全カラムの統計サマリーテーブル
- 相関行列ヒートマップ

### 8.4 共通機能

- Excelファイルアップロード
- サンプルデータ生成
- 分布判定と自然言語解説
- 店舗間統計検定

---

## 9. 参考リンク

- **GitHub**: https://github.com/TakashiTakenouchi/qstorm-eda-agent
- **Render Dashboard**: Renderコンソールで確認
- **比較参考**: profit-improvement-dashboard（Streamlit版、正常動作）

---

## 10. 引き継ぎ者への注意事項

1. **`app.py` は使用しない** - FastAPI版はAPIキーが必要。ハイブリッド版は `streamlit_app.py`
2. **テストは必須** - ローカルで動作確認してからプッシュ
3. **render.yaml** - startCommandが `streamlit run` になっていることを確認
4. **フォント** - 日本語フォントはDejaVu Sansにフォールバック設定済み

---

*最終更新: 2026-01-19 by Claude Code*
*デプロイ修正完了: Procfile uvicorn → Streamlit (commit: 0893e29)*
