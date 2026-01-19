#!/usr/bin/env python3
"""
Q-Storm EDA Distribution Analyzer - FastAPI Web API
====================================================

確率分布分析エージェントのWeb API。
自然言語で質問を受け付け、確率分布の分析結果をJSONで返します。

使用方法:
    # ローカル起動
    uv run uvicorn app:app --reload --port 8000

    # 本番起動
    uv run uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from store_histogram_agent import (
    StoreHistogramAgent,
    TARGET_COLUMNS,
    COLUMN_DEFINITIONS,
    EXAMPLE_QUESTIONS,
)


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    """質問リクエスト"""
    question: str = Field(..., description="自然言語の質問", min_length=1)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "売上高はどのような確率分布に従っていますか？"},
                {"question": "客数の分布を分析してください"},
            ]
        }
    }


class QueryResponse(BaseModel):
    """質問レスポンス"""
    question: str
    answer: str
    cost_usd: float


class AnalyzeRequest(BaseModel):
    """分析リクエスト"""
    column: str = Field(..., description="分析対象のカラム名")
    shop: str | None = Field(None, description="店舗名でフィルタ（恵比寿, 横浜元町）")


class AnalyzeResponse(BaseModel):
    """分析レスポンス"""
    column: str
    column_ja: str
    analysis: str
    cost_usd: float


class CompareRequest(BaseModel):
    """比較リクエスト"""
    column: str = Field(..., description="比較対象のカラム名")


class ColumnInfo(BaseModel):
    """カラム情報"""
    name: str
    name_ja: str


class ExampleQuestion(BaseModel):
    """質問例"""
    question: str
    column: str | None


# =============================================================================
# Global State
# =============================================================================

agent: StoreHistogramAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    global agent

    # 起動時: エージェント初期化
    print("🚀 Initializing Q-Storm EDA Agent...")
    agent = StoreHistogramAgent(verbose=False)

    # データ読み込み
    try:
        await agent.query("load_store_data でデータを読み込んでください。")
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"⚠️ Data loading failed: {e}")

    yield

    # 終了時
    print(f"📊 Session stats: {agent.get_stats() if agent else 'N/A'}")
    print("👋 Shutting down...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Q-Storm EDA Distribution Analyzer",
    description="""
## 確率分布分析AI Agent

恵比寿店・横浜元町店の売上データを分析し、確率分布を判定して自然言語で解説します。

### 主な機能
- **自然言語Q&A**: 確率分布に関する質問に日本語で回答
- **ヒストグラム分析**: 指定カラムの分布を自動判定
- **店舗間比較**: 統計的検定による比較分析

### 判定可能な分布
- 正規分布 (Normal Distribution)
- ポアソン分布 (Poisson Distribution)
- 負の二項分布 (Negative Binomial Distribution)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """ホームページ - 質問入力フォーム"""
    html_content = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q-Storm EDA Distribution Analyzer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 1.8em;
        }
        .subtitle { color: #666; margin-bottom: 20px; }
        .input-group { margin-bottom: 20px; }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        input[type="text"], select {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102,126,234,0.4);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .examples {
            display: grid;
            gap: 10px;
            margin-top: 15px;
        }
        .example-btn {
            background: #f5f5f5;
            color: #333;
            padding: 12px 15px;
            border-radius: 8px;
            text-align: left;
            font-size: 14px;
        }
        .example-btn:hover {
            background: #e8e8e8;
            transform: none;
            box-shadow: none;
        }
        .result {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            white-space: pre-wrap;
            font-family: inherit;
            line-height: 1.6;
            display: none;
        }
        .result.show { display: block; }
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .cost {
            color: #888;
            font-size: 12px;
            margin-top: 10px;
            text-align: right;
        }
        .columns-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .column-tag {
            background: #e8f4f8;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 13px;
            cursor: pointer;
        }
        .column-tag:hover { background: #d0e8f0; }
        .column-tag code {
            background: #fff;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🏪 Q-Storm EDA Distribution Analyzer</h1>
            <p class="subtitle">確率分布分析AI Agent - 自然言語で質問してください</p>

            <div class="input-group">
                <label for="question">📝 質問を入力</label>
                <input type="text" id="question" placeholder="例: 売上高はどのような確率分布に従っていますか？">
            </div>

            <button onclick="submitQuestion()" id="submitBtn">🔍 分析する</button>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>分析中です... しばらくお待ちください</p>
            </div>

            <div class="result" id="result"></div>
        </div>

        <div class="card">
            <h3>💡 質問例</h3>
            <div class="examples">
                <button class="example-btn" onclick="setQuestion('売上高はどのような確率分布に従っていますか？')">
                    📊 売上高はどのような確率分布に従っていますか？
                </button>
                <button class="example-btn" onclick="setQuestion('客数の分布を分析してください')">
                    👥 客数の分布を分析してください
                </button>
                <button class="example-btn" onclick="setQuestion('恵比寿店と横浜元町店の売上高の分布に違いはありますか？')">
                    🏬 恵比寿店と横浜元町店の売上高の分布に違いはありますか？
                </button>
                <button class="example-btn" onclick="setQuestion('営業利益は正規分布に従っていると言えますか？')">
                    📈 営業利益は正規分布に従っていると言えますか？
                </button>
                <button class="example-btn" onclick="setQuestion('在庫月数の分布パラメータを解釈してください')">
                    📦 在庫月数の分布パラメータを解釈してください
                </button>
            </div>
        </div>

        <div class="card">
            <h3>📊 分析可能なカラム</h3>
            <div class="columns-grid" id="columns"></div>
        </div>
    </div>

    <script>
        const columns = {columns_json};

        // カラム一覧を表示
        const columnsDiv = document.getElementById('columns');
        columns.forEach(col => {
            const tag = document.createElement('div');
            tag.className = 'column-tag';
            tag.innerHTML = `<code>${col.name}</code> ${col.name_ja}`;
            tag.onclick = () => setQuestion(`${col.name_ja}の分布を分析してください`);
            columnsDiv.appendChild(tag);
        });

        function setQuestion(q) {
            document.getElementById('question').value = q;
        }

        async function submitQuestion() {
            const question = document.getElementById('question').value.trim();
            if (!question) {
                alert('質問を入力してください');
                return;
            }

            const btn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            btn.disabled = true;
            loading.classList.add('show');
            result.classList.remove('show');

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();

                if (response.ok) {
                    result.innerHTML = data.answer +
                        `<div class="cost">💰 Cost: $${data.cost_usd.toFixed(4)}</div>`;
                    result.classList.add('show');
                } else {
                    result.innerHTML = `❌ エラー: ${data.detail || 'Unknown error'}`;
                    result.classList.add('show');
                }
            } catch (error) {
                result.innerHTML = `❌ エラー: ${error.message}`;
                result.classList.add('show');
            } finally {
                btn.disabled = false;
                loading.classList.remove('show');
            }
        }

        // Enterキーで送信
        document.getElementById('question').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') submitQuestion();
        });
    </script>
</body>
</html>
    """

    # Generate columns JSON safely (avoid CSS brace conflicts with .format())
    columns_data = [
        {"name": c, "name_ja": COLUMN_DEFINITIONS.get(c, c)}
        for c in TARGET_COLUMNS
        if c not in ["shop", "shop_code"]
    ]
    html_content = html_content.replace("{columns_json}", json.dumps(columns_data, ensure_ascii=False))

    return HTMLResponse(content=html_content)


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    自然言語で質問

    確率分布に関する質問を受け付け、分析結果を日本語で返します。
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        answer = await agent.query(request.question)
        return QueryResponse(
            question=request.question,
            answer=answer,
            cost_usd=agent.total_cost
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    特定カラムの分布を分析

    指定されたカラムのヒストグラムを作成し、確率分布を判定します。
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if request.column not in TARGET_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid column. Available: {TARGET_COLUMNS}"
        )

    try:
        analysis = await agent.analyze_column(request.column, request.shop)
        return AnalyzeResponse(
            column=request.column,
            column_ja=COLUMN_DEFINITIONS.get(request.column, request.column),
            analysis=analysis,
            cost_usd=agent.total_cost
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
async def compare(request: CompareRequest):
    """
    店舗間の分布を比較

    恵比寿店と横浜元町店のデータを統計的に比較します。
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if request.column not in TARGET_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid column. Available: {TARGET_COLUMNS}"
        )

    try:
        result = await agent.query(
            f"compare_shops で {request.column} を店舗間で比較してください。"
            f"統計的検定の結果と、ビジネス的な解釈を日本語で詳しく解説してください。"
        )
        return {
            "column": request.column,
            "column_ja": COLUMN_DEFINITIONS.get(request.column, request.column),
            "comparison": result,
            "cost_usd": agent.total_cost
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/columns", response_model=list[ColumnInfo])
async def get_columns():
    """分析可能なカラム一覧を取得"""
    return [
        ColumnInfo(name=col, name_ja=COLUMN_DEFINITIONS.get(col, col))
        for col in TARGET_COLUMNS
        if col not in ["shop", "shop_code"]
    ]


@app.get("/api/examples", response_model=list[ExampleQuestion])
async def get_examples():
    """質問例を取得"""
    return [
        ExampleQuestion(question=q, column=c)
        for q, c in EXAMPLE_QUESTIONS
    ]


@app.get("/api/stats")
async def get_stats():
    """セッション統計を取得"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent.get_stats()


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
