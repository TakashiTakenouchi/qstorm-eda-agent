#!/usr/bin/env python3
"""
Store Data Histogram Distribution Analyzer Agent
=================================================

恵比寿店・横浜元町店の売上データからヒストグラムを作成し、
確率分布（正規分布/ポアソン分布/負の二項分布）を判定して
自然言語で解説するAI Agent。

使用方法:
    # インタラクティブモード
    uv run python store_histogram_agent.py

    # 単一カラム分析
    uv run python store_histogram_agent.py --column Total_Sales

    # 全カラム分析
    uv run python store_histogram_agent.py --all

    # カスタムデータファイル
    uv run python store_histogram_agent.py --file path/to/data.xlsx
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)


# =============================================================================
# 定数定義
# =============================================================================

# デフォルトのデータファイルパス
DEFAULT_DATA_PATH = r"C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\Ensuring consistency between tabular data and time series forecast data\fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx"

# 分析対象カラム
TARGET_COLUMNS = [
    "shop",
    "shop_code",
    "Total_Sales",
    "gross_profit",
    "discount",
    "purchasing",
    "rent",
    "personnel_expenses",
    "depreciation",
    "sales_promotion",
    "head_office_expenses",
    "operating_cost",
    "Operating_profit",
    "Mens_JACKETS&OUTER2",
    "Mens_KNIT",
    "Mens_PANTS",
    "WOMEN'S_JACKETS2",
    "WOMEN'S_TOPS",
    "WOMEN'S_ONEPIECE",
    "WOMEN'S_bottoms",
    "WOMEN'S_SCARF & STOLES",
    "Inventory",
    "Months_of_inventory",
    "BEP",
    "Average_Temperature",
    "Number_of_guests",
    "Price_per_customer",
]

# カラム名の日本語定義
COLUMN_DEFINITIONS = {
    "shop": "店舗名称",
    "shop_code": "店舗コード",
    "Total_Sales": "売上高",
    "gross_profit": "粗利",
    "discount": "値引き",
    "purchasing": "仕入",
    "rent": "家賃",
    "personnel_expenses": "人件費",
    "depreciation": "減価償却費用",
    "sales_promotion": "販売促進費",
    "head_office_expenses": "本社費用",
    "operating_cost": "業務費用",
    "Operating_profit": "営業利益",
    "Mens_JACKETS&OUTER2": "男性用JACKETS&OUTER売上高",
    "Mens_KNIT": "男性用KNIT売上高",
    "Mens_PANTS": "男性用PANTS売上高",
    "WOMEN'S_JACKETS2": "女性用JACKETS売上高",
    "WOMEN'S_TOPS": "女性用TOPS売上高",
    "WOMEN'S_ONEPIECE": "女性用ONEPIECE売上高",
    "WOMEN'S_bottoms": "女性用bottoms売上高",
    "WOMEN'S_SCARF & STOLES": "女性用SCARF&STOLES売上高",
    "Inventory": "在庫金額",
    "Months_of_inventory": "在庫月数",
    "BEP": "損益分岐点",
    "Average_Temperature": "平均気温",
    "Number_of_guests": "客数",
    "Price_per_customer": "客単価",
}


# =============================================================================
# システムプロンプト
# =============================================================================

SYSTEM_PROMPT = """
あなたは小売店舗データの統計解析専門家です。

## 役割
恵比寿店・横浜元町店の売上データを分析し、各項目のヒストグラムから
最適な確率分布を特定し、ビジネスインサイトを含めて日本語で解説します。

## データコンテキスト
- 対象店舗: 恵比寿店、横浜元町店
- データ種類: 月次の売上・損益データ
- 分析項目: 売上高、粗利、各カテゴリ売上、在庫、客数など

## 分析手順
1. load_store_data でExcelデータを読み込む
2. create_histogram で指定カラムのヒストグラムを作成
3. analyze_distribution で確率分布を判定
4. ユーザーの質問に対して日本語で詳細に回答

## 判定基準
- 正規分布: 連続データで、Shapiro-Wilk検定のp値 > 0.05
- ポアソン分布: 離散カウントデータで、分散/平均 ≈ 1.0
- 負の二項分布: 離散カウントデータで、分散/平均 > 1.0（過分散）

## 出力要件
- 専門用語には簡単な説明を添える
- ビジネス的な解釈と示唆を含める
- 店舗比較がある場合は両店舗の違いを説明
"""


# =============================================================================
# グローバルデータストア
# =============================================================================

class DataStore:
    """データを保持するシングルトンクラス"""
    _instance = None
    _df: pd.DataFrame | None = None
    _file_path: str | None = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Excelデータを読み込み"""
        self._file_path = file_path
        self._df = pd.read_excel(file_path)
        return self._df

    @property
    def df(self) -> pd.DataFrame | None:
        return self._df

    @property
    def file_path(self) -> str | None:
        return self._file_path


# =============================================================================
# カスタムツール定義
# =============================================================================

@tool(
    "load_store_data",
    "Excelファイルから店舗データを読み込み、基本情報を返します。",
    {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Excelファイルのパス（省略時はデフォルトパス）"
            }
        }
    }
)
async def load_store_data(args: dict[str, Any]) -> dict[str, Any]:
    """店舗データを読み込み"""
    try:
        file_path = args.get("file_path", DEFAULT_DATA_PATH)
        store = DataStore.get_instance()
        df = store.load_data(file_path)

        # 基本情報を取得
        shops = df["shop"].unique().tolist() if "shop" in df.columns else []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        result = {
            "success": True,
            "file_path": file_path,
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "shops": shops,
            "date_range": {
                "start": str(df["Date"].min()) if "Date" in df.columns else None,
                "end": str(df["Date"].max()) if "Date" in df.columns else None
            },
            "numeric_columns": numeric_cols,
            "available_target_columns": [c for c in TARGET_COLUMNS if c in df.columns]
        }

        return {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"エラー: {str(e)}"}],
            "is_error": True
        }


@tool(
    "create_histogram",
    "指定カラムのヒストグラムを作成し、ビン情報と度数を返します。店舗別のフィルタも可能です。",
    {
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": "ヒストグラムを作成するカラム名"
            },
            "bins": {
                "type": "integer",
                "description": "ビン数（デフォルト: 10）"
            },
            "shop_filter": {
                "type": "string",
                "description": "店舗名でフィルタ（例: '恵比寿', '横浜元町'）。省略時は全店舗"
            },
            "save_image": {
                "type": "boolean",
                "description": "ヒストグラム画像を保存するか（デフォルト: False）"
            }
        },
        "required": ["column"]
    }
)
async def create_histogram(args: dict[str, Any]) -> dict[str, Any]:
    """ヒストグラムを作成"""
    try:
        store = DataStore.get_instance()
        df = store.df

        if df is None:
            return {
                "content": [{"type": "text", "text": "エラー: データが読み込まれていません。先にload_store_dataを実行してください。"}],
                "is_error": True
            }

        column = args["column"]
        bins = args.get("bins", 10)
        shop_filter = args.get("shop_filter")
        save_image = args.get("save_image", False)

        if column not in df.columns:
            return {
                "content": [{"type": "text", "text": f"エラー: カラム '{column}' が見つかりません。"}],
                "is_error": True
            }

        # 店舗フィルタ
        data_df = df
        if shop_filter and "shop" in df.columns:
            data_df = df[df["shop"] == shop_filter]
            if len(data_df) == 0:
                return {
                    "content": [{"type": "text", "text": f"エラー: 店舗 '{shop_filter}' のデータが見つかりません。"}],
                    "is_error": True
                }

        # 数値データを取得
        data = data_df[column].dropna()

        if len(data) == 0:
            return {
                "content": [{"type": "text", "text": f"エラー: カラム '{column}' に有効なデータがありません。"}],
                "is_error": True
            }

        # カテゴリカルデータの場合
        if data.dtype == 'object':
            value_counts = data.value_counts()
            result = {
                "column": column,
                "column_ja": COLUMN_DEFINITIONS.get(column, column),
                "data_type": "categorical",
                "shop_filter": shop_filter,
                "n_samples": len(data),
                "unique_values": value_counts.index.tolist(),
                "counts": value_counts.values.tolist()
            }
        else:
            # 数値データのヒストグラム
            counts, bin_edges = np.histogram(data, bins=bins)

            # 基本統計
            mean_val = float(data.mean())
            std_val = float(data.std())
            min_val = float(data.min())
            max_val = float(data.max())

            result = {
                "column": column,
                "column_ja": COLUMN_DEFINITIONS.get(column, column),
                "data_type": "numeric",
                "shop_filter": shop_filter,
                "n_samples": len(data),
                "bin_edges": [round(x, 2) for x in bin_edges.tolist()],
                "counts": counts.tolist(),
                "statistics": {
                    "mean": round(mean_val, 2),
                    "std": round(std_val, 2),
                    "min": round(min_val, 2),
                    "max": round(max_val, 2)
                }
            }

            # 画像保存
            if save_image:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
                ax.set_xlabel(COLUMN_DEFINITIONS.get(column, column))
                ax.set_ylabel('度数')
                title = f'{COLUMN_DEFINITIONS.get(column, column)} のヒストグラム'
                if shop_filter:
                    title += f' ({shop_filter})'
                ax.set_title(title)
                ax.axvline(mean_val, color='red', linestyle='--', label=f'平均: {mean_val:,.0f}')
                ax.legend()

                # 保存
                output_dir = Path("histograms")
                output_dir.mkdir(exist_ok=True)
                filename = f"{column}_{shop_filter or 'all'}.png"
                filepath = output_dir / filename
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                result["image_path"] = str(filepath)

        return {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"エラー: {str(e)}"}],
            "is_error": True
        }


@tool(
    "analyze_distribution",
    "ヒストグラムデータから確率分布を判定し、詳細な統計分析結果を返します。",
    {
        "type": "object",
        "properties": {
            "bin_edges": {
                "type": "array",
                "items": {"type": "number"},
                "description": "ヒストグラムのビン境界値"
            },
            "counts": {
                "type": "array",
                "items": {"type": "number"},
                "description": "各ビンの度数"
            },
            "column_name": {
                "type": "string",
                "description": "分析対象のカラム名（結果表示用）"
            }
        },
        "required": ["bin_edges", "counts"]
    }
)
async def analyze_distribution(args: dict[str, Any]) -> dict[str, Any]:
    """確率分布を分析"""
    try:
        bin_edges = np.array(args["bin_edges"])
        counts = np.array(args["counts"])
        column_name = args.get("column_name", "不明")

        # ビン中心値から擬似データを生成
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        data = np.repeat(bin_centers, counts.astype(int))

        if len(data) == 0:
            return {
                "content": [{"type": "text", "text": "エラー: データが空です"}],
                "is_error": True
            }

        n = len(data)
        mean_val = float(np.mean(data))
        variance = float(np.var(data, ddof=1))
        std_dev = float(np.std(data, ddof=1))
        skewness = float(stats.skew(data))
        kurtosis = float(stats.kurtosis(data))

        # 分散/平均比
        dispersion_index = variance / mean_val if mean_val > 0 else float('inf')

        # データ特性判定
        is_discrete = all(float(x).is_integer() for x in data) and data.min() >= 0

        # 正規性検定
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
        else:
            sample = np.random.choice(data, size=5000, replace=False)
            shapiro_stat, shapiro_p = stats.shapiro(sample)

        data_standardized = (data - mean_val) / std_dev if std_dev > 0 else data
        ks_stat, ks_p = stats.kstest(data_standardized, 'norm')

        normality_passed = shapiro_p > 0.05 and ks_p > 0.05

        # 分布フィッティング
        fits = {}

        # 正規分布
        mu, sigma = stats.norm.fit(data)
        ll_normal = np.sum(stats.norm.logpdf(data, mu, sigma))
        aic_normal = 4 - 2 * ll_normal
        fits["normal"] = {
            "parameters": {"mu": round(mu, 2), "sigma": round(sigma, 2)},
            "aic": round(aic_normal, 2)
        }

        # ポアソン分布（非負データのみ）
        if mean_val > 0 and data.min() >= 0:
            lambda_poisson = mean_val
            data_int = np.maximum(np.round(data).astype(int), 0)
            ll_poisson = np.sum(stats.poisson.logpmf(data_int, lambda_poisson))
            aic_poisson = 2 - 2 * ll_poisson
            fits["poisson"] = {
                "parameters": {"lambda": round(lambda_poisson, 2)},
                "aic": round(aic_poisson, 2)
            }

        # 負の二項分布
        if mean_val > 0 and variance > mean_val and data.min() >= 0:
            p = mean_val / variance if variance > 0 else 0.5
            p = max(0.001, min(0.999, p))
            r = mean_val * p / (1 - p) if p < 1 else 1.0
            r = max(0.1, r)
            data_int = np.maximum(np.round(data).astype(int), 0)
            ll_nbinom = np.sum(stats.nbinom.logpmf(data_int, r, p))
            aic_nbinom = 4 - 2 * ll_nbinom
            fits["negative_binomial"] = {
                "parameters": {"r": round(r, 2), "p": round(p, 4)},
                "aic": round(aic_nbinom, 2)
            }

        # 最適分布の判定
        valid_fits = {k: v for k, v in fits.items() if np.isfinite(v["aic"])}
        best_by_aic = min(valid_fits.items(), key=lambda x: x[1]["aic"])[0] if valid_fits else "unknown"

        # 最終判定
        evidence = []
        confidence = 0.0

        if not is_discrete:
            distribution_type = "normal"
            evidence.append("データが連続値である")
            confidence = 0.7
            if normality_passed:
                evidence.append("正規性検定にパスした")
                confidence += 0.2
            if best_by_aic == "normal":
                evidence.append("AICで正規分布が最良")
                confidence += 0.1
        else:
            if 0.8 <= dispersion_index <= 1.2:
                distribution_type = "poisson"
                evidence.append(f"分散/平均比 = {dispersion_index:.2f} ≈ 1.0")
                confidence = 0.75
            elif dispersion_index > 1.2:
                distribution_type = "negative_binomial"
                evidence.append(f"分散/平均比 = {dispersion_index:.2f} > 1.0（過分散）")
                confidence = 0.75
            else:
                distribution_type = "poisson"
                evidence.append(f"分散/平均比 = {dispersion_index:.2f}")
                confidence = 0.5

        confidence = max(0.0, min(1.0, confidence))

        result = {
            "column": column_name,
            "column_ja": COLUMN_DEFINITIONS.get(column_name, column_name),
            "distribution_type": distribution_type,
            "confidence": round(confidence, 2),
            "statistics": {
                "n_samples": n,
                "mean": round(mean_val, 2),
                "variance": round(variance, 2),
                "std_dev": round(std_dev, 2),
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "dispersion_index": round(dispersion_index, 4)
            },
            "normality_tests": {
                "shapiro_wilk": {"statistic": round(shapiro_stat, 4), "p_value": round(shapiro_p, 4)},
                "kolmogorov_smirnov": {"statistic": round(ks_stat, 4), "p_value": round(ks_p, 4)},
                "passed": normality_passed
            },
            "distribution_fits": fits,
            "best_fit_by_aic": best_by_aic,
            "evidence": evidence,
            "is_discrete": is_discrete
        }

        return {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"エラー: {str(e)}"}],
            "is_error": True
        }


@tool(
    "compare_shops",
    "指定カラムについて店舗間の分布を比較します。",
    {
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": "比較するカラム名"
            }
        },
        "required": ["column"]
    }
)
async def compare_shops(args: dict[str, Any]) -> dict[str, Any]:
    """店舗間の分布を比較"""
    try:
        store = DataStore.get_instance()
        df = store.df

        if df is None:
            return {
                "content": [{"type": "text", "text": "エラー: データが読み込まれていません。"}],
                "is_error": True
            }

        column = args["column"]
        if column not in df.columns:
            return {
                "content": [{"type": "text", "text": f"エラー: カラム '{column}' が見つかりません。"}],
                "is_error": True
            }

        if "shop" not in df.columns:
            return {
                "content": [{"type": "text", "text": "エラー: 店舗カラムが見つかりません。"}],
                "is_error": True
            }

        shops = df["shop"].unique()
        comparison = {}

        for shop in shops:
            shop_data = df[df["shop"] == shop][column].dropna()
            if len(shop_data) > 0 and shop_data.dtype in [np.float64, np.int64]:
                comparison[shop] = {
                    "n_samples": len(shop_data),
                    "mean": round(float(shop_data.mean()), 2),
                    "std": round(float(shop_data.std()), 2),
                    "min": round(float(shop_data.min()), 2),
                    "max": round(float(shop_data.max()), 2),
                    "median": round(float(shop_data.median()), 2)
                }

        # 統計検定（2店舗の場合）
        test_result = None
        if len(shops) == 2:
            data1 = df[df["shop"] == shops[0]][column].dropna()
            data2 = df[df["shop"] == shops[1]][column].dropna()
            if len(data1) > 0 and len(data2) > 0:
                # t検定
                t_stat, t_p = stats.ttest_ind(data1, data2)
                # Mann-Whitney U検定
                u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                test_result = {
                    "t_test": {"statistic": round(t_stat, 4), "p_value": round(t_p, 4)},
                    "mann_whitney_u": {"statistic": round(u_stat, 4), "p_value": round(u_p, 4)},
                    "significant_difference": t_p < 0.05 or u_p < 0.05
                }

        result = {
            "column": column,
            "column_ja": COLUMN_DEFINITIONS.get(column, column),
            "shops_comparison": comparison,
            "statistical_test": test_result
        }

        return {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"エラー: {str(e)}"}],
            "is_error": True
        }


@tool(
    "list_columns",
    "分析可能なカラムの一覧と日本語名を返します。",
    {
        "type": "object",
        "properties": {}
    }
)
async def list_columns(args: dict[str, Any]) -> dict[str, Any]:
    """カラム一覧を返す"""
    try:
        store = DataStore.get_instance()
        df = store.df

        if df is None:
            # データ未読み込み時はターゲットカラムのみ返す
            columns = [
                {"name": col, "name_ja": COLUMN_DEFINITIONS.get(col, col)}
                for col in TARGET_COLUMNS
            ]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [
                {"name": col, "name_ja": COLUMN_DEFINITIONS.get(col, col), "available": col in df.columns}
                for col in TARGET_COLUMNS
                if col in numeric_cols or col in ["shop", "shop_code"]
            ]

        result = {
            "target_columns": columns,
            "total": len(columns)
        }

        return {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"エラー: {str(e)}"}],
            "is_error": True
        }


# =============================================================================
# エージェントクラス
# =============================================================================

class StoreHistogramAgent:
    """店舗データのヒストグラム・分布分析エージェント"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # MCPサーバー作成
        self.stats_server = create_sdk_mcp_server(
            name="store_stats",
            version="1.0.0",
            tools=[
                load_store_data,
                create_histogram,
                analyze_distribution,
                compare_shops,
                list_columns
            ]
        )

        self.options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"stats": self.stats_server},
            allowed_tools=[
                "mcp__stats__load_store_data",
                "mcp__stats__create_histogram",
                "mcp__stats__analyze_distribution",
                "mcp__stats__compare_shops",
                "mcp__stats__list_columns"
            ],
            permission_mode="bypassPermissions",
            max_turns=20
        )

        self.total_cost = 0.0
        self.query_count = 0

    async def query(self, question: str) -> str:
        """自然言語で質問し、回答を取得"""
        results = []

        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(question)

            async for message in client.receive_response():
                if self.verbose:
                    print(f"[{type(message).__name__}]", end=" ")

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            results.append(block.text)
                        elif isinstance(block, ToolUseBlock) and self.verbose:
                            print(f"\n  Tool: {block.name}")

                elif isinstance(message, ResultMessage):
                    self.total_cost += message.total_cost_usd
                    self.query_count += 1
                    if self.verbose:
                        print(f"\n  Cost: ${message.total_cost_usd:.4f}")

        return "\n".join(results)

    async def analyze_column(self, column: str, shop: str | None = None) -> dict:
        """特定カラムの分布を分析"""
        shop_text = f"（{shop}）" if shop else ""
        prompt = f"""
{column}（{COLUMN_DEFINITIONS.get(column, column)}）{shop_text}のヒストグラムを作成し、
確率分布を分析してください。

以下の手順で実行してください:
1. load_store_data でデータを読み込む
2. create_histogram で {column} のヒストグラムを作成{'（shop_filter: ' + shop + '）' if shop else ''}
3. analyze_distribution で分布を判定
4. 結果を日本語で詳しく解説

特に以下の点を説明してください:
- どの確率分布に従うか
- なぜその分布と判定したか
- ビジネス的にどのような意味があるか
"""
        return await self.query(prompt)

    async def analyze_all_columns(self) -> dict:
        """全ターゲットカラムを分析"""
        results = {}

        # まずデータを読み込む
        await self.query("load_store_data でデータを読み込んでください。")

        numeric_targets = [c for c in TARGET_COLUMNS if c not in ["shop", "shop_code"]]

        for column in numeric_targets:
            print(f"\n分析中: {column} ({COLUMN_DEFINITIONS.get(column, column)})")
            try:
                result = await self.analyze_column(column)
                results[column] = result
            except Exception as e:
                results[column] = f"エラー: {str(e)}"

        return results

    def get_stats(self) -> dict:
        """セッション統計を取得"""
        return {
            "query_count": self.query_count,
            "total_cost_usd": round(self.total_cost, 4)
        }


# =============================================================================
# 質問例（確率分布関連）
# =============================================================================

EXAMPLE_QUESTIONS = [
    # 基本的な分布分析
    ("売上高はどのような確率分布に従っていますか？", "Total_Sales"),
    ("客数の分布を分析してください", "Number_of_guests"),
    ("営業利益の確率分布を教えてください", "Operating_profit"),
    ("在庫月数はどんな分布ですか？", "Months_of_inventory"),

    # 分布の比較
    ("恵比寿店と横浜元町店の売上高の分布に違いはありますか？", None),
    ("両店舗の客数の分布を比較してください", None),

    # 分布の解釈
    ("売上高が正規分布に従うとしたら、どのような意味がありますか？", None),
    ("客数がポアソン分布に従う場合、ビジネス上の示唆は何ですか？", None),

    # 統計的検定
    ("売上高は正規分布に従っていると言えますか？検定結果を教えてください", None),
    ("粗利のデータに過分散は見られますか？", "gross_profit"),

    # パラメータの解釈
    ("売上高の平均と標準偏差から、どの程度のばらつきがありますか？", None),
    ("客単価の分布パラメータを解釈してください", "Price_per_customer"),

    # カテゴリ別分析
    ("男性用商品と女性用商品の売上分布に違いはありますか？", None),
    ("季節変動を考慮すると、売上高の分布はどうなりますか？", None),
]


# =============================================================================
# インタラクティブモード
# =============================================================================

def print_welcome_screen():
    """ウェルカム画面を表示"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║   🏪 店舗データ 確率分布分析 AI Agent                   ║")
    print("║   Q-Storm EDA - Distribution Analyzer                   ║")
    print("║" + " " * 58 + "║")
    print("║   恵比寿店・横浜元町店の売上データを分析し、            ║")
    print("║   確率分布を判定して自然言語で解説します。              ║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()


def print_help():
    """ヘルプを表示"""
    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ 📖 使い方                                               │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│                                                         │")
    print("│ 🔹 自然言語で質問:                                      │")
    print("│    「売上高の分布を教えて」                             │")
    print("│    「客数はどんな確率分布に従う？」                     │")
    print("│    「恵比寿店と横浜元町店の売上を比較して」             │")
    print("│                                                         │")
    print("│ 🔹 コマンド:                                            │")
    print("│    /examples  - 質問例を表示                            │")
    print("│    /columns   - 分析可能なカラム一覧                    │")
    print("│    /analyze <カラム名> - 特定カラムの分布分析           │")
    print("│    /compare <カラム名> - 店舗間比較                     │")
    print("│    /help      - このヘルプを表示                        │")
    print("│    exit       - 終了                                    │")
    print("│                                                         │")
    print("└─────────────────────────────────────────────────────────┘")
    print()


def print_example_questions():
    """質問例を表示"""
    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ 💡 質問例（確率分布関連）                               │")
    print("├─────────────────────────────────────────────────────────┤")

    categories = {
        "分布分析": EXAMPLE_QUESTIONS[:4],
        "店舗比較": EXAMPLE_QUESTIONS[4:6],
        "分布の解釈": EXAMPLE_QUESTIONS[6:8],
        "統計的検定": EXAMPLE_QUESTIONS[8:10],
        "パラメータ解釈": EXAMPLE_QUESTIONS[10:12],
    }

    for category, questions in categories.items():
        print(f"│                                                         │")
        print(f"│ 【{category}】                                          │"[:60])
        for i, (q, _) in enumerate(questions, 1):
            # 質問を適切な長さに切り詰め
            q_display = q if len(q) <= 45 else q[:42] + "..."
            print(f"│   {i}. {q_display:<51} │"[:60])

    print("│                                                         │")
    print("│ 番号を入力するとその質問を実行します（例: 1）          │")
    print("└─────────────────────────────────────────────────────────┘")
    print()


def print_columns_summary():
    """カラムサマリーを表示"""
    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ 📊 分析可能なカラム                                     │")
    print("├─────────────────────────────────────────────────────────┤")

    for col in TARGET_COLUMNS:
        if col not in ["shop", "shop_code"]:
            ja_name = COLUMN_DEFINITIONS.get(col, col)
            print(f"│   {col:<25} {ja_name:<20} │"[:60])

    print("└─────────────────────────────────────────────────────────┘")
    print()


async def interactive_mode():
    """対話型モード"""
    print_welcome_screen()
    print_help()

    agent = StoreHistogramAgent(verbose=False)

    # 初回データ読み込み
    print("📥 データを読み込んでいます...")
    try:
        await agent.query("load_store_data でデータを読み込んでください。読み込み結果を簡潔に報告してください。")
        print("✅ データ読み込み完了")
    except Exception as e:
        print(f"⚠️  データ読み込みエラー: {e}")

    print()
    print("=" * 60)
    print("自然言語で確率分布に関する質問をしてください。")
    print("/examples で質問例を表示、/help でヘルプを表示")
    print("=" * 60)

    while True:
        try:
            print()
            user_input = input("🔍 質問を入力> ").strip()

            if not user_input:
                continue

            # 終了コマンド
            if user_input.lower() in ("exit", "quit", "q", "終了"):
                print()
                print("=" * 60)
                print(f"📊 セッション統計: {agent.get_stats()}")
                print("👋 ご利用ありがとうございました！")
                print("=" * 60)
                break

            # ヘルプ
            if user_input.lower() in ("/help", "help", "?", "ヘルプ"):
                print_help()
                continue

            # 質問例表示
            if user_input.lower() in ("/examples", "/ex", "例", "質問例"):
                print_example_questions()
                continue

            # 番号による質問選択
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(EXAMPLE_QUESTIONS):
                    question, column = EXAMPLE_QUESTIONS[idx]
                    print(f"\n選択された質問: {question}")
                    user_input = question
                else:
                    print(f"⚠️  1〜{len(EXAMPLE_QUESTIONS)} の番号を入力してください")
                    continue

            # カラム一覧
            if user_input.lower() in ("/columns", "/cols", "カラム"):
                print_columns_summary()
                continue

            # 分析コマンド
            if user_input.startswith("/analyze "):
                column = user_input[9:].strip()
                print(f"\n📊 {column} ({COLUMN_DEFINITIONS.get(column, column)}) を分析中...")
                response = await agent.analyze_column(column)
            elif user_input.startswith("/compare "):
                column = user_input[9:].strip()
                print(f"\n📊 {column} の店舗間比較を実行中...")
                response = await agent.query(
                    f"compare_shops で {column} を店舗間で比較してください。"
                    f"統計的検定の結果と、ビジネス的な解釈を日本語で詳しく解説してください。"
                )
            else:
                # 自然言語質問
                print("\n🤔 分析中...")
                response = await agent.query(user_input)

            # 回答表示
            print()
            print("┌" + "─" * 58 + "┐")
            print("│ 📝 回答                                                 │")
            print("└" + "─" * 58 + "┘")
            print()
            print(response)
            print()
            print(f"💰 累計コスト: ${agent.total_cost:.4f}")

        except KeyboardInterrupt:
            print(f"\n\n📊 統計: {agent.get_stats()}")
            print("中断されました。")
            break
        except Exception as e:
            print(f"\n❌ エラー: {type(e).__name__}: {e}")


# =============================================================================
# メイン
# =============================================================================

async def main():
    """メインエントリーポイント"""
    args = sys.argv[1:]

    if "--column" in args:
        idx = args.index("--column")
        if idx + 1 < len(args):
            column = args[idx + 1]
            agent = StoreHistogramAgent(verbose=True)
            result = await agent.analyze_column(column)
            print(result)
            print(f"\n統計: {agent.get_stats()}")
        else:
            print("エラー: --column には引数が必要です")
    elif "--all" in args:
        agent = StoreHistogramAgent(verbose=True)
        results = await agent.analyze_all_columns()
        print(f"\n完了: {len(results)} カラム分析")
        print(f"統計: {agent.get_stats()}")
    else:
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
