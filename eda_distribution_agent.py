#!/usr/bin/env python3
"""
EDA Distribution Analysis Agent
================================

Q-Storm EDA機能のための確率分布解析エージェント。
ヒストグラムデータから確率分布（正規分布/ポアソン分布/負の二項分布）を
判定し、日本語で解説します。

使用方法:
    # 直接呼び出し
    agent = EDADistributionAgent()
    result = await agent.analyze({
        "bin_edges": [0, 10, 20, 30, 40, 50],
        "counts": [5, 15, 25, 18, 7]
    })
    print(result["explanation_ja"])

    # コマンドラインテスト
    uv run python eda_distribution_agent.py
"""

import asyncio
import json
from typing import Any

import numpy as np
from scipy import stats
from scipy.special import gammaln

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)


# =============================================================================
# システムプロンプト（日本語解説用）
# =============================================================================

SYSTEM_PROMPT = """
あなたは統計データ解析の専門家です。

## 役割
ヒストグラムデータを分析し、最適な確率分布を特定します。

## 分析手順
1. まず load_histogram_data でデータを読み込みます
2. compute_descriptive_stats で記述統計量を計算します
3. データが連続値の場合は test_normality で正規性検定を実行します
4. fit_distributions で複数の分布をフィッティングします
5. 最後に classify_distribution で最終判定を行います

## 判定基準
- 正規分布: 連続データで、Shapiro-Wilk検定のp値 > 0.05
- ポアソン分布: 離散カウントデータで、分散/平均 ≈ 1.0
- 負の二項分布: 離散カウントデータで、分散/平均 > 1.0（過分散）

## 出力要件
分析が完了したら、以下の形式でJSON出力してください:
- distribution_type: 判定された分布タイプ（"normal", "poisson", "negative_binomial"）
- confidence: 判定の確信度（0.0〜1.0）
- parameters: 推定されたパラメータ（分布タイプに応じて）
- test_results: 実行した統計検定の結果
- evidence: 判定根拠のリスト
- explanation_ja: 日本語での詳細な解説

専門用語には簡単な説明を添えてください。
実務的な解釈と注意点も含めてください。
"""


# =============================================================================
# 出力スキーマ定義
# =============================================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "distribution_type": {
            "type": "string",
            "enum": ["normal", "poisson", "negative_binomial"],
            "description": "判定された確率分布のタイプ"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "判定の確信度"
        },
        "parameters": {
            "type": "object",
            "properties": {
                "mu": {"type": "number", "description": "平均（正規分布）"},
                "sigma": {"type": "number", "description": "標準偏差（正規分布）"},
                "lambda": {"type": "number", "description": "λパラメータ（ポアソン分布）"},
                "r": {"type": "number", "description": "rパラメータ（負の二項分布）"},
                "p": {"type": "number", "description": "pパラメータ（負の二項分布）"}
            },
            "description": "推定されたパラメータ"
        },
        "test_results": {
            "type": "object",
            "properties": {
                "shapiro_wilk": {
                    "type": "object",
                    "properties": {
                        "statistic": {"type": "number"},
                        "p_value": {"type": "number"}
                    }
                },
                "kolmogorov_smirnov": {
                    "type": "object",
                    "properties": {
                        "statistic": {"type": "number"},
                        "p_value": {"type": "number"}
                    }
                },
                "aic": {"type": "object"},
                "bic": {"type": "object"}
            },
            "description": "統計検定の結果"
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "判定根拠のリスト"
        },
        "explanation_ja": {
            "type": "string",
            "description": "日本語での詳細な解説"
        }
    },
    "required": ["distribution_type", "confidence", "explanation_ja"]
}


# =============================================================================
# カスタムツール定義
# =============================================================================

@tool(
    "load_histogram_data",
    "ヒストグラムまたは生データを読み込み、分析用の配列に変換します。bin_edgesとcountsの組み合わせ、またはraw_dataを受け取ります。",
    {
        "type": "object",
        "properties": {
            "bin_edges": {
                "type": "array",
                "items": {"type": "number"},
                "description": "ヒストグラムのビン境界値（n+1個）"
            },
            "counts": {
                "type": "array",
                "items": {"type": "number"},
                "description": "各ビンの度数（n個）"
            },
            "raw_data": {
                "type": "array",
                "items": {"type": "number"},
                "description": "生データ配列（bin_edges/countsの代わりに使用可能）"
            }
        }
    }
)
async def load_histogram_data(args: dict[str, Any]) -> dict[str, Any]:
    """ヒストグラムまたは生データを読み込み、分析用の配列に変換"""
    try:
        if "raw_data" in args and args["raw_data"]:
            # 生データが提供された場合
            data = np.array(args["raw_data"], dtype=float)
            bin_edges = args.get("bin_edges")
            counts = args.get("counts")

            if bin_edges is None:
                # 自動でヒストグラムを計算
                counts_arr, bin_edges_arr = np.histogram(data, bins='auto')
                bin_edges = bin_edges_arr.tolist()
                counts = counts_arr.tolist()

        elif "bin_edges" in args and "counts" in args:
            # ヒストグラムデータが提供された場合
            bin_edges = args["bin_edges"]
            counts = args["counts"]

            # ビン中心値から擬似的な生データを生成
            bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts))]
            data_list = []
            for center, count in zip(bin_centers, counts):
                data_list.extend([center] * int(count))
            data = np.array(data_list)
        else:
            return {
                "content": [{"type": "text", "text": "エラー: bin_edges/counts または raw_data を指定してください"}],
                "is_error": True
            }

        # データ特性を分析
        is_discrete = all(float(x).is_integer() for x in data) and data.min() >= 0
        n_samples = len(data)

        result = {
            "data_loaded": True,
            "n_samples": n_samples,
            "bin_edges": bin_edges,
            "counts": counts,
            "data_range": [float(data.min()), float(data.max())],
            "is_discrete": is_discrete,
            "data_type": "離散カウントデータ" if is_discrete else "連続データ"
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
    "compute_descriptive_stats",
    "記述統計量（平均、分散、標準偏差、歪度、尖度、分散/平均比）を計算します。分散/平均比はポアソン分布と負の二項分布の判別に重要です。",
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
            }
        },
        "required": ["bin_edges", "counts"]
    }
)
async def compute_descriptive_stats(args: dict[str, Any]) -> dict[str, Any]:
    """記述統計量を計算"""
    try:
        bin_edges = np.array(args["bin_edges"])
        counts = np.array(args["counts"])

        # ビン中心値から擬似データを生成
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        data = np.repeat(bin_centers, counts.astype(int))

        if len(data) == 0:
            return {
                "content": [{"type": "text", "text": "エラー: データが空です"}],
                "is_error": True
            }

        mean = float(np.mean(data))
        variance = float(np.var(data, ddof=1))
        std_dev = float(np.std(data, ddof=1))
        skewness = float(stats.skew(data))
        kurtosis = float(stats.kurtosis(data))

        # 分散/平均比（Index of Dispersion）
        # ポアソン分布では ≈ 1.0、過分散では > 1.0
        dispersion_index = variance / mean if mean > 0 else float('inf')

        result = {
            "n_samples": len(data),
            "mean": round(mean, 4),
            "variance": round(variance, 4),
            "std_dev": round(std_dev, 4),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "dispersion_index": round(dispersion_index, 4),
            "interpretation": {
                "dispersion": "等分散（ポアソン的）" if 0.8 <= dispersion_index <= 1.2
                             else ("過分散（負の二項的）" if dispersion_index > 1.2
                                   else "過少分散")
            }
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
    "test_normality",
    "正規性検定（Shapiro-Wilk検定、Kolmogorov-Smirnov検定）を実行します。p値が0.05より大きい場合、正規分布の可能性が高いです。",
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
            }
        },
        "required": ["bin_edges", "counts"]
    }
)
async def test_normality(args: dict[str, Any]) -> dict[str, Any]:
    """正規性検定を実行"""
    try:
        bin_edges = np.array(args["bin_edges"])
        counts = np.array(args["counts"])

        # ビン中心値から擬似データを生成
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        data = np.repeat(bin_centers, counts.astype(int))

        if len(data) < 3:
            return {
                "content": [{"type": "text", "text": "エラー: サンプル数が少なすぎます（最低3個必要）"}],
                "is_error": True
            }

        # Shapiro-Wilk検定（サンプル数5000以下）
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
        else:
            # 大きなサンプルの場合はサブサンプリング
            sample = np.random.choice(data, size=5000, replace=False)
            shapiro_stat, shapiro_p = stats.shapiro(sample)

        # Kolmogorov-Smirnov検定
        # 標準化してから標準正規分布と比較
        data_standardized = (data - np.mean(data)) / np.std(data)
        ks_stat, ks_p = stats.kstest(data_standardized, 'norm')

        # 判定
        alpha = 0.05
        is_normal_shapiro = shapiro_p > alpha
        is_normal_ks = ks_p > alpha

        result = {
            "shapiro_wilk": {
                "statistic": round(float(shapiro_stat), 4),
                "p_value": round(float(shapiro_p), 4),
                "is_normal": is_normal_shapiro,
                "interpretation": "正規分布と判定" if is_normal_shapiro else "正規分布ではない"
            },
            "kolmogorov_smirnov": {
                "statistic": round(float(ks_stat), 4),
                "p_value": round(float(ks_p), 4),
                "is_normal": is_normal_ks,
                "interpretation": "正規分布と判定" if is_normal_ks else "正規分布ではない"
            },
            "overall_conclusion": "正規分布の可能性が高い" if (is_normal_shapiro and is_normal_ks) else "正規分布ではない可能性が高い",
            "alpha": alpha
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
    "fit_distributions",
    "複数の確率分布（正規分布、ポアソン分布、負の二項分布）をデータにフィッティングし、AIC/BICで比較します。",
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
            }
        },
        "required": ["bin_edges", "counts"]
    }
)
async def fit_distributions(args: dict[str, Any]) -> dict[str, Any]:
    """複数の分布をフィッティング"""
    try:
        bin_edges = np.array(args["bin_edges"])
        counts = np.array(args["counts"])

        # ビン中心値から擬似データを生成
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        data = np.repeat(bin_centers, counts.astype(int))
        n = len(data)

        if n == 0:
            return {
                "content": [{"type": "text", "text": "エラー: データが空です"}],
                "is_error": True
            }

        results = {}

        # 1. 正規分布フィッティング
        mu, sigma = stats.norm.fit(data)
        log_likelihood_normal = np.sum(stats.norm.logpdf(data, mu, sigma))
        k_normal = 2  # パラメータ数
        aic_normal = 2 * k_normal - 2 * log_likelihood_normal
        bic_normal = k_normal * np.log(n) - 2 * log_likelihood_normal

        results["normal"] = {
            "parameters": {"mu": round(float(mu), 4), "sigma": round(float(sigma), 4)},
            "log_likelihood": round(float(log_likelihood_normal), 4),
            "aic": round(float(aic_normal), 4),
            "bic": round(float(bic_normal), 4)
        }

        # 2. ポアソン分布フィッティング（離散データの場合）
        mean_val = np.mean(data)
        if mean_val > 0 and all(x >= 0 for x in data):
            # ポアソン分布のλ = 平均
            lambda_poisson = mean_val

            # 離散データに変換
            data_int = np.round(data).astype(int)
            data_int = np.maximum(data_int, 0)  # 負の値を0に

            # 対数尤度計算
            log_likelihood_poisson = np.sum(stats.poisson.logpmf(data_int, lambda_poisson))
            k_poisson = 1
            aic_poisson = 2 * k_poisson - 2 * log_likelihood_poisson
            bic_poisson = k_poisson * np.log(n) - 2 * log_likelihood_poisson

            results["poisson"] = {
                "parameters": {"lambda": round(float(lambda_poisson), 4)},
                "log_likelihood": round(float(log_likelihood_poisson), 4),
                "aic": round(float(aic_poisson), 4),
                "bic": round(float(bic_poisson), 4)
            }

        # 3. 負の二項分布フィッティング
        if mean_val > 0 and all(x >= 0 for x in data):
            variance = np.var(data, ddof=1)

            if variance > mean_val:
                # Method of moments による推定
                # E[X] = r(1-p)/p, Var[X] = r(1-p)/p^2
                # → p = mean / variance
                # → r = mean * p / (1 - p)
                p = mean_val / variance if variance > 0 else 0.5
                p = max(0.001, min(0.999, p))  # 0 < p < 1 に制限
                r = mean_val * p / (1 - p) if p < 1 else 1.0
                r = max(0.1, r)  # r > 0 に制限

                # 離散データに変換
                data_int = np.round(data).astype(int)
                data_int = np.maximum(data_int, 0)

                # 対数尤度計算（負の二項分布）
                # PMF: C(k+r-1, k) * p^r * (1-p)^k
                log_likelihood_nbinom = np.sum(stats.nbinom.logpmf(data_int, r, p))
                k_nbinom = 2
                aic_nbinom = 2 * k_nbinom - 2 * log_likelihood_nbinom
                bic_nbinom = k_nbinom * np.log(n) - 2 * log_likelihood_nbinom

                results["negative_binomial"] = {
                    "parameters": {"r": round(float(r), 4), "p": round(float(p), 4)},
                    "log_likelihood": round(float(log_likelihood_nbinom), 4),
                    "aic": round(float(aic_nbinom), 4),
                    "bic": round(float(bic_nbinom), 4)
                }

        # AIC/BICによるベストモデル選択
        aic_values = {k: v["aic"] for k, v in results.items() if "aic" in v and np.isfinite(v["aic"])}
        bic_values = {k: v["bic"] for k, v in results.items() if "bic" in v and np.isfinite(v["bic"])}

        best_aic = min(aic_values.items(), key=lambda x: x[1])[0] if aic_values else "unknown"
        best_bic = min(bic_values.items(), key=lambda x: x[1])[0] if bic_values else "unknown"

        output = {
            "fits": results,
            "model_comparison": {
                "best_by_aic": best_aic,
                "best_by_bic": best_bic,
                "aic_values": {k: round(v, 2) for k, v in aic_values.items()},
                "bic_values": {k: round(v, 2) for k, v in bic_values.items()}
            }
        }

        return {
            "content": [{"type": "text", "text": json.dumps(output, ensure_ascii=False, indent=2)}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"エラー: {str(e)}"}],
            "is_error": True
        }


@tool(
    "classify_distribution",
    "すべての分析結果を総合して最終的な分布タイプを判定します。",
    {
        "type": "object",
        "properties": {
            "is_discrete": {
                "type": "boolean",
                "description": "データが離散カウントデータかどうか"
            },
            "dispersion_index": {
                "type": "number",
                "description": "分散/平均比（Index of Dispersion）"
            },
            "normality_test_passed": {
                "type": "boolean",
                "description": "正規性検定にパスしたかどうか"
            },
            "best_fit_by_aic": {
                "type": "string",
                "description": "AICで最良とされた分布"
            },
            "best_fit_by_bic": {
                "type": "string",
                "description": "BICで最良とされた分布"
            }
        },
        "required": ["is_discrete", "dispersion_index"]
    }
)
async def classify_distribution(args: dict[str, Any]) -> dict[str, Any]:
    """分布タイプを最終判定"""
    try:
        is_discrete = args.get("is_discrete", False)
        dispersion_index = args.get("dispersion_index", 1.0)
        normality_passed = args.get("normality_test_passed", False)
        best_aic = args.get("best_fit_by_aic", "")
        best_bic = args.get("best_fit_by_bic", "")

        evidence = []
        confidence = 0.0
        distribution_type = "normal"

        # 判定ロジック
        if not is_discrete:
            # 連続データの場合
            distribution_type = "normal"
            evidence.append("データが連続値である")
            confidence = 0.7

            if normality_passed:
                evidence.append("正規性検定にパスした")
                confidence += 0.2
            else:
                evidence.append("正規性検定にパスしなかったが、連続データのため正規分布と仮定")
                confidence -= 0.1

            if best_aic == "normal" or best_bic == "normal":
                evidence.append(f"情報量基準（AIC/BIC）でも正規分布が最良")
                confidence += 0.1

        else:
            # 離散カウントデータの場合
            evidence.append("データが離散カウントデータである")

            if 0.8 <= dispersion_index <= 1.2:
                # 分散/平均比 ≈ 1.0 → ポアソン分布
                distribution_type = "poisson"
                evidence.append(f"分散/平均比 = {dispersion_index:.2f} ≈ 1.0（等分散）")
                confidence = 0.75

                if best_aic == "poisson" or best_bic == "poisson":
                    evidence.append("情報量基準でもポアソン分布が最良")
                    confidence += 0.15

            elif dispersion_index > 1.2:
                # 分散/平均比 > 1.0 → 負の二項分布（過分散）
                distribution_type = "negative_binomial"
                evidence.append(f"分散/平均比 = {dispersion_index:.2f} > 1.0（過分散）")
                confidence = 0.75

                if best_aic == "negative_binomial" or best_bic == "negative_binomial":
                    evidence.append("情報量基準でも負の二項分布が最良")
                    confidence += 0.15

            else:
                # 過少分散（稀なケース）- ポアソンまたは二項分布
                distribution_type = "poisson"
                evidence.append(f"分散/平均比 = {dispersion_index:.2f} < 1.0（過少分散）")
                evidence.append("過少分散のためポアソン分布の近似と仮定")
                confidence = 0.5

        # 確信度を0-1の範囲に制限
        confidence = max(0.0, min(1.0, confidence))

        result = {
            "distribution_type": distribution_type,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "decision_factors": {
                "is_discrete": is_discrete,
                "dispersion_index": round(dispersion_index, 4),
                "normality_test_passed": normality_passed,
                "best_fit_by_aic": best_aic,
                "best_fit_by_bic": best_bic
            }
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

class EDADistributionAgent:
    """ヒストグラムから確率分布を判定するエージェント"""

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: True の場合、詳細なログを出力
        """
        self.verbose = verbose

        # カスタムツールでMCPサーバー作成
        self.stats_server = create_sdk_mcp_server(
            name="statistics",
            version="1.0.0",
            tools=[
                load_histogram_data,
                compute_descriptive_stats,
                test_normality,
                fit_distributions,
                classify_distribution
            ]
        )

        self.options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"stats": self.stats_server},
            allowed_tools=[
                "mcp__stats__load_histogram_data",
                "mcp__stats__compute_descriptive_stats",
                "mcp__stats__test_normality",
                "mcp__stats__fit_distributions",
                "mcp__stats__classify_distribution"
            ],
            permission_mode="bypassPermissions",
            max_turns=15
        )

        self.total_cost = 0.0
        self.query_count = 0

    async def analyze(self, histogram_data: dict) -> dict[str, Any]:
        """
        ヒストグラムを分析して分布を判定

        Args:
            histogram_data: {
                "bin_edges": [0, 10, 20, ...],  # ビン境界値
                "counts": [5, 15, 25, ...]       # 各ビンの度数
            }
            または
            {
                "raw_data": [1, 2, 3, ...]  # 生データ
            }

        Returns:
            {
                "distribution_type": "normal" | "poisson" | "negative_binomial",
                "confidence": float,
                "parameters": {...},
                "evidence": [...],
                "explanation_ja": "日本語での解説"
            }
        """
        if "bin_edges" in histogram_data and "counts" in histogram_data:
            prompt = f"""
以下のヒストグラムデータを分析して、最適な確率分布を判定してください。

ビン境界: {histogram_data['bin_edges']}
度数: {histogram_data['counts']}

ツールを順番に使用して分析し、最後に日本語で詳細な解説を含む結果をJSON形式で出力してください。
"""
        elif "raw_data" in histogram_data:
            prompt = f"""
以下の生データを分析して、最適な確率分布を判定してください。

データ: {histogram_data['raw_data'][:50]}{'...' if len(histogram_data['raw_data']) > 50 else ''}
（全{len(histogram_data['raw_data'])}件）

ツールを順番に使用して分析し、最後に日本語で詳細な解説を含む結果をJSON形式で出力してください。
"""
        else:
            raise ValueError("histogram_data には bin_edges/counts または raw_data が必要です")

        result_text = ""

        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(prompt)

            async for message in client.receive_response():
                if self.verbose:
                    print(f"[{type(message).__name__}]")

                if isinstance(message, ResultMessage):
                    self.total_cost += message.total_cost_usd
                    self.query_count += 1

                    if self.verbose:
                        print(f"  Cost: ${message.total_cost_usd:.4f}")

                # AssistantMessageからテキストを抽出
                if hasattr(message, 'content'):
                    for block in message.content:
                        if hasattr(block, 'text'):
                            result_text = block.text

        # JSONを抽出してパース
        try:
            # JSON部分を抽出
            import re
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "distribution_type": "unknown",
                    "confidence": 0.0,
                    "explanation_ja": result_text
                }
        except json.JSONDecodeError:
            return {
                "distribution_type": "unknown",
                "confidence": 0.0,
                "explanation_ja": result_text
            }

    def get_stats(self) -> dict:
        """セッション統計を取得"""
        return {
            "query_count": self.query_count,
            "total_cost_usd": round(self.total_cost, 4)
        }


# =============================================================================
# テスト用メイン関数
# =============================================================================

async def main():
    """テスト実行"""
    print("=" * 60)
    print("  EDA Distribution Analysis Agent")
    print("  Q-Storm確率分布解析エージェント")
    print("=" * 60)
    print()

    agent = EDADistributionAgent(verbose=True)

    # テストケース1: 正規分布的なデータ
    print("【テスト1】正規分布的なヒストグラム")
    print("-" * 40)
    normal_data = {
        "bin_edges": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "counts": [2, 5, 12, 25, 35, 32, 22, 10, 4, 1]
    }

    result = await agent.analyze(normal_data)
    print(f"\n判定結果: {result.get('distribution_type', 'unknown')}")
    print(f"確信度: {result.get('confidence', 0)}")
    print(f"\n解説:\n{result.get('explanation_ja', 'なし')}")

    print("\n" + "=" * 60)
    print(f"統計: {agent.get_stats()}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
