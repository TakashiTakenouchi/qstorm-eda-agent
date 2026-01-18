#!/usr/bin/env python3
"""
EDA Distribution Agent テストスイート
=====================================

各分布タイプのテストケースを実行して、エージェントの判定精度を検証します。

使用方法:
    uv run python test_eda_agent.py                  # 全テスト実行
    uv run python test_eda_agent.py --quick          # クイックテスト（1件のみ）
    uv run python test_eda_agent.py --type normal    # 特定の分布タイプのみ
"""

import asyncio
import sys
import numpy as np
from scipy import stats

from eda_distribution_agent import EDADistributionAgent


# =============================================================================
# テストデータ生成
# =============================================================================

def generate_normal_histogram(n: int = 1000, mean: float = 50, std: float = 15, bins: int = 10) -> dict:
    """正規分布に従うヒストグラムを生成"""
    data = np.random.normal(mean, std, n)
    counts, bin_edges = np.histogram(data, bins=bins)
    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "expected": "normal",
        "params": {"mu": mean, "sigma": std}
    }


def generate_poisson_histogram(n: int = 1000, lam: float = 5.0, bins: int = 15) -> dict:
    """ポアソン分布に従うヒストグラムを生成"""
    data = np.random.poisson(lam, n)
    max_val = int(data.max()) + 1
    bin_edges = np.arange(0, max_val + 1) if max_val < bins else np.linspace(0, max_val, bins + 1)
    counts, bin_edges = np.histogram(data, bins=bin_edges)
    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "expected": "poisson",
        "params": {"lambda": lam}
    }


def generate_negative_binomial_histogram(n: int = 1000, r: float = 5, p: float = 0.3, bins: int = 15) -> dict:
    """負の二項分布に従うヒストグラムを生成"""
    data = np.random.negative_binomial(r, p, n)
    max_val = int(data.max()) + 1
    bin_edges = np.arange(0, max_val + 1) if max_val < bins else np.linspace(0, max_val, bins + 1)
    counts, bin_edges = np.histogram(data, bins=bin_edges)

    # 分散/平均比を計算して確認
    mean = np.mean(data)
    var = np.var(data, ddof=1)
    dispersion = var / mean if mean > 0 else 1.0

    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts.tolist(),
        "expected": "negative_binomial",
        "params": {"r": r, "p": p},
        "dispersion_index": dispersion
    }


# =============================================================================
# テストケース定義
# =============================================================================

TEST_CASES = {
    "normal_standard": {
        "description": "標準的な正規分布（平均50、標準偏差15）",
        "generator": lambda: generate_normal_histogram(1000, 50, 15, 10),
        "expected": "normal"
    },
    "normal_narrow": {
        "description": "狭い正規分布（平均100、標準偏差5）",
        "generator": lambda: generate_normal_histogram(500, 100, 5, 12),
        "expected": "normal"
    },
    "poisson_low_lambda": {
        "description": "ポアソン分布（λ=3）- 低い発生率",
        "generator": lambda: generate_poisson_histogram(800, 3.0, 12),
        "expected": "poisson"
    },
    "poisson_high_lambda": {
        "description": "ポアソン分布（λ=10）- 高い発生率",
        "generator": lambda: generate_poisson_histogram(1000, 10.0, 15),
        "expected": "poisson"
    },
    "nbinom_overdispersed": {
        "description": "負の二項分布（r=5, p=0.3）- 過分散データ",
        "generator": lambda: generate_negative_binomial_histogram(1000, 5, 0.3, 15),
        "expected": "negative_binomial"
    },
    "nbinom_mild": {
        "description": "負の二項分布（r=10, p=0.5）- 軽度の過分散",
        "generator": lambda: generate_negative_binomial_histogram(800, 10, 0.5, 12),
        "expected": "negative_binomial"
    }
}


# =============================================================================
# テスト実行
# =============================================================================

async def run_test(agent: EDADistributionAgent, test_name: str, test_case: dict) -> dict:
    """単一テストを実行"""
    print(f"\n{'='*60}")
    print(f"テスト: {test_name}")
    print(f"説明: {test_case['description']}")
    print(f"期待される分布: {test_case['expected']}")
    print("-" * 60)

    # テストデータ生成
    data = test_case["generator"]()
    print(f"ビン数: {len(data['counts'])}")
    print(f"サンプル数: {sum(data['counts'])}")
    if "dispersion_index" in data:
        print(f"分散/平均比: {data['dispersion_index']:.2f}")

    # 分析実行
    print("\n分析中...")
    histogram_input = {
        "bin_edges": data["bin_edges"],
        "counts": data["counts"]
    }

    try:
        result = await agent.analyze(histogram_input)

        detected = result.get("distribution_type", "unknown")
        confidence = result.get("confidence", 0)
        expected = test_case["expected"]

        success = detected == expected
        status = "PASS" if success else "FAIL"

        print(f"\n結果: [{status}]")
        print(f"  検出: {detected} (確信度: {confidence:.0%})")
        print(f"  期待: {expected}")

        if "explanation_ja" in result:
            # 解説の最初の200文字を表示
            explanation = result["explanation_ja"]
            if len(explanation) > 200:
                explanation = explanation[:200] + "..."
            print(f"\n解説（抜粋）:\n  {explanation}")

        return {
            "test_name": test_name,
            "success": success,
            "detected": detected,
            "expected": expected,
            "confidence": confidence
        }

    except Exception as e:
        print(f"\nエラー: {type(e).__name__}: {e}")
        return {
            "test_name": test_name,
            "success": False,
            "error": str(e)
        }


async def run_all_tests(test_filter: str | None = None):
    """全テストを実行"""
    print("=" * 60)
    print("  EDA Distribution Agent テストスイート")
    print("=" * 60)

    agent = EDADistributionAgent(verbose=False)

    # テストフィルタリング
    tests_to_run = TEST_CASES
    if test_filter:
        tests_to_run = {k: v for k, v in TEST_CASES.items() if test_filter in k or test_filter == v["expected"]}
        if not tests_to_run:
            print(f"警告: '{test_filter}' に一致するテストがありません")
            return

    results = []
    for test_name, test_case in tests_to_run.items():
        result = await run_test(agent, test_name, test_case)
        results.append(result)

    # サマリー
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)

    passed = sum(1 for r in results if r.get("success", False))
    total = len(results)

    for r in results:
        status = "PASS" if r.get("success", False) else "FAIL"
        if "error" in r:
            print(f"  [{status}] {r['test_name']}: エラー - {r['error']}")
        else:
            print(f"  [{status}] {r['test_name']}: 検出={r['detected']}, 期待={r['expected']}")

    print("-" * 60)
    print(f"合計: {passed}/{total} テスト成功 ({100*passed/total:.0f}%)")
    print(f"累計コスト: ${agent.total_cost:.4f}")
    print("=" * 60)

    return passed == total


async def run_quick_test():
    """クイックテスト（1件のみ）"""
    print("=" * 60)
    print("  クイックテスト（正規分布）")
    print("=" * 60)

    agent = EDADistributionAgent(verbose=True)

    # 固定のテストデータ
    test_data = {
        "bin_edges": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "counts": [2, 5, 12, 25, 35, 32, 22, 10, 4, 1]
    }

    print(f"ビン境界: {test_data['bin_edges']}")
    print(f"度数: {test_data['counts']}")
    print("-" * 60)

    result = await agent.analyze(test_data)

    print("\n" + "=" * 60)
    print("分析結果")
    print("=" * 60)
    print(f"分布タイプ: {result.get('distribution_type', 'unknown')}")
    print(f"確信度: {result.get('confidence', 0):.0%}")

    if "parameters" in result:
        print(f"パラメータ: {result['parameters']}")

    if "evidence" in result:
        print(f"判定根拠:")
        for e in result["evidence"]:
            print(f"  - {e}")

    print(f"\n解説:\n{result.get('explanation_ja', 'なし')}")

    print("\n" + "-" * 60)
    print(f"統計: {agent.get_stats()}")

    return result


# =============================================================================
# メイン
# =============================================================================

async def main():
    """メインエントリーポイント"""
    args = sys.argv[1:]

    if "--quick" in args:
        await run_quick_test()
    elif "--type" in args:
        idx = args.index("--type")
        if idx + 1 < len(args):
            await run_all_tests(test_filter=args[idx + 1])
        else:
            print("エラー: --type には引数が必要です（normal, poisson, negative_binomial）")
    else:
        await run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
