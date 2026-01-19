#!/usr/bin/env python3
"""
Q-Storm EDA Distribution Analyzer - Hybrid Streamlit App
=========================================================

確率分布分析ダッシュボード（ハイブリッド版）
- メニュー選択式UI（メイン）
- ルールベースNLP（自然言語入力サポート）
- APIキー不要で動作

使用方法:
    streamlit run streamlit_app.py
"""

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']


# =============================================================================
# 定数定義
# =============================================================================

DEFAULT_DATA_PATH = r"C:\Users\竹之内隆\Documents\MBS_Lessons\MBS2025\Data Set\Ensuring consistency between tabular data and time series forecast data\fixed_extended_store_data_2024-FIX_kaizen_monthlyvol6_new.xlsx"

TARGET_COLUMNS = [
    "Total_Sales", "gross_profit", "discount", "purchasing", "rent",
    "personnel_expenses", "depreciation", "sales_promotion",
    "head_office_expenses", "operating_cost", "Operating_profit",
    "Mens_JACKETS&OUTER2", "Mens_KNIT", "Mens_PANTS",
    "WOMEN'S_JACKETS2", "WOMEN'S_TOPS", "WOMEN'S_ONEPIECE",
    "WOMEN'S_bottoms", "WOMEN'S_SCARF & STOLES",
    "Inventory", "Months_of_inventory", "BEP",
    "Average_Temperature", "Number_of_guests", "Price_per_customer",
]

COLUMN_DEFINITIONS = {
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

# 逆引き辞書（日本語→英語）
COLUMN_REVERSE = {v: k for k, v in COLUMN_DEFINITIONS.items()}


# =============================================================================
# ルールベースNLP
# =============================================================================

class RuleBasedNLP:
    """キーワードマッチングによる自然言語解析"""

    # 意図パターン
    INTENT_PATTERNS = {
        "histogram": ["ヒストグラム", "分布", "ばらつき", "散らばり", "グラフ", "可視化"],
        "analyze": ["分析", "判定", "判別", "特定", "調べ", "確認"],
        "compare": ["比較", "違い", "差", "対比", "vs", "ＶＳ"],
        "normality": ["正規", "正規性", "ガウス", "検定", "テスト"],
        "summary": ["概要", "サマリー", "一覧", "まとめ", "全体"],
    }

    # 店舗パターン
    SHOP_PATTERNS = {
        "恵比寿": ["恵比寿", "えびす", "エビス", "ebisu"],
        "横浜元町": ["横浜", "元町", "よこはま", "ヨコハマ", "yokohama"],
    }

    @classmethod
    def parse_query(cls, query: str) -> dict:
        """自然言語クエリを解析"""
        query_lower = query.lower()
        result = {
            "intent": None,
            "column": None,
            "shop": None,
            "raw_query": query,
            "confidence": 0.0,
        }

        # 意図検出
        for intent, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in query:
                    result["intent"] = intent
                    result["confidence"] += 0.3
                    break
            if result["intent"]:
                break

        # デフォルト意図
        if not result["intent"]:
            result["intent"] = "analyze"
            result["confidence"] += 0.1

        # カラム検出（日本語名）
        for ja_name, en_name in COLUMN_REVERSE.items():
            if ja_name in query:
                result["column"] = en_name
                result["confidence"] += 0.4
                break

        # カラム検出（英語名）
        if not result["column"]:
            for col in TARGET_COLUMNS:
                if col.lower() in query_lower:
                    result["column"] = col
                    result["confidence"] += 0.4
                    break

        # カラム検出（部分一致キーワード）
        if not result["column"]:
            keyword_map = {
                "売上": "Total_Sales",
                "客数": "Number_of_guests",
                "客単価": "Price_per_customer",
                "利益": "Operating_profit",
                "粗利": "gross_profit",
                "在庫": "Inventory",
                "人件費": "personnel_expenses",
                "家賃": "rent",
            }
            for keyword, col in keyword_map.items():
                if keyword in query:
                    result["column"] = col
                    result["confidence"] += 0.3
                    break

        # 店舗検出
        for shop, patterns in cls.SHOP_PATTERNS.items():
            for pattern in patterns:
                if pattern in query or pattern in query_lower:
                    result["shop"] = shop
                    result["confidence"] += 0.2
                    break
            if result["shop"]:
                break

        # 比較意図で両店舗検出
        if result["intent"] == "compare":
            result["shop"] = None  # 比較時は両店舗使用

        result["confidence"] = min(result["confidence"], 1.0)
        return result

    @classmethod
    def get_suggestion(cls, parsed: dict) -> str:
        """解析結果から提案メッセージを生成"""
        intent = parsed["intent"]
        column = parsed["column"]
        shop = parsed["shop"]

        intent_names = {
            "histogram": "ヒストグラム表示",
            "analyze": "分布分析",
            "compare": "店舗間比較",
            "normality": "正規性検定",
            "summary": "全体概要",
        }

        msg = f"📝 解釈結果: **{intent_names.get(intent, intent)}**"
        if column:
            msg += f" / カラム: **{COLUMN_DEFINITIONS.get(column, column)}**"
        if shop:
            msg += f" / 店舗: **{shop}**"
        msg += f" (信頼度: {parsed['confidence']*100:.0f}%)"

        return msg


# =============================================================================
# 統計分析関数
# =============================================================================

def analyze_distribution(data: np.ndarray, column_name: str) -> dict:
    """確率分布を分析"""
    n = len(data)
    mean_val = float(np.mean(data))
    variance = float(np.var(data, ddof=1))
    std_dev = float(np.std(data, ddof=1))
    skewness = float(stats.skew(data))
    kurtosis = float(stats.kurtosis(data))

    dispersion_index = variance / mean_val if mean_val > 0 else float('inf')
    is_discrete = all(float(x).is_integer() for x in data) and data.min() >= 0

    # 正規性検定
    sample_data = data if len(data) <= 5000 else np.random.choice(data, size=5000, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample_data)

    data_standardized = (data - mean_val) / std_dev if std_dev > 0 else data
    ks_stat, ks_p = stats.kstest(data_standardized, 'norm')
    normality_passed = shapiro_p > 0.05 and ks_p > 0.05

    # 分布フィッティング
    fits = {}

    # 正規分布
    mu, sigma = stats.norm.fit(data)
    ll_normal = np.sum(stats.norm.logpdf(data, mu, sigma))
    aic_normal = 4 - 2 * ll_normal
    fits["normal"] = {"parameters": {"mu": round(mu, 2), "sigma": round(sigma, 2)}, "aic": round(aic_normal, 2)}

    # ポアソン分布
    if mean_val > 0 and data.min() >= 0:
        lambda_poisson = mean_val
        data_int = np.maximum(np.round(data).astype(int), 0)
        ll_poisson = np.sum(stats.poisson.logpmf(data_int, lambda_poisson))
        aic_poisson = 2 - 2 * ll_poisson
        fits["poisson"] = {"parameters": {"lambda": round(lambda_poisson, 2)}, "aic": round(aic_poisson, 2)}

    # 負の二項分布
    if mean_val > 0 and variance > mean_val and data.min() >= 0:
        p = mean_val / variance if variance > 0 else 0.5
        p = max(0.001, min(0.999, p))
        r = mean_val * p / (1 - p) if p < 1 else 1.0
        r = max(0.1, r)
        data_int = np.maximum(np.round(data).astype(int), 0)
        ll_nbinom = np.sum(stats.nbinom.logpmf(data_int, r, p))
        aic_nbinom = 4 - 2 * ll_nbinom
        fits["negative_binomial"] = {"parameters": {"r": round(r, 2), "p": round(p, 4)}, "aic": round(aic_nbinom, 2)}

    valid_fits = {k: v for k, v in fits.items() if np.isfinite(v["aic"])}
    best_by_aic = min(valid_fits.items(), key=lambda x: x[1]["aic"])[0] if valid_fits else "unknown"

    # 判定ロジック
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

    return {
        "column": column_name,
        "column_ja": COLUMN_DEFINITIONS.get(column_name, column_name),
        "distribution_type": distribution_type,
        "confidence": round(min(confidence, 1.0), 2),
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


def compare_shops(df: pd.DataFrame, column: str) -> dict:
    """店舗間の分布を比較"""
    shops = df["shop"].unique()
    comparison = {}

    for shop in shops:
        shop_data = df[df["shop"] == shop][column].dropna()
        if len(shop_data) > 0:
            comparison[shop] = {
                "n_samples": len(shop_data),
                "mean": round(float(shop_data.mean()), 2),
                "std": round(float(shop_data.std()), 2),
                "min": round(float(shop_data.min()), 2),
                "max": round(float(shop_data.max()), 2),
                "median": round(float(shop_data.median()), 2)
            }

    test_result = None
    if len(shops) == 2:
        data1 = df[df["shop"] == shops[0]][column].dropna()
        data2 = df[df["shop"] == shops[1]][column].dropna()
        if len(data1) > 0 and len(data2) > 0:
            t_stat, t_p = stats.ttest_ind(data1, data2)
            u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            test_result = {
                "t_test": {"statistic": round(t_stat, 4), "p_value": round(t_p, 4)},
                "mann_whitney_u": {"statistic": round(u_stat, 4), "p_value": round(u_p, 4)},
                "significant_difference": t_p < 0.05 or u_p < 0.05
            }

    return {
        "column": column,
        "column_ja": COLUMN_DEFINITIONS.get(column, column),
        "shops_comparison": comparison,
        "statistical_test": test_result
    }


def create_histogram_figure(data: np.ndarray, column_name: str, shop_filter: str = None) -> plt.Figure:
    """ヒストグラム作成"""
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_val = float(data.mean())
    std_val = float(data.std())

    ax.hist(data, bins=15, edgecolor='black', alpha=0.7, color='#667eea')
    ax.set_xlabel(COLUMN_DEFINITIONS.get(column_name, column_name), fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)

    title = f'{COLUMN_DEFINITIONS.get(column_name, column_name)}'
    if shop_filter:
        title += f' ({shop_filter})'
    ax.set_title(title, fontsize=14)

    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:,.0f}')
    ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: {mean_val-std_val:,.0f}')
    ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: {mean_val+std_val:,.0f}')
    ax.legend()

    plt.tight_layout()
    return fig


def get_distribution_explanation(result: dict) -> str:
    """分布判定結果の解説"""
    dist_type = result["distribution_type"]
    stats_info = result["statistics"]
    evidence = result["evidence"]

    dist_names = {
        "normal": "正規分布 (Normal Distribution)",
        "poisson": "ポアソン分布 (Poisson Distribution)",
        "negative_binomial": "負の二項分布 (Negative Binomial Distribution)"
    }

    explanation = f"""
### 🎯 分布判定結果: {dist_names.get(dist_type, dist_type)}

**信頼度**: {result['confidence'] * 100:.0f}%

#### 判定根拠
"""
    for e in evidence:
        explanation += f"- {e}\n"

    explanation += f"""
#### 📊 基本統計量
| 統計量 | 値 |
|--------|-----|
| サンプル数 | {stats_info['n_samples']:,} |
| 平均 | {stats_info['mean']:,.2f} |
| 標準偏差 | {stats_info['std_dev']:,.2f} |
| 分散 | {stats_info['variance']:,.2f} |
| 歪度 | {stats_info['skewness']:.4f} |
| 尖度 | {stats_info['kurtosis']:.4f} |
| 分散/平均比 | {stats_info['dispersion_index']:.4f} |

#### 🧪 正規性検定
| 検定 | 統計量 | p値 | 結果 |
|------|--------|-----|------|
| Shapiro-Wilk | {result['normality_tests']['shapiro_wilk']['statistic']:.4f} | {result['normality_tests']['shapiro_wilk']['p_value']:.4f} | {'✅ Pass' if result['normality_tests']['shapiro_wilk']['p_value'] > 0.05 else '❌ Fail'} |
| Kolmogorov-Smirnov | {result['normality_tests']['kolmogorov_smirnov']['statistic']:.4f} | {result['normality_tests']['kolmogorov_smirnov']['p_value']:.4f} | {'✅ Pass' if result['normality_tests']['kolmogorov_smirnov']['p_value'] > 0.05 else '❌ Fail'} |
"""

    # ビジネス解釈
    explanation += "\n#### 💼 ビジネス的解釈\n"
    if dist_type == "normal":
        explanation += """
正規分布に従うデータは、平均値を中心に対称的に分布しています。
多くの独立した要因が加法的に影響している場合に見られるパターンです。
- **管理指標**: 平均±2σ（約95%）の範囲で管理
- **異常検知**: 3σを超える値は異常値の可能性
"""
    elif dist_type == "poisson":
        explanation += """
ポアソン分布は、一定期間における発生件数をモデル化するのに適しています。
- **特徴**: 分散と平均がほぼ等しい
- **適用例**: 客数、来店件数、問い合わせ件数
- **予測**: λ（平均発生率）を使って確率計算可能
"""
    elif dist_type == "negative_binomial":
        explanation += """
負の二項分布は、過分散（分散 > 平均）のあるカウントデータに適しています。
- **特徴**: ポアソン分布より裾が重い
- **原因**: 季節変動、店舗差、顧客セグメント差
- **対策**: 層別分析で変動要因を特定することを推奨
"""

    return explanation


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Q-Storm EDA Distribution Analyzer",
        page_icon="📊",
        layout="wide"
    )

    # カスタムCSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .nlp-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
    </style>
    """, unsafe_allow_html=True)

    # ヘッダー
    st.markdown('<div class="main-header">📊 Q-Storm EDA Distribution Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">確率分布分析ダッシュボード - ハイブリッド版（API不要）</div>', unsafe_allow_html=True)

    # サイドバー
    st.sidebar.header("⚙️ 設定")

    # データ読み込み
    uploaded_file = st.sidebar.file_uploader("📁 Excelファイルをアップロード", type=['xlsx', 'xls'])

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"✅ データ読み込み完了: {len(df):,}行")
        except Exception as e:
            st.sidebar.error(f"❌ 読み込みエラー: {e}")
    else:
        if os.path.exists(DEFAULT_DATA_PATH):
            try:
                df = pd.read_excel(DEFAULT_DATA_PATH)
                st.sidebar.info(f"📂 デフォルトデータ: {len(df):,}行")
            except:
                pass

    # サンプルデータ生成
    if df is None:
        st.sidebar.markdown("---")
        if st.sidebar.button("🎲 サンプルデータで試す"):
            np.random.seed(42)
            df = pd.DataFrame({
                "shop": np.repeat(["恵比寿", "横浜元町"], 50),
                "Total_Sales": np.random.normal(5000000, 1000000, 100),
                "Number_of_guests": np.random.poisson(500, 100),
                "Operating_profit": np.random.normal(200000, 100000, 100),
                "gross_profit": np.random.normal(1500000, 300000, 100),
                "Inventory": np.random.normal(2000000, 500000, 100),
                "Price_per_customer": np.random.normal(10000, 2000, 100),
            })
            st.session_state['df'] = df
            st.rerun()

    if 'df' in st.session_state:
        df = st.session_state['df']

    if df is None:
        st.info("👆 サイドバーからExcelファイルをアップロード、または「サンプルデータで試す」をクリックしてください")
        return

    # 利用可能カラム
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available_cols = [c for c in TARGET_COLUMNS if c in numeric_cols]
    if not available_cols:
        available_cols = numeric_cols[:10]

    # メインコンテンツ
    tab1, tab2, tab3 = st.tabs(["💬 自然言語入力", "📋 メニュー選択", "📊 全体概要"])

    # =========================
    # Tab 1: 自然言語入力
    # =========================
    with tab1:
        st.markdown("### 💬 自然言語で質問")
        st.markdown('<div class="nlp-box">', unsafe_allow_html=True)

        query = st.text_input(
            "質問を入力してください",
            placeholder="例: 売上高の分布を見たい / 客数は正規分布？ / 恵比寿と横浜を比較",
            key="nlp_query"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("🔍 分析", type="primary", key="nlp_analyze")

        st.markdown('</div>', unsafe_allow_html=True)

        if query and analyze_btn:
            # NLP解析
            parsed = RuleBasedNLP.parse_query(query)
            st.markdown(RuleBasedNLP.get_suggestion(parsed))

            st.markdown("---")

            # 解析結果に基づく処理
            intent = parsed["intent"]
            column = parsed["column"]
            shop = parsed["shop"]

            if not column and intent != "summary":
                st.warning("⚠️ カラムを特定できませんでした。メニュー選択タブをお使いください。")
            elif intent == "compare":
                if "shop" not in df.columns:
                    st.warning("店舗カラムがありません")
                else:
                    result = compare_shops(df, column)
                    st.markdown(f"### 🏪 {result['column_ja']} の店舗間比較")

                    cols = st.columns(len(result['shops_comparison']))
                    for i, (shop_name, stats_data) in enumerate(result['shops_comparison'].items()):
                        with cols[i]:
                            st.metric(f"🏪 {shop_name}", f"{stats_data['mean']:,.0f}", f"±{stats_data['std']:,.0f}")

                    if result['statistical_test']:
                        test = result['statistical_test']
                        if test['significant_difference']:
                            st.success(f"✅ 統計的に有意な差あり (t検定 p={test['t_test']['p_value']:.4f})")
                        else:
                            st.info(f"ℹ️ 有意な差なし (t検定 p={test['t_test']['p_value']:.4f})")

            elif intent in ["histogram", "analyze", "normality"]:
                # データ取得
                if shop:
                    data = df[df["shop"] == shop][column].dropna().values
                else:
                    data = df[column].dropna().values

                if len(data) > 0:
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        fig = create_histogram_figure(data, column, shop)
                        st.pyplot(fig)
                        plt.close(fig)

                    with col2:
                        result = analyze_distribution(data, column)
                        st.markdown(get_distribution_explanation(result))

            elif intent == "summary":
                st.markdown("### 📋 全カラム概要")
                summary_data = []
                for col in available_cols[:10]:
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        summary_data.append({
                            "カラム": COLUMN_DEFINITIONS.get(col, col),
                            "平均": f"{col_data.mean():,.0f}",
                            "標準偏差": f"{col_data.std():,.0f}",
                            "最小": f"{col_data.min():,.0f}",
                            "最大": f"{col_data.max():,.0f}",
                        })
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # 質問例
        st.markdown("---")
        st.markdown("#### 💡 質問例")
        example_cols = st.columns(3)
        examples = [
            "売上高の分布を見たい",
            "客数は正規分布？",
            "恵比寿と横浜を比較",
            "営業利益のヒストグラム",
            "在庫の統計を教えて",
            "全体の概要を見せて",
        ]
        for i, ex in enumerate(examples):
            with example_cols[i % 3]:
                if st.button(ex, key=f"example_{i}"):
                    st.session_state['nlp_query'] = ex
                    st.rerun()

    # =========================
    # Tab 2: メニュー選択
    # =========================
    with tab2:
        st.markdown("### 📋 メニュー選択式分析")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_column = st.selectbox(
                "📊 分析カラム",
                available_cols,
                format_func=lambda x: f"{COLUMN_DEFINITIONS.get(x, x)} ({x})"
            )

        with col2:
            analysis_type = st.selectbox(
                "🔬 分析タイプ",
                ["ヒストグラム + 分布判定", "店舗間比較", "正規性検定のみ"]
            )

        with col3:
            shop_options = ["全店舗"]
            if "shop" in df.columns:
                shop_options += df["shop"].unique().tolist()
            selected_shop = st.selectbox("🏪 店舗フィルタ", shop_options)
            if selected_shop == "全店舗":
                selected_shop = None

        if st.button("🚀 分析実行", type="primary", key="menu_analyze"):
            if analysis_type == "店舗間比較":
                if "shop" not in df.columns:
                    st.warning("店舗カラムがありません")
                else:
                    result = compare_shops(df, selected_column)
                    st.markdown(f"### 🏪 {result['column_ja']} の店舗間比較")

                    cols = st.columns(len(result['shops_comparison']))
                    for i, (shop_name, stats_data) in enumerate(result['shops_comparison'].items()):
                        with cols[i]:
                            st.markdown(f"#### {shop_name}")
                            st.metric("平均", f"{stats_data['mean']:,.0f}")
                            st.metric("標準偏差", f"{stats_data['std']:,.0f}")
                            st.metric("サンプル数", stats_data['n_samples'])

                    if result['statistical_test']:
                        st.markdown("---")
                        test = result['statistical_test']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**t検定**")
                            st.write(f"統計量: {test['t_test']['statistic']:.4f}")
                            st.write(f"p値: {test['t_test']['p_value']:.4f}")
                        with col2:
                            st.markdown("**Mann-Whitney U検定**")
                            st.write(f"統計量: {test['mann_whitney_u']['statistic']:.4f}")
                            st.write(f"p値: {test['mann_whitney_u']['p_value']:.4f}")

                        if test['significant_difference']:
                            st.success("✅ 店舗間に統計的に有意な差があります")
                        else:
                            st.info("ℹ️ 有意な差は検出されませんでした")
            else:
                # ヒストグラム + 分布判定
                if selected_shop:
                    data = df[df["shop"] == selected_shop][selected_column].dropna().values
                else:
                    data = df[selected_column].dropna().values

                if len(data) > 0:
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        fig = create_histogram_figure(data, selected_column, selected_shop)
                        st.pyplot(fig)
                        plt.close(fig)

                    with col2:
                        result = analyze_distribution(data, selected_column)
                        st.markdown(get_distribution_explanation(result))
                else:
                    st.warning("データがありません")

    # =========================
    # Tab 3: 全体概要
    # =========================
    with tab3:
        st.markdown("### 📊 全カラム統計概要")

        summary_data = []
        for col in available_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                summary_data.append({
                    "カラム名": col,
                    "日本語名": COLUMN_DEFINITIONS.get(col, col),
                    "サンプル数": len(col_data),
                    "平均": f"{col_data.mean():,.2f}",
                    "標準偏差": f"{col_data.std():,.2f}",
                    "最小": f"{col_data.min():,.2f}",
                    "最大": f"{col_data.max():,.2f}",
                    "中央値": f"{col_data.median():,.2f}",
                })

        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, height=400)

        # 相関行列
        if len(available_cols) >= 2:
            st.markdown("### 🔗 相関行列（上位10カラム）")
            corr_cols = available_cols[:10]
            corr_matrix = df[corr_cols].corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr_cols)))
            ax.set_yticks(range(len(corr_cols)))
            ax.set_xticklabels([COLUMN_DEFINITIONS.get(c, c)[:8] for c in corr_cols], rotation=45, ha='right')
            ax.set_yticklabels([COLUMN_DEFINITIONS.get(c, c)[:8] for c in corr_cols])
            plt.colorbar(im)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.85rem;">
        Q-Storm EDA Distribution Analyzer v2.0 |
        Powered by Streamlit + SciPy |
        <b>API Key Not Required</b> ✅
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
