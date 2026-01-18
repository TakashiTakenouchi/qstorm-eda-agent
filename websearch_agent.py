#!/usr/bin/env python3
"""
WebSearch Agent - 自然言語でWeb検索を実行するClaude Agent SDK
==============================================================

使用方法:
    uv run python websearch_agent.py                    # インタラクティブモード
    uv run python websearch_agent.py "検索クエリ"       # 単発クエリモード

例:
    uv run python websearch_agent.py "2025年のAI技術トレンドを教えて"
    uv run python websearch_agent.py "Python 3.13の新機能は？"
"""

import asyncio
import sys
from datetime import datetime
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk.types import (
    SystemMessage,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)


class WebSearchAgent:
    """自然言語でWeb検索を実行するエージェント"""

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: True の場合、詳細なログを出力
        """
        self.verbose = verbose
        self.options = ClaudeAgentOptions(
            allowed_tools=["WebSearch", "WebFetch"],
            permission_mode="bypassPermissions",
            max_turns=10,
        )
        self.total_cost = 0.0
        self.query_count = 0

    async def search(self, query: str) -> str:
        """
        自然言語クエリでWeb検索を実行

        Args:
            query: 自然言語の検索クエリ

        Returns:
            検索結果のテキスト
        """
        results = []
        
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(query)

            async for message in client.receive_response():
                if isinstance(message, SystemMessage):
                    if self.verbose:
                        print(f"[System] Session: {message.session_id[:8]}...")

                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            results.append(block.text)
                        elif isinstance(block, ToolUseBlock) and self.verbose:
                            print(f"[Tool] {block.name}: {block.input}")

                elif isinstance(message, ResultMessage):
                    self.total_cost += message.total_cost_usd
                    self.query_count += 1
                    if self.verbose:
                        print(f"[Info] Cost: ${message.total_cost_usd:.4f}")

        return "\n".join(results)

    def print_stats(self):
        """累計統計を表示"""
        print(f"\n{'='*50}")
        print(f"セッション統計")
        print(f"  クエリ数: {self.query_count}")
        print(f"  累計コスト: ${self.total_cost:.4f}")
        print(f"{'='*50}")


async def interactive_mode(agent: WebSearchAgent):
    """インタラクティブモード: ユーザーからの入力を繰り返し受け付ける"""
    print("=" * 60)
    print("  WebSearch Agent - 自然言語Web検索")
    print("  Claude Agent SDK powered")
    print("=" * 60)
    print()
    print("検索したい内容を自然言語で入力してください。")
    print("終了: 'exit' または 'quit' または Ctrl+C")
    print("-" * 60)

    while True:
        try:
            print()
            query = input("🔍 検索> ").strip()

            if not query:
                continue

            if query.lower() in ("exit", "quit", "q", "終了"):
                agent.print_stats()
                print("\nさようなら！")
                break

            print()
            print("検索中...")
            print("-" * 40)

            result = await agent.search(query)
            print(result)
            print("-" * 40)
            print(f"[コスト: ${agent.total_cost:.4f} (累計)]")

        except KeyboardInterrupt:
            agent.print_stats()
            print("\n\n中断されました。")
            break
        except Exception as e:
            print(f"\n[エラー] {type(e).__name__}: {e}")


async def single_query_mode(agent: WebSearchAgent, query: str):
    """単発クエリモード: 引数で渡されたクエリを実行"""
    print(f"検索: {query}")
    print("=" * 60)
    print()

    result = await agent.search(query)
    print(result)

    print()
    print("-" * 60)
    print(f"コスト: ${agent.total_cost:.4f}")


async def main():
    """メインエントリーポイント"""
    # 詳細ログを表示するかどうか（環境変数 VERBOSE=1 で有効）
    import os
    verbose = os.environ.get("VERBOSE", "0") == "1"

    agent = WebSearchAgent(verbose=verbose)

    if len(sys.argv) > 1:
        # コマンドライン引数がある場合は単発クエリモード
        query = " ".join(sys.argv[1:])
        await single_query_mode(agent, query)
    else:
        # 引数がない場合はインタラクティブモード
        await interactive_mode(agent)


if __name__ == "__main__":
    asyncio.run(main())
