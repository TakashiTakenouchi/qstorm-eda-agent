import anyio
from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.types import (
    SystemMessage,
    AssistantMessage,
    ResultMessage,
    TextBlock
)

async def main():
    async for message in query(
        prompt="List Python files in current directory",
        options=ClaudeAgentOptions(
            allowed_tools=["Bash", "Glob", "Read"],
            max_turns=5
        )
    ):
        # メッセージタイプ別の処理
        if isinstance(message, SystemMessage):
            print(f"Session started: {message.data.get('session_id', 'N/A')}")

        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")

        elif isinstance(message, ResultMessage):
            print(f"Cost: ${message.total_cost_usd:.4f}")
            print(f"Duration: {message.duration_ms}ms")

anyio.run(main)
