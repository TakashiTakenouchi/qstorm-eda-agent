# test_agent.py
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    print("Claude Agent SDK テスト開始...")

    async for message in query(
        prompt="What is 2 + 2?",
        options=ClaudeAgentOptions(
            allowed_tools=["Bash"]
        )
    ):
        print(f"Type: {type(message).__name__}")
        print(message)
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
