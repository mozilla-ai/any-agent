import asyncio
import os
from any_agent import AnyAgent, AgentConfig
from any_agent.config import MCPStdio, MCPStreamableHttp, MCPSse


async def main():
    print("Starting agent")
    config = AgentConfig(
        model_id="openai/gpt-5-mini",
        tools=[
            MCPStreamableHttp(
                url="https://api.githubcopilot.com/mcp/",
                headers={"Authorization": f"Bearer {os.environ['GITHUB_PAT']}"},
                client_session_timeout_seconds=10,
            ),
        ],
        model_args={"temperature": 1},
    )
    print("Creating agent")
    agent = await AnyAgent.create_async("tinyagent", agent_config=config)
    print("Running agent")
    result = agent.run("What is my github username?")
    print("Result")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())