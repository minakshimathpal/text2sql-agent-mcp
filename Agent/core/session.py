# core/session.py

import os
import sys
from typing import Optional, Any, List, Dict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import asyncio
import traceback


class MCP:
    """
    Lightweight wrapper for one-time MCP tool calls using stdio transport.
    Each call spins up a new subprocess and terminates cleanly.
    """

    def __init__(
        self,
        server_script: str = "mcp_server_2.py",
        working_dir: Optional[str] = None,
        server_command: Optional[str] = None,
    ):
        self.server_script = server_script
        self.working_dir = working_dir or os.getcwd()
        self.server_command = server_command or sys.executable

    async def list_tools(self):
        server_params = StdioServerParameters(
            command=self.server_command,
            args=[self.server_script],
            cwd=self.working_dir
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return tools_result.tools

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        server_params = StdioServerParameters(
            command=self.server_command,
            args=[self.server_script],
            cwd=self.working_dir
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments=arguments)


class MultiMCP:
    """
    Stateless version: discovers tools from multiple MCP servers, but reconnects per tool call.
    Each call_tool() uses a fresh session based on tool-to-server mapping.
    """

    def __init__(self, server_configs: List[dict]):
        self.server_configs = server_configs
        self.tool_map: Dict[str, Dict[str, Any]] = {}  # tool_name → {config, tool}
        self.sse_connections = {}

    async def initialize_sse_server(self, server_url: str, config):
        """
        Connects to an MCP server using SSE.
        - Opens an SSE connection via sse_client.
        - Creates and initializes an MCP ClientSession.
        - Retrieves the available tools and converts them for Gemini.
        """
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()
        self._session_context = ClientSession(*streams)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()
        print("Initialized SSE client...")
        print("Listing SSE server tools...")
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # Store the connection info for this server
        self.sse_connections[server_url] = {
            'streams_context': self._streams_context,
            'session_context': self._session_context,
            'session': self.session,
            'config': config
        }

        for tool in tools:
            self.tool_map[tool.name] = {
                                        "config": config,
                                        "tool": tool
                                    }
        return 
    
    async def cleanup(self):
        """Properly clean up all SSE connections"""
        for server_url, connection_info in self.sse_connections.items():
            try:
                if 'session_context' in connection_info:
                    await connection_info['session_context'].__aexit__(None, None, None)
                if 'streams_context' in connection_info:
                    await connection_info['streams_context'].__aexit__(None, None, None)
                print(f"✅ Cleaned up SSE connection to {server_url}")
            except Exception as e:
                print(f"⚠️ Error cleaning up SSE connection to {server_url}: {e}")
        
        self.sse_connections.clear()

    async def initialize(self):
        for config in self.server_configs:
            
            if config["type"] == "stdio":
                try:
                    params = StdioServerParameters(
                        command=sys.executable,
                        args=[config["script"]],
                        cwd=config.get("cwd", os.getcwd())
                    )
                    print(f"→ Scanning tools from: {config['script']} in {params.cwd}")
                    async with stdio_client(params) as (read, write):
                        print("Connection established, creating session...")
                        try:
                            async with ClientSession(read, write) as session:
                                print("[agent] Session created, initializing...")
                                await session.initialize()
                                print("[agent] MCP session initialized")
                                tools = await session.list_tools()
                                print(f"→ Tools received: {[tool.name for tool in tools.tools]}")
                                for tool in tools.tools:
                                    self.tool_map[tool.name] = {
                                        "config": config,
                                        "tool": tool
                                    }
                        except Exception as se:
                            print(f"❌ Session error: {se}")
                except Exception as e:
                    print(f"❌ Error initializing MCP server {config['script']}: {e}")

            else:
                try:
                    await self.initialize_sse_server(server_url=config["url"], config=config)
                except Exception as e:
                    print(f"❌ Error initializing MCP server {config['script']}: {e}")
                    traceback.print_exception(type(e), e, e.__traceback__)

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        entry = self.tool_map.get(tool_name)
        
        if not entry:
            raise ValueError(f"Tool '{tool_name}' not found on any server.")
        

        config = entry["config"]
        if config["type"] == "stdio":
            params = StdioServerParameters(
                command=sys.executable,
                args=[config["script"]],
                cwd=config.get("cwd", os.getcwd())
            )

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(tool_name, arguments)
        
        else:
            return await self.session.call_tool(tool_name, arguments)

    async def list_all_tools(self) -> List[str]:
        return list(self.tool_map.keys())

    def get_all_tools(self) -> List[Any]:
        return [entry["tool"] for entry in self.tool_map.values()]

async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run client.py <URL of SSE MCP server (i.e. http://localhost:8080/sse)>")
        sys.exit(1)

    client = MultiMCP()
    
    try:
        await client.connect_to_sse_server(server_url=sys.argv[1])
        await client.chat_loop()
    
    except Exception as e:
        print(traceback.format_exc())



if __name__ == "__main__":
    import sys
    asyncio.run(main())