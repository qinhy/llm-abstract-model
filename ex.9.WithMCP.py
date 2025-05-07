import os
import json
import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs

store = LLMsStore()

vendor = store.add_new(Model4LLMs.OpenAIVendor)(
                    api_key=os.environ.get('OPENAI_API_KEY','null'))
llm = store.add_new(Model4LLMs.ChatGPT41Nano)(
                    vendor_id=vendor.get_id())


# Server script path
SERVER_SCRIPT = "./tmp/letter_counter.py"
# letter_counter.py
# from fastmcp import FastMCP
# mcp = FastMCP("Letter Counter")
# @mcp.tool()
# def letter_counter(word: str, letter: str) -> int:
#     return word.lower().count(letter.lower())
# if __name__ == "__main__":
#     mcp.run(transport="stdio")

async def get_tools()->str:
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)
    async with Client(transport) as client:
        tools = await client.list_tools()
        return json.dumps([tool.model_dump() for tool in tools])

async def call_tool(tool_name, tool_args)->str:
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)
    async with Client(transport) as client:
        result = await client.call_tool(tool_name, tool_args)
        return json.dumps(result)

ts = asyncio.run(get_tools())
llm.set_mcp_tools(ts)
print(llm.mcp_tools)
print(llm)