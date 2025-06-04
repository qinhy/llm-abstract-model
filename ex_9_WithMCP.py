import os
import json
import asyncio

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

from LLMAbstractModel import LLMsStore
from LLMAbstractModel.LLMsModel import Model4LLMs

store = LLMsStore()

vendor = store.add_new(
            Model4LLMs.OpenAIVendor)(
                api_key=os.environ.get('OPENAI_API_KEY','null'))
llm:Model4LLMs.AbstractLLM = store.add_new(
            Model4LLMs.ChatGPT41Nano)(
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
        return json.dumps([r.text for r in result][0])
    
async def openai_call_tools(res):
    """Extract and invoke tool calls from the assistant's response."""
    if "calls" not in res: return []

    results = []
    for call in res["calls"]:
        tool_data = call.get(call['type'])
        tool_call_id = call['id']
        if not tool_data: continue
        tool_name = tool_data['name']
        print(f'calling func: {tool_name}')
        tool_args = json.loads(tool_data['arguments'])
        results.append({'role': 'tool','name': tool_name,
            'tool_call_id':tool_call_id})
        try:
            result = await call_tool(tool_name, tool_args)
            results[-1]['content'] = result
        except Exception as e:
            results[-1]['content'] = f"Error calling tool: {str(e)}"
    return results

def one_query(ask:str='How many "r" in "raspberrypi"?',llm=llm):
    
    # Initialize structured messages
    messages = [{"role": "user", "content": ask}]
    # First assistant response (might suggest a tool)
    res = llm(messages)
    # Handle tool calls
    if "calls" not in res: return res
    # Insert assistant message with tool_calls
    messages.append({
        "role": "assistant",
            # might be empty or partial
        "content": res.get("content", ""),
        "tool_calls": res["calls"]
    })
    # Run the tool(s)
    tool_results = asyncio.run(openai_call_tools(res))
    # For each tool call, insert a 'tool' message
    for i, tr in enumerate(tool_results):
        messages.append({
            "role": "tool",
            "tool_call_id": res["calls"][i]["id"],  # MUST match
            "name": tr["name"],"content": tr["content"]
        })
    # Now ask assistant for final response
    messages.append({
        "role": "user",
        "content": "Please give me the final answer."
    })
    return llm(messages)

# --- Main Conversation Flow ---
llm.set_mcp_tools(asyncio.run(get_tools()))
print(llm('tell me your tools.'))
# data = store.dumps()
# del store
# store = LLMsStore()
# store.loads(data)
# llm = store.find_all('ChatGPT41Nano*')[0]
print(one_query('How many "r" in "raspberrypi"?',llm))

# ### advance with History in ex.8.TryAgentsAndHistory.py
from ex_8_TryAgentsAndHistory import HistoryAssistantAgent
agent = HistoryAssistantAgent(llm=llm)
agent.set_mcp_tools(
            asyncio.run(get_tools()),
            lambda rs:asyncio.run(openai_call_tools(rs))
    )
print(agent('How many "r" in "raspberrypi"?',auto_tool=True))
# agent("Please give me the final answer.")