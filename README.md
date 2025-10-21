# LLMAbstractModel

Vendor-agnostic tooling for building language-model workflows with a single, composable API. `LLMAbstractModel` helps you manage providers such as OpenAI, Anthropic, DeepSeek, Google Gemini, xAI Grok, and local Ollama models while chaining calls, utilities, and custom logic in Python.

## Highlights
- Orchestrate multiple vendors and models through a shared `LLMsStore`
- Switch models without rewriting prompt logic by using vendor-agnostic controllers
- Compose rich inference flows with the Mermaid workflow engine and reusable functions
- Manage embeddings, content objects, and auxiliary utilities (regex extractors, templating, file chunking, base64 helpers)
- Explore ready-to-run examples covering prompts, chains, agents, tools, and memory

## Requirements
- Python 3.11+
- API keys for the vendors you intend to use (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`)
- Optional: an [Ollama](https://ollama.com/) runtime for local models

## Installation

Clone the repository and install in a virtual environment. The project includes a `pyproject.toml` and `uv.lock`, so you can choose between `uv` or `pip`.

```bash
git clone https://github.com/qinhy/llm-abstract-model.git
cd llm-abstract-model

# using uv (recommended)
uv sync

# or using pip
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

## Quickstart

```python
import os
from LLMAbstractModel import LLMsStore, Model4LLMs

store = LLMsStore()

# Register the vendor and an OpenAI model
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Set OPENAI_API_KEY before running this example.")

openai_vendor = store.add_new_vendor(Model4LLMs.OpenAIVendor)(api_key=api_key)
assistant = store.add_new_llm(Model4LLMs.ChatGPT4oMini)(
    vendor_id=openai_vendor.get_id(),
    system_prompt="You are a concise assistant."
)

print(assistant("Give me two haiku ideas about espresso."))
```

The `vendor_id` can be set to `"auto"` for many built-in models. For example:

```python
assistant = store.add_new_llm(Model4LLMs.ChatGPT4oMini)(
    vendor_id="auto"  # maps to the first OpenAI vendor in the store
)
```

## Vendor and Model Coverage

| Vendor                        | Examples                                                     | Notes                                                      |
|------------------------------|---------------------------------------------------------------|------------------------------------------------------------|
| OpenAI                       | `ChatGPT4o`, `ChatGPT5`, `ChatGPTO3`, `TextEmbedding3Small`   | Tool-call ready; supports reasoning variants               |
| Anthropic                    | `Claude35`, `Claude37`, `ClaudeDynamic`                       | Uses Claude Messages API                                   |
| DeepSeek (Anthropic protocol)| `DeepSeek`, `DeepSeekReasoner`, `DeepSeekDynamic`             | Shares Claude-compatible interface                         |
| Google Gemini                | `Gemini25Pro`, `Gemini25Flash`, `TextGeminiEmbedding001`      | Handles text, tool calls, and embeddings                   |
| xAI                          | `Grok`                                                        | Standard OpenAI-style chat payloads                        |
| Ollama                       | `Gemma2`, `Phi3`, `Llama`                                     | Requires local Ollama runtime                              |

Every model inherits from a common controller that normalizes payload construction, handles retries, and exposes helper methods such as `construct_messages`, `construct_payload`, and `set_mcp_tools`.

## Workflows with Mermaid

`MermaidWorkflowEngine` lets you express chains of functions (LLM calls, utilities, custom Python) using Mermaid diagrams. The engine parses the graph, validates dependencies, and executes nodes topologically.

```python
from LLMAbstractModel import (
    LLMsStore,
    MermaidWorkflowEngine,
    Model4LLMs,
    StringTemplate,
)

store = LLMsStore()
prompt = store.add_new_function(
    StringTemplate(para={"string": "Summarise {topic} in exactly {sentences} sentences."}).build()
)
summariser = store.add_new_llm(Model4LLMs.ChatGPT4oMini)(vendor_id="auto")

mermaid = """
graph TD
    Prompt
    LLM

    Prompt -- "{'data':'prompt'}" --> LLM
"""

engine = MermaidWorkflowEngine(store=store, mermaid_text=mermaid)
engine.registry.update({"Prompt": prompt, "LLM": summariser})

result = engine.run(topic="Mermaid workflows", sentences=3)
print(result)
```

The workflow engine supports argument mapping, custom validation, and exporting/importing node definitions as JSON.

## Utilities and Building Blocks
- `RegxExtractor`: extract structured data with regex (optionally returning JSON)
- `StringTemplate`: format reusable string templates (supports dynamic call generation)
- `TextFile`: lazy reader that yields overlapping chunks from large documents
- `StringToBase64Encoder` / `Base64ToStringDecoder`: ready-to-use transformation nodes

All utility classes register themselves with the global store so they can participate in workflows.

## Repository Guide
- `LLMAbstractModel/BasicModel.py`: foundation for model controllers, storage, and Pydantic objects
- `LLMAbstractModel/LLMsModel.py`: vendor, model, embedding, workflow, and storage implementations
- `LLMAbstractModel/MermaidWorkflowEngine.py`: Mermaid parsing and execution engine
- `LLMAbstractModel/utils.py`: templating, extraction, file helpers, and workflow functions
- `ex_*` scripts: cookbook of chains, agents, MCP tools, memory, and workflow scenarios
- `tests/`: unit tests covering vendor and model abstractions

## Running the Examples

Each script in the repo shows a focused scenario:
- `ex_1_1_HowToUse.py`: minimal vendor + model setup
- `ex_2_*`: building multi-step chains over text and code
- `ex_5_*`: defining custom workflows and complex branching
- `ex_6_*` to `ex_8_*`: integrating agents, self-iteration, and memory
- `ex_9_WithMCP.py`: wiring Model Context Protocol tools

Activate your environment, export the necessary API keys, and run any example:

```bash
python ex_5_2_ComplexWorkflow.py
```

## Development

Install development dependencies and run the test suite:

```bash
uv sync --group dev  # or: pip install pytest google-genai ipython openai
pytest
```

Pull requests and issue reports are welcome. If you add support for a new vendor or workflow node, include unit tests and an example where possible.

## License

This project is released under the [MIT License](LICENSE).
