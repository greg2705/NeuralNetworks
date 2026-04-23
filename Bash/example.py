import json
from openai import OpenAI

client = OpenAI()

# 1) Your local Python tool
def get_repo_file(path: str) -> str:
    fake_files = {
        "README.md": "# Demo repo\nThis is a sample file.",
        "tool_schema.json": '{"name":"get_repo_file"}',
    }
    return fake_files.get(path, f"File not found: {path}")

# 2) Your tool schema
tools = [
    {
        "type": "function",
        "name": "get_repo_file",
        "description": "Read a file from the local repository by path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path, for example README.md"
                }
            },
            "required": ["path"],
            "additionalProperties": False
        }
    }
]

# 3) Initial user request
messages = [
    {
        "role": "user",
        "content": "Read README.md and tell me what this repo is about."
    }
]

# 4) Ask the model; it may choose to call your tool
response = client.responses.create(
    model="gpt-4.1-mini",
    input=messages,
    tools=tools,
)

# 5) Append model output and execute tool calls
messages += response.output

for item in response.output:
    if item.type == "function_call" and item.name == "get_repo_file":
        args = json.loads(item.arguments)
        result = get_repo_file(args["path"])

        messages.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": result
        })

# 6) Ask the model again so it can use the tool result
final_response = client.responses.create(
    model="gpt-4.1-mini",
    input=messages,
    tools=tools,
)

print(final_response.output_text)
