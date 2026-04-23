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

message = response.choices[0].message

if message.tool_calls:
    messages.append(message)

    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if name == "get_repo_file":
            result = get_repo_file(args["path"])
        else:
            result = f"Unknown tool: {name}"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    print(final_response.choices[0].message.content)
else:
    print(message.content)

print(final_response.output_text)
