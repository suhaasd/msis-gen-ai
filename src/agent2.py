import json
import os
import subprocess
from openai import OpenAI

"""
activate the virtual environment:
$ cd source ~/teaching/nov2025
$ source ./bin/activate

run this code:
$ python3.14 agent1.py

"""


def client():
    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY", "<your OpenAI API key environment variable>"
        )
    )
    return client


def llm_run_with_tools(cl, context, tools):
    models = ["gpt-4o", "o4-mini", "gpt-4.1", "gpt-5-nano", "GPT-5 mini", "gpt-5"]
    return cl.responses.create(model=models[1], tools=tools, input=context)


def llm_call_with_context_tools(cl, context, tools, prompt):
    context.append({"role": "user", "content": prompt})
    response = llm_run_with_tools(cl, context, tools)
    context.append({"role": "assistant", "content": response.output_text})
    return response.output_text


## https://platform.openai.com/docs/guides/function-calling
tools = [
    {
        "type": "function",
        "name": "ping_api_host",
        "description": "ping some host on the internet",
        "parameters": {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "hostname or IP",
                },
            },
            "required": ["server"],
        },
    },
    {
        "type": "function",
        "name": "authorize_user",
        "description": "authorize a user to access a protected resource",
        "parameters": {
            "type": "object",
            "properties": {
                "who": {
                    "type": "string",
                    "description": "name of the user accessing a protected resource",
                },
                "what": {
                    "type": "string",
                    "description": "the protected resource being accessed",
                },
            },
            "required": ["who", "what"],
        },
    },
]


def authorize_user(who="", what=""):
    return f"User '{who}' is authorized to access resource '{what}'."


def tool_call_authorize_user(item):  # handles one tool call
    result = authorize_user(**json.loads(item.arguments))
    return [
        item,
        {"type": "function_call_output", "call_id": item.call_id, "output": result},
    ]


def ping_api_host(server=""):
    try:
        print(f"ping_api_host: trying {server} ...")
        result = subprocess.run(
            ["ping", "-c", "5", server],
            text=True,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
        )
        return result.stdout
    except Exception as e:
        return f"error: {e}"


def tool_call_ping_api_host(item):  # handles one tool call
    result = ping_api_host(**json.loads(item.arguments))
    return [
        item,
        {"type": "function_call_output", "call_id": item.call_id, "output": result},
    ]


def process_tool_calls(context, llm_resp_output):
    # https://platform.openai.com/docs/api-reference/responses/object#responses-object-output-reasoning
    # https://platform.openai.com/docs/api-reference/responses/object#responses-object-output-reasoning-type
    if llm_resp_output[0].type == "reasoning":
        context.append(llm_resp_output[0])
    ctx_len = len(context)
    for item in llm_resp_output:
        # https://platform.openai.com/docs/api-reference/responses/object#responses-object-output-function_tool_call

        if item.type == "function_call":
            print(f"process_tool_calls: invoking tool with item {item} ...")
            context.extend(tool_call_authorize_user(item))
            # context.extend(tool_call_ping_api_host(item))
    return len(context) != ctx_len


def llm_call_with_context_tools(cl, context, tools, prompt):
    # update context with user prompt and ask LLM
    context.append({"role": "user", "content": prompt})
    response = llm_run_with_tools(cl, context, tools)
    # did the model request any tool calls?
    while process_tool_calls(context, response.output):
        response = llm_run_with_tools(cl, context, tools)
    context.append({"role": "assistant", "content": response.output_text})
    return response.output_text


def run_agent_with_tool(cl):
    prompt_api_health = """
    Describe our connectivity to the API server web3pleb.org.
    Use different servers.
    Check if the servers are running and available.
    Produce output in formatted JSON.
    The JSON should include only two fields: 'result', 'reachable' and 'summary' in that order.
    The 'result' field stores a boolean value representing the success or failure to reach the endpoint.
    The reachable endpoints must be stored under 'reachable' key.
    The 'summary' field stores a brief text description of the connectivity status.
    """

    prompt = """
    Bob wants to access the critical resource 'PhotoPrint001'.
    """

    text = llm_call_with_context_tools(cl, [], tools, prompt)
    print(text)


if __name__ == "__main__":
    cl = client()
    run_agent_with_tool(cl)
