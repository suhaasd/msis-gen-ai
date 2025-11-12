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
    return cl.responses.create(model=models[5], tools=tools, input=context)


def llm_call_with_context_tools(cl, context, tools, prompt):
    context.append({"role": "user", "content": prompt})
    response = llm_run_with_tools(cl, context, tools)
    context.append({"role": "assistant", "content": response.output_text})
    return response.output_text


## https://platform.openai.com/docs/guides/function-calling
tools = [
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
    if who.lower() in ["bob", "charlie"]:
        return f"User '{who}' is authorized to access resource '{what}'."
    return f"User '{who}' is NOT authorized to access resource '{what}'."


def tool_call_authorize_user(item):  # handles one tool call
    result = authorize_user(**json.loads(item.arguments))
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
    # update context with system and user prompts, and ask the LLM.
    context.append(
        {
            "role": "system",
            "content": """
                    You are a manager in the office.
                    You are responsible for ensuring that resources are properly utilized for right purposes.
                    A critical resource is a resource that is expensive or limited in availability.
                    A critical resource usually bears a name tag like 'ScannerX' or 'SecureServerY'.
                    Only when a resource is critical, should you call the tool 'authorize_user' to authorize access.
                    If you conclude that a resource is not critical, you should not call the tool.
                    Always respond in a concise manner.
                    Produce output in formatted JSON.
                    The JSON MUST ALWAYS include these FOUR fields: 'result', 'who', 'what', and 'auth_time' in that order.
                    The 'who' and 'what' fields store the name of the user and the resource respectively.
                    The 'what' field should not include quantity of allocated items. It should only include the resource name.
                    The 'auth_time' field MUST store the time when the authorization decision is made, as a string value.
                    The 'auth_time' is in the format 'YYYY-MM-DD HH:MM:SS'.
                    For a critical resource, the JSON response object MUST be as follows:
                        The 'result' field SHOULD store a boolean value indicating if the request was successfully fulfilled or not.
                        When the authorization is successful,
                            The 'start_time' field MUST indicate the time from when the access is authorized, as a string value.
                            The 'start_time' field should be in the format 'YYYY-MM-DD HH:MM:SS'.
                            The 'start_time' indicates a time after the authorization decision is made.
                            In other words, the 'start_time' value MUST BE greater than the 'auth_time' value.
                            The 'duration' field MUST store how many minutes the access is authorized for, as a string value.
                            The 'duration' value ranges from '5' to '45'.
                        When 'result' is false, that is, when authorization fails, the 'duration' field MUST BE set to "0".
                    For a non-critical resource, the JSON response object MUST be as follows:
                        The 'result' field must be true.
                        The 'who' and 'what' fields must store the name of the user and the resource respectively.
                        The 'auth_time' field MUST store the time when the authorization decision is made, as a string value.
                        The 'start_time' field MUST NOT BE present.
                        The 'duration' field MUST NOT be present.
                    When a request is not fulfilled, provide a brief reason in the 'summary' field.
                    """,
        }
    )
    context.append({"role": "user", "content": prompt})
    response = llm_run_with_tools(cl, context, tools)
    # did the model request any tool calls?
    while process_tool_calls(context, response.output):
        response = llm_run_with_tools(cl, context, tools)
    context.append({"role": "assistant", "content": response.output_text})
    return response.output_text


# This detects that Bob and Charlie cannot access the same resource at the same time.\
# without being explicitly told about it.


def run_agent_with_tool(cl):
    resource_requests = """
    Bob wants to use the photocopier machine.
    Bob also wants to use the photo printer 'PhotoPrint_01' after two days, at 11 am, for 18 minutes.
    Bob wants to use the dust bin.
    Bob wants a dozen paper clips and a couple sharpies.
    Charlie wants to use the photo printer 'PhotoPrint_01' after two days, at 11.10 am, for 5 minutes.
    Alice also wants to use the photo printer 'PhotoPrint_02'.
    """

    text = llm_call_with_context_tools(cl, [], tools, resource_requests)
    print(text)


if __name__ == "__main__":
    cl = client()
    run_agent_with_tool(cl)
