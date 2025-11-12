import os
from openai import OpenAI

"""
activate the virtual environment:
$ cd source ~/teaching/nov2025
$ source ./bin/activate

run this code:
$ python3.14 agent0.py

"""

"""
OpenAI Documentation: https://platform.openai.com/docs/api-reference/introduction
"""


def client():
    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY", "<your OpenAI API key environment variable>"
        )
    )
    return client


def llm_run(cl, context):
    models = ["gpt-4o", "o4-mini", "gpt-4.1", "gpt-5-nano", "GPT-5 mini", "gpt-5"]
    return cl.responses.create(model=models[1], input=context)


def llm_call_with_context(cl, context, prompt):
    context.append({"role": "user", "content": prompt})
    response = llm_run(cl, context)
    context.append({"role": "assistant", "content": response.output_text})
    return response.output_text


def test_empty_context(cl):
    user_ask = [
        "what is the value of the expression 100 * 12",
        "<q>HDFC</q><a>BANK</a>  <q>IIT</q><a>EDUCATION</a>  <q>FBI</q><a>?</a>",
        "if base of a right-angled triangle is 3 and height is 4, what is the hypotenuse?",
        "which one of my questions so far is about mathematics?",
    ]

    for prompt in user_ask:
        output = llm_call_with_context(
            cl,
            [],
            prompt,
        )
        print("User:", prompt)
        print("Agent:", output)
        print("-----")


def test_context(cl):
    user_ask = [
        "what is the value of the expression 100 * 12",
        "<q>HDFC</q><a>BANK</a>  <q>IIT</q><a>EDUCATION</a>  <q>FBI</q><a>?</a>",
        "if base of a right-angled triangle is 3 and height is 4, what is the hypotenuse?",
        "which one of my questions so far is about mathematics?",
    ]

    context = []
    for prompt in user_ask:
        output = llm_call_with_context(
            cl,
            context,
            prompt,
        )
        print("User:", prompt)
        print("Agent:", output)
        print("-----")


def run_interactive_turn(cl, context):
    while True:
        prompt = input("User: ")
        if prompt.lower() in {"exit", "quit"}:
            break
        output = llm_call_with_context(
            cl,
            context,
            prompt,
        )
        print("Agent:", output)


if __name__ == "__main__":
    cl = client()
    # test_empty_context(cl)
    test_context(cl)
