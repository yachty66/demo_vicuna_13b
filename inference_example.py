import banana_dev as client

my_model = client.Client(
    api_key="08515c3a-dd0c-4cb9-baa0-7e066d38a93f",
    model_key="51c122f1-a0ab-4530-bc7a-0848fe3e2fa8",
    url="https://demo_vicuna_13b-lkv21kyo08.run.banana.dev",
)

#if no temperature or max_new_tokens are specified, the default values "temperature": 0.7, "max_new_tokens": 512 will be used:
#inputs = {
#    "prompt": "In the summer I like [MASK]."
#}
inputs = {
    "prompt": '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

    USER: Hello, who are you?
    ASSISTANT:
    ''',
    "temperature": 0.7,
    "max_new_tokens": 512
}

result, meta = my_model.call("/", inputs)

print(result)