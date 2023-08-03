import banana_dev as client

my_model = client.Client(
    api_key="08515c3a-dd0c-4cb9-baa0-7e066d38a93f",
    model_key="f65e007c-a93b-465a-87fc-03a88cb0068b",
    url="https://demo_vicuna_7b-lkvmehhy08.run.banana.dev",
)

#if no temperature or max_new_tokens are specified, the default values "temperature": 0.7, "max_new_tokens": 512 will be used:
#inputs = {
#    '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
#
#    USER: Hello, who are you?
#    ASSISTANT:
#    '''
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

#