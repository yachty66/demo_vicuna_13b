# pip3 install banana-dev

import banana_dev as client

# Create a reference to your model on Banana
my_model = client.Client(
    api_key="08515c3a-dd0c-4cb9-baa0-7e066d38a93f",
    model_key="51c122f1-a0ab-4530-bc7a-0848fe3e2fa8",
    url="https://demo_vicuna_13b-lkv21kyo08.run.banana.dev",
)

# Specify the model's input JSON, what you expect 
# to receive in your Potassium app. Here is an 
# example for a basic BERT model:
inputs = {
    "prompt": '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

    USER: Hello, who are you?
    ASSISTANT:
    '''
}

# Call your model's inference endpoint on Banana.
# If you have set up your Potassium app with a
# non-default endpoint, change the first 
# method argument ("/")to specify a 
# different route.
result, meta = my_model.call("/", inputs)

print(result)
