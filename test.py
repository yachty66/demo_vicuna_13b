# pip3 install banana-dev

import banana_dev as client

# Create a reference to your model on Banana
my_model = client.Client(
    api_key="08515c3a-dd0c-4cb9-baa0-7e066d38a93f",
    model_key="cfd2d153-e095-4f27-9ea5-4bd6d6da2227",
    url="https://demo_vicuna_13b-lkuy40zc08.run.banana.dev",
)

# Specify the model's input JSON, what you expect 
# to receive in your Potassium app. Here is an 
# example for a basic BERT model:
inputs = {
    "prompt": "In the summer I like [MASK].",
}

# Call your model's inference endpoint on Banana.
# If you have set up your Potassium app with a
# non-default endpoint, change the first 
# method argument ("/")to specify a 
# different route.
result, meta = my_model.call("/", inputs)

print(result)
