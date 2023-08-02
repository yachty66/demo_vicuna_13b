from potassium import Potassium, Request, Response
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from transformers import pipeline
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    #model = pipeline('text generation', model='TheBloke/vicuna-7B-v1.3-GPTQ', device=device)
    model_name_or_path = "TheBloke/vicuna-7B-v1.3-GPTQ"
    model_basename = "vicuna-7b-v1.3-GPTQ-4bit-128g.no-act.order"
    use_triton = False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,  
        use_safetensors=True,
        trust_remote_code=True,
        device="cpu",
        use_triton=use_triton,
        quantize_config=None)
    #the only thing which needs to work is that here the right model is returned
    context = {
        "model": model,
        "tokenizer": tokenizer
    }
    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    tokenizer = context.get("tokenizer")
    
    # Tokenize the prompt first before inputting it into the model
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    outputs = model(input_ids)
    print("outputs")
    print(outputs)
    return Response(
        json = {"outputs": outputs}, 
        status=200
    )


if __name__ == "__main__":
    app.serve()