from potassium import Potassium, Request, Response
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from transformers import pipeline
import torch

app = Potassium("my_app")

#i need to run the fucking model on inference. on banana. how are we going to do this here

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    #device = 0 if torch.cuda.is_available() else -1
    #model = pipeline('text generation', model='TheBloke/vicuna-7B-v1.3-GPTQ', device=device)
    model_name_or_path = "TheBloke/vicuna-7B-v1.3-GPTQ"
    model_basename = "vicuna-7b-v1.3-GPTQ-4bit-128g.no-act.order"
    use_triton = False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,  
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
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
    
    
    #prompt = event['input']['prompt']
    prompt_template=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

    USER: Hello, who are you?
    ASSISTANT:
    '''
    
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
    result = tokenizer.decode(output[0])
    
    return Response(
        json = {"outputs": result}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()