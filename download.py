# In this file, we define download_model
# It runs during container build time to get model weights built into the container
# In this example: A Huggingface BERT model
from transformers import pipeline
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def download_model():
    model_name_or_path = "TheBloke/vicuna-7B-v1.3-GPTQ"
    model_basename = "vicuna-7b-v1.3-GPTQ-4bit-128g.no-act.order"
    use_triton = False
    print("downloading model...")
    AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename,  
        use_safetensors=True,
        trust_remote_code=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)   
   
    print("downloading tokenizer...")
    AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

if __name__ == "__main__":
    download_model()