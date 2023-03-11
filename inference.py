import argparse
import os
import sys
from fastgpt import CausalLMModelForOnnxGeneration

file_dir = os.path.dirname(os.path.abspath(__file__))
codegen_dir = os.path.join(file_dir, "CodeGen")
sys.path.append(codegen_dir)
from jaxformer.hf.sample import create_custom_gpt2_tokenizer
from jaxformer.hf.codegen.configuration_codegen import CodeGenConfig

"""
    A function to get the tokenizer for a given CodeGen model.
"""
def get_codegen_model_tokenizer(model_name="codegen-350M-mono", threads=1):
    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = "left"
    pad = 50256
    tokenizer.pad_token = pad
    model_path = os.path.join("checkpoints", model_name)
    onnx_model_path = os.path.join(model_path, "onnx/model-quantized.onnx")
    config = CodeGenConfig.from_pretrained(model_path)
    model = CausalLMModelForOnnxGeneration(onnx_model_path, model_path, config, threads)
    return model, tokenizer

"""
    Generates Python code starting from a given prompt in natural language (English).
"""
def generate_code(args):
    model, tokenizer = get_codegen_model_tokenizer()
    input_ids = tokenizer(
            args.prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids
    generated_ids = model.generate(
            input_ids,
            max_length=args.max_lenght + input_ids.shape[1],
            decoder_start_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
            output_scores=True,
            temperature=args.temperature,
            repetition_penalty=1.0,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
            length_penalty=2.0,
            early_stopping=True,
        )
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", 
                        type=str,
                        required=True,
                        help="The input text.")
    parser.add_argument("--temperature", 
                        type=float,
                        default=0.6,
                        help="The sampling temperature.")
    parser.add_argument("--max_lenght", 
                        type=int,
                        default=128,
                        help="The maximum number of tokens to generate.")
    args = parser.parse_args()
    generate_code(args)