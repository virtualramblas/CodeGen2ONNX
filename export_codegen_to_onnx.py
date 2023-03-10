import argparse
import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
codegen_dir = os.path.join(file_dir, "CodeGen")
sys.path.append(codegen_dir)
from jaxformer.hf.sample import (
    create_model,
    create_custom_gpt2_tokenizer,
    set_seed,
)

from fastgpt import (
    generate_onnx_representation,
    quantize,
    test_onnx_inference,
    test_torch_inference,
)

""" 
    Function to test the original CodeGen model.
"""
def test_original_model(model):
    # Settings to try out the pre-trained PyTorch model
    rng_seed = 42
    rng_deterministic = True
    p = 0.95
    t = 0.2
    max_length = 128
    batch_size = 1
    context = "def hello_world():"
    set_seed(rng_seed, deterministic=rng_deterministic)

    is_test_successful = True
    try:
        # Test the model
        test_torch_inference(model)
    except:
        is_test_successful = False

    return is_test_successful

""" 
    Function that converts a CodeGen pre-trained model
    in ONNX format first and then performs quantization
    of the latter.
"""
def export(checkpoint="checkpoints/codegen-350M-mono",
            delete_temp_model=False,
            delete_original_model=False):
    pad_token = 50256
    model = create_model(ckpt=checkpoint, fp16=False)
    model.eval()
    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = "left"
    tokenizer.pad_token = pad_token

    # Test the pre-trained model before converting it
    if test_original_model(model):
        # Convert the original model to the ONNX format
        onnx_path = generate_onnx_representation(model)
        model_path = checkpoint
        onnx_path = os.path.join(model_path, "onnx/model.onnx")
        test_onnx_inference(onnx_path, model.config)

        # Quantize the ONNX model
        quantized_onnx_path = quantize(onnx_path)
        test_onnx_inference(quantized_onnx_path, model.config)

        # Delete the original and/or temporal model
        if delete_temp_model:
            os.remove(onnx_path)
        if delete_original_model:
            os.remove(os.path.join(model_path, "pytorch_model.bin"))
    else:
        print("Execution of the original model didn't complete successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", 
                        type=str,
                        required=True,
                        help="The path of the pre-trained CodeGen model to export.")
    parser.add_argument("--delete_temp_model", 
                        type=str,
                        default=True,
                        help="If True, delete the temporary ONNX model (before quantization).")
    parser.add_argument("--delete_original_model", 
                        type=str,
                        default=False,
                        help="If True, delete the original CodeGen model.")
    args = parser.parse_args()
    export(args.model_path, args.delete_temp_model, args.delete_original_model)
