import argparse

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM


def initialize_model(model_name):
    qeff_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # num_hidden_layers=2,
        continuous_batching=True,
    )
    print(f"{model_name} optimized for AI 100 \n", qeff_model)
    return qeff_model


def export_model(qeff_model):
    generated_onnx_path = qeff_model.export()
    print(generated_onnx_path)


def main():
    parser = argparse.ArgumentParser(description="Run QEfficient model")
    parser.add_argument(
        "--model_name", type=str, required=True, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    # parser.add_argument("--num_hidden_layers", type=int, required=False, default=2)  
    args = parser.parse_args()
    # print(args.__dict__)
    # print("\n")
    
    qeff_model = initialize_model(
        model_name=args.model_name,
    )
    export_model(qeff_model)


if __name__ == "__main__":
    main()

"""
python3 /local/mnt/workspace/sanising/pipeline_prefill/efficient-transformers/run_sampler_copy2.py --model_name meta-llama/Llama-3.3-70B-Instruct
"""
