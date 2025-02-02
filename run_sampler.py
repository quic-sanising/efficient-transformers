from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
qeff_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    num_hidden_layers=2,
    include_sampler=True,
    # is_tlm=True,
    # return_pdfs=True,
)
print(f"{model_name} optimized for AI 100 \n", qeff_model)

generated_onnx_path = qeff_model.export()
print(generated_onnx_path)

generated_qpc_path = qeff_model.compile(onnx_path=generated_onnx_path, num_speculative_tokens=0)
print(generated_qpc_path)
