from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 指定保存路径，例如：./Llama-3.1-8B
save_path = "./Llama-3.1-8B"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
