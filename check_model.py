from transformers import AutoConfig

config = AutoConfig.from_pretrained("./bert_stress_model_v2")
print(config.num_labels)