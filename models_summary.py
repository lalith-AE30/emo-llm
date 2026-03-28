import numpy as np

model_names = [
    "Llama3.2_1B",
    "Llama3.1_8B",
    "gemma-2-2b-it",
    "gemma-2-9b-it",
    "Phi-3.5-mini-instruct",
    "Phi-3-medium-128k-instruct",
    # 'OLMo-1B-hf', 'OLMo-7B-0724-Instruct-hf',
    "Ministral-8B-Instruct-2410",
    "Mistral-Nemo-Instruct-2407",
    "OLMo-2-1124-7B-Instruct",
    "OLMo-2-1124-13B-Instruct",
]
layers_counts = [
    16,
    32,
    26,
    42,
    32,
    40,
    #  16, 32,
    36,
    40,
    32,
    40,
]

model_formal_names = [
    "Llama3 1B",
    "Llama3 8B",
    "Gemma2 2B",
    "Gemma2 9B",
    "Phi3 4B",
    "Phi3 14B",
    #'OLMo 1B', 'OLMo 7B',
    "Mistral 8B",
    "Mistral 12B",
    "OLMo2 7B",
    "OLMo2 13B",
]

model_hf_names = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-medium-128k-instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
]

# sort the models based on the layer counts

# get the sorted indices based on the layer counts
indices = np.argsort(layers_counts)

# model_names = [model_names[i] for i in indices]
# model_formal_names = [model_formal_names[i] for i in indices]
# model_hf_names = [model_hf_names[i] for i in indices]
# layers_counts = [layers_counts[i] for i in indices]


from transformers import AutoConfig, AutoTokenizer

print(len(model_hf_names))
for model_name in model_hf_names[:]:
    print("----------------------------------")
    try:
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        # continue
        print(" ", e)
    print(
        f"{model_name}: hidden size (d_model) {config.hidden_size}, num layers {config.num_hidden_layers}, Pre-Norm:?, Post-Norm:?, Post-attention dropout:?, Layer-Norm-Type:?, Non-Linearity {config.hidden_act}, Feedforward dimension {config.intermediate_size}, Grouped Query: {config.num_key_value_heads != config.num_attention_heads}, Num heads {config.num_attention_heads}, Num KV heads {config.num_key_value_heads}, Head Size: {config.hidden_size // config.num_attention_heads}, Tied Embeddings: {config.tie_word_embeddings}, Tokenizer: {tokenizer.__class__.__name__}, Vocab Size: {config.vocab_size}, Max Position Embeddings: {config.max_position_embeddings}"
    )
    print(" ")
    # print(f"Model configuration: {config}")
