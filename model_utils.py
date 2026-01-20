import torch
from transformers import AutoTokenizer, AutoModel

def load_gpt2():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained('gpt2', output_hidden_states=True)
    return tokenizer, model

def register_hooks(model):
    attn_outputs = []
    mlp_outputs = []
    hook_handles = []

    def save_attn_output(module, input, output):
        attn_outputs.append(output[0])

    def save_mlp_output(module, input, output):
        mlp_outputs.append(output)

    for block in model.h:
        hook_handles.append(block.attn.register_forward_hook(save_attn_output))
        hook_handles.append(block.mlp.register_forward_hook(save_mlp_output))
        
    return attn_outputs, mlp_outputs, hook_handles
