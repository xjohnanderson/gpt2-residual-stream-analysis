import torch
from model_utils import load_gpt2, register_hooks
from data_utils import tokenize_sentences
from analysis_utils import calculate_magnitudes
from plotting_utils import plot_distributions, plot_layer_evolution

# 1. Setup
tokenizer, model = load_gpt2()
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are powerful models for natural language processing.",
    "Residual connections are crucial for deep neural networks."
]

# 2. Register Hooks
attn_outputs, mlp_outputs, handles = register_hooks(model)

# 3. Inference
inputs = tokenize_sentences(sentences, tokenizer)
model.eval()
with torch.no_grad():
    model(**inputs)

# 4. Cleanup Hooks
for h in handles:
    h.remove()

# 5. Analysis
attn_layer_mags, all_attn_mags = calculate_magnitudes(attn_outputs)
mlp_layer_mags, all_mlp_mags = calculate_magnitudes(mlp_outputs)

# 6. Visualization
plot_distributions(all_attn_mags, all_mlp_mags)
plot_layer_evolution(attn_layer_mags, mlp_layer_mags)
