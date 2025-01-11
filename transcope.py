import numpy as np
import plotly.express as px
import transformer_lens
import streamlit as st
from sklearn.decomposition import PCA

available_models = [
    "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl",
    "distillgpt2", "opt-125m", "opt-1.3b", "opt-2.7b", "opt-6.7b", "opt-13b", "opt-30b", "opt-66b",
    "gpt-neo-125M", "gpt-neo-1.3B", "gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20b", 
    "stanford-gpt2-small-a", "stanford-gpt2-small-b", "stanford-gpt2-small-c", "stanford-gpt2-small-d", "stanford-gpt2-small-e", 
    "stanford-gpt2-medium-a", "stanford-gpt2-medium-b", "stanford-gpt2-medium-c", "stanford-gpt2-medium-d", "stanford-gpt2-medium-e",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
    "pythia-70m-deduped", "pythia-160m-deduped", "pythia-410m-deduped", "pythia-1b-deduped", "pythia-1.4b-deduped", "pythia-2.8b-deduped", "pythia-6.9b-deduped", "pythia-12b-deduped",
    "pythia-70m-v0", "pythia-160m-v0", "pythia-410m-v0", "pythia-1b-v0", "pythia-1.4b-v0", "pythia-2.8b-v0", "pythia-6.9b-v0", "pythia-12b-v0",
    "pythia-70m-deduped-v0", "pythia-160m-deduped-v0", "pythia-410m-deduped-v0", "pythia-1b-deduped-v0", "pythia-1.4b-deduped-v0", "pythia-2.8b-deduped-v0", "pythia-6.9b-deduped-v0", "pythia-12b-deduped-v0",
    "pythia-160m-seed1", "pythia-160m-seed2", "pythia-160m-seed3", "solu-1l-pile", "solu-2l-pile", "solu-4l-pile", "solu-6l-pile", "solu-8l-pile",
    "solu-10l-pile", "solu-12l-pile", "solu-1l", "solu-2l", "solu-3l", "solu-4l", "solu-6l", "solu-8l", "solu-10l", "solu-12l",
    "gelu-1l", "gelu-2l", "gelu-3l", "gelu-4l", "attn-only-1l", "attn-only-2l", "attn-only-3l", "attn-only-4l", "attn-only-2l-demo",
    "solu-1l-wiki", "solu-4l-wiki", "redwood_attn_2l", "llama-7b", "llama-13b", "llama-30b", "llama-65b", "Llama-2-7b", "Llama-2-7b-chat", 
    "Llama-2-13b", "Llama-2-13b-chat", "Llama-2-70b-chat", "CodeLlamallama-2-7b", "CodeLlama-7b-python", "CodeLlama-7b-instruct",
    "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-70B", "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct", "othello-gpt", "bert-base-cased", "tiny-stories-1M", "tiny-stories-3M", "tiny-stories-8M", "tiny-stories-28M", 
    "tiny-stories-33M", "tiny-stories-instruct-1M", "tiny-stories-instruct-3M", "tiny-stories-instruct-8M", "tiny-stories-instruct-28M", "tiny-stories-instruct-33M", 
    "tiny-stories-1L-21M", "tiny-stories-2L-33M", "tiny-stories-instruct-1L-21M", "tiny-stories-instruct-2L-33M", "stablelm-base-alpha-3b", "stablelm-base-alpha-7b",
    "stablelm-tuned-alpha-3b", "stablelm-tuned-alpha-7b", "mistral-7b", "mistral-7b-instruct", "mistral-nemo-base-2407", "mixtral", "mixtral-instruct",
    "bloom-560m", "bloom-1b1", "bloom-1b7", "bloom-3b", "bloom-7b1", "santacoder", "qwen-1.8b", "qwen-7b", "qwen-14b", "qwen-1.8b-chat", "qwen-7b-chat", 
    "qwen-14b-chat", "qwen1.5-0.5b", "qwen1.5-0.5b-chat", "qwen1.5-1.8b", "qwen1.5-1.8b-chat", "qwen1.5-4b", "qwen1.5-4b-chat", "qwen1.5-7b", "qwen1.5-7b-chat", 
    "qwen1.5-14b", "qwen1.5-14b-chat", "Qwen/Qwen2-0.5B", "Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B", "Qwen/Qwen2-1.5B-Instruct", "Qwen/Qwen2-7B", 
    "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B", 
    "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B", 
    "Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct", "Qwen/QwQ-32B-Preview", "phi-1", "phi-1_5", "phi-2", "phi-3", "gemma-2b", 
    "gemma-7b", "gemma-2b-it", "gemma-7b-it", "gemma-2-2b", "gemma-2-2b-it", "gemma-2-9b", "gemma-2-9b-it", "gemma-2-27b", "gemma-2-27b-it", 
    "yi-6b", "yi-34b", "yi-6b-chat", "yi-34b-chat", "t5-small", "t5-base", "t5-large", "mGPT"
]

def normalize_to_range(matrix):
    return np.tanh(matrix)

def apply_pca_(matrix, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(matrix)

def create_heatmap(matrix, color_scale='Viridis', width=800, height=600, scale=1.0):
    fig = px.imshow(matrix, color_continuous_scale=color_scale, title="Activation Heatmap")
    
    fig.update_layout(
        width=width, 
        height=height,
        title="Activation Heatmap",
        xaxis=dict(scaleanchor="y"),  
        yaxis=dict(scaleanchor="x"),
    )

    fig.update_traces(
        colorbar=dict(thickness=15),  
        colorscale=color_scale,       
        showscale=True                
    )
    
    return fig

def generate_plot(matrix, normalization=False, apply_pca=False, color_scale='Viridis'):
    if normalization:
        matrix = normalize_to_range(matrix)
    
    
    if apply_pca:
        matrix = apply_pca_(matrix)

    return create_heatmap(matrix, color_scale=color_scale)

st.title("transcope")

model_name = st.sidebar.selectbox("Select Model", available_models)

model = transformer_lens.HookedTransformer.from_pretrained(model_name)

text_input = st.sidebar.text_input("Enter your text", value="Hello World")

logits, activations = model.run_with_cache(text_input)

layer_name = st.sidebar.selectbox("Select Layer", list(activations.keys()))

normalize_option = st.sidebar.checkbox("Normalize Matrix Values", value=True)

color_scale = st.sidebar.selectbox("Select Color Scale", ['Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Rainbow'])

apply_pca = st.sidebar.checkbox("Apply PCA to Reduce Dimensions", value=True)

activation_tensor = activations[layer_name].cpu().numpy()

reshaped_activation = activation_tensor.reshape(-1, activation_tensor.shape[-1])

st.write(f"Layer: {layer_name} Shape: {activation_tensor.shape}")
plot = generate_plot(reshaped_activation, normalization=normalize_option, apply_pca=apply_pca, color_scale=color_scale)

st.plotly_chart(plot)
