# inference/app.py
import gradio as gr
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer, AutoTokenizer
from huggingface_hub import HfApi
import os
import yaml

# --- Configuration ---
# Assuming params.yaml is present in the inference directory or we hardcode defaults
# For simplicity in this initial implementation, we will use hardcoded base model.
# In a real scenario, you might copy params.yaml or have a separate inference_config.yaml
# For dynamic Lora adapter loading, we'll fetch available adapters from Hugging Face.

# Base model to load
# This should match hf_model_repo from training/params.yaml
BASE_MODEL_REPO = "unsloth/llama-3.2-alpaca-bnb-4bit"
# This is a placeholder, in a real scenario you would list available LoRA adapters
# or fetch them from a Hugging Face user/org.
# We expect LoRA adapters to be pushed with model.push_to_hub(repo_id, token=True)
# The repo_id should be something like "YOUR_HF_USERNAME/your-lora-adapter"
DEFAULT_LORA_ADAPTER_REPO = "YOUR_HF_USERNAME/my-llama-3.2-lora-adapter" # Placeholder


# --- Global Variables for Model ---
model = None
tokenizer = None
current_adapter_repo = None

# --- Helper Functions ---

def load_model_and_tokenizer(adapter_repo: str = None):
    """
    Loads the base model and tokenizer, and optionally attaches a LoRA adapter.
    """
    global model, tokenizer, current_adapter_repo

    if model is not None and tokenizer is not None and current_adapter_repo == adapter_repo:
        print(f"Model and tokenizer for {adapter_repo} already loaded.")
        return model, tokenizer

    print(f"Loading base model: {BASE_MODEL_REPO}")
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_REPO,
        max_seq_length=2048, # Must match training max_seq_length
        dtype=torch.bfloat16, # Use bfloat16 for inference on modern GPUs
        load_in_4bit=True,
    )

    if adapter_repo and adapter_repo != "No LoRA Adapter":
        print(f"Loading LoRA adapter: {adapter_repo}")
        try:
            # Re-initialize model with adapter if changing
            if current_adapter_repo != adapter_repo and current_adapter_repo is not None:
                # Need to re-load base model if changing adapters without merging
                # For simplicity, we re-load the base model always
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=BASE_MODEL_REPO,
                    max_seq_length=2048,
                    dtype=torch.bfloat16,
                    load_in_4bit=True,
                )
            
            # Load the LoRA adapter
            FastLanguageModel.load_lora_into_model(
                model,
                adapter_repo,
                token=os.getenv("HF_TOKEN") # Use HF_TOKEN from environment variables for private repos
            )
            model.eval() # Set model to evaluation mode
            current_adapter_repo = adapter_repo
            print(f"Successfully loaded adapter: {adapter_repo}")
        except Exception as e:
            print(f"Failed to load adapter {adapter_repo}: {e}")
            # Fallback to base model if adapter loading fails
            current_adapter_repo = "No LoRA Adapter"
            print("Falling back to base model for inference.")
    elif adapter_repo == "No LoRA Adapter":
        current_adapter_repo = "No LoRA Adapter"
        print("Using base model without LoRA adapter.")
    
    # Ensure model is in inference mode
    FastLanguageModel.for_inference(model)

    return model, tokenizer

def get_available_lora_adapters():
    """
    Fetches a list of LoRA adapters from the Hugging Face Hub under the user's namespace.
    This requires HF_TOKEN to be set in environment variables if repos are private.
    """
    api = HfApi()
    
    # Get the authenticated user's organization/username
    # This might require specific permissions or a more robust way to get user namespace
    # For now, let's assume a common user/org prefix or hardcode
    
    # This is a mock function. In a real scenario, you'd list repos by user/org
    # and filter for LoRA adapters.
    # Example: filter by tags, or by checking content of the repo.
    
    # For initial implementation, we can list common unsloth LoRA adapters or a predefined list
    available_adapters = ["No LoRA Adapter", DEFAULT_LORA_ADAPTER_REPO]
    
    # Add an example if the user has updated their params.yaml
    if "YOUR_HF_USERNAME" not in DEFAULT_LORA_ADAPTER_REPO:
        available_adapters.append(DEFAULT_LORA_ADAPTER_REPO)

    # In a real application, you would list repositories for the authenticated user
    # or a specific organization and filter for LoRA adapters.
    # e.g., repos = api.list_repos(author="your_hf_username", search="lora-adapter")
    
    print(f"Available adapters: {available_adapters}")
    return list(set(available_adapters)) # Return unique adapters

# --- Gradio Interface Functions ---

def chat_response(message, history, adapter_repo):
    """
    Generates a response from the LLM based on the chat history and selected adapter.
    """
    global model, tokenizer

    # Ensure model and tokenizer are loaded with the selected adapter
    model, tokenizer = load_model_and_tokenizer(adapter_repo)

    # Convert chat history to model input format
    messages = []
    for human, agent in history:
        messages.append({"role": "user", "content": human})
        if agent:
            messages.append({"role": "assistant", "content": agent})
    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda") # Assuming GPU is available in HF Spaces

    # Use TextStreamer for streaming output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generate response
    # The generation happens in a separate thread/process for streaming in Gradio
    response_generator = model.generate(
        inputs,
        streamer=streamer,
        max_new_tokens=256,
        use_cache=True,
    )
    
    # Gradio expects a generator for streaming
    for new_token in response_generator:
        # Decode only the new tokens to send to Gradio
        decoded_text = tokenizer.decode(new_token, skip_special_tokens=True)
        yield decoded_text

# --- Gradio UI ---

with gr.Blocks(title="LLM Inference with Dynamic LoRA Adapters") as demo:
    gr.Markdown(
        """
        # LLM Inference with Dynamic LoRA Adapters
        Select a LoRA adapter from the dropdown to switch models dynamically.
        """
    )

    available_adapters = gr.Dropdown(
        choices=get_available_lora_adapters(),
        label="Select LoRA Adapter",
        value="No LoRA Adapter", # Default to no adapter
        interactive=True
    )

    chatbot = gr.ChatInterface(
        fn=chat_response,
        additional_inputs=[available_adapters],
        examples=[
            ["日本の首都はどこですか？"],
            ["私は犬が好きです。これを否定文に変換してください。"],
        ],
        title="Chat with LLM",
        # Customizing send button text
        submit_btn="Send",
        stop_btn="Stop",
        clear_btn="Clear",
    )

    # Initial model load when the app starts
    demo.load(lambda: load_model_and_tokenizer(available_adapters.value))

if __name__ == "__main__":
    demo.launch(debug=True)
