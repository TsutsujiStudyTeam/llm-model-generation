# inference/app.py
from pathlib import Path

import gradio as gr
import torch
from unsloth import FastLanguageModel
from huggingface_hub import HfApi
import os
import yaml

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# --- Configuration ---
# Assuming params.yaml is present in the inference directory or we hardcode defaults
# For simplicity in this initial implementation, we will use hardcoded base model.
# In a real scenario, you might copy params.yaml or have a separate inference_config.yaml
# For dynamic Lora adapter loading, we'll fetch available adapters from Hugging Face.

# Base model to load
# This should match hf_model_repo from training/params.yaml
BASE_MODEL_REPO = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
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

    hf_token = os.getenv("HF_TOKEN") or None
    load_kw = dict(
        max_seq_length=2048,  # Must match training max_seq_length
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    if hf_token:
        load_kw["token"] = hf_token

    # Unsloth 2026.x: FastLanguageModel.load_lora_into_model は無い。Hub の LoRA リポジトリ ID を
    # from_pretrained に渡すと adapter_config からベースを解決して読み込む。
    try:
        if adapter_repo and adapter_repo != "No LoRA Adapter":
            print(f"Loading base + LoRA from Hub (PEFT repo): {adapter_repo}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_repo,
                **load_kw,
            )
            current_adapter_repo = adapter_repo
            print(f"Successfully loaded adapter: {adapter_repo}")
        else:
            print(f"Loading base model: {BASE_MODEL_REPO}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=BASE_MODEL_REPO,
                **load_kw,
            )
            current_adapter_repo = "No LoRA Adapter"
            print("Using base model without LoRA adapter.")
        model.eval()
    except Exception as e:
        if adapter_repo and adapter_repo != "No LoRA Adapter":
            print(f"Failed to load adapter {adapter_repo}: {e}")
            current_adapter_repo = "No LoRA Adapter"
            print("Falling back to base model for inference.")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=BASE_MODEL_REPO,
                **load_kw,
            )
            model.eval()
        else:
            raise

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

    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    dev = next(model.parameters()).device
    if hasattr(enc, "to"):
        inputs = enc.to(dev)
        prompt_len = int(inputs.shape[-1])
    else:
        input_ids = enc["input_ids"].to(dev)
        prompt_len = int(input_ids.shape[-1])
        attn = enc.get("attention_mask")
        inputs = dict(input_ids=input_ids)
        if attn is not None:
            inputs["attention_mask"] = attn.to(dev)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # use_cache=False: Unsloth fast KV + recent Transformers で RoPE 形状不整合を避ける（finetune.ipynb 簡易推論と同じ）
    gen_kw = dict(
        max_new_tokens=256,
        use_cache=False,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if isinstance(inputs, dict):
        out_ids = model.generate(**inputs, **gen_kw)
    else:
        out_ids = model.generate(inputs, **gen_kw)

    new_tokens = out_ids[0, prompt_len:]
    yield tokenizer.decode(new_tokens, skip_special_tokens=True)

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
