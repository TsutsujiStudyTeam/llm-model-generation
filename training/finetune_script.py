
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments, TextStreamer, AutoTokenizer
from trl import SFTTrainer
from peft import LoraConfig
import yaml
import os
from datasets import load_dataset
# from huggingface_hub import notebook_login, HfApi # notebook_loginはipynbに残す

def main():
    # Load parameters from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    hf_model_repo = params["hf_model_repo"]
    
    # Input your Hugging Face LoRA Repository Name
    # (e.g., "your-username/your-repo-name")
    hf_lora_repo_input = input("Enter your Hugging Face LoRA Repository Name: ")
    if hf_lora_repo_input:
        hf_lora_repo = hf_lora_repo_input
    else:
        hf_lora_repo = params["hf_lora_repo"] # Fallback to params.yaml if no input
        print(f"No Hugging Face LoRA Repository Name entered, using default from params.yaml: {hf_lora_repo}")

    lora_r = params["lora_r"]
    lora_alpha = params["lora_alpha"]
    lora_dropout = params["lora_dropout"]
    max_seq_length = params["max_seq_length"]
    per_device_train_batch_size = params["per_device_train_batch_size"]
    gradient_accumulation_steps = params["gradient_accumulation_steps"]
    warmup_steps = params["warmup_steps"]
    max_steps = params["max_steps"]
    learning_rate = params["learning_rate"]
    fp16 = params["fp16"]
    logging_steps = params["logging_steps"]
    output_dir = params["output_dir"]
    optim = params["optim"]
    seed = params["seed"]

    # 2. Data Preparation
    print("2. Data Preparation")
    # Load your dataset
    # Ensure your dataset.jsonl is in the `data/` directory
    dataset = load_dataset("json", data_files={"train": "../data/dataset.jsonl"}, split="train")

    # Define prompt template for fine-tuning
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    # Define an EOS token for the model
    # tokenizerは後でロードされるので、ここではNoneで初期化しておく
    # EOS_TOKEN = tokenizer.eos_token if tokenizer else "<|end_of_text|>"
    EOS_TOKEN = "<|end_of_text|>" # tokenizerロード前に使うのでハードコード

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN
            text = alpaca_prompt.format(instruction, input_text, output_text) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    print("Data preparation complete.")

    # 3. Model Loading
    print("3. Model Loading")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = hf_model_repo,
        max_seq_length = max_seq_length,
        dtype = None, # None for auto detection. Use torch.bfloat16 for A100.
        load_in_4bit = True,
    )

    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        bias = "none",
        use_gradient_checkpointing = "current_device",
        random_state = seed,
        max_seq_length = max_seq_length,
    )
    print("Model loading and LoRA configuration complete.")

    # 4. Tokenizer (Already handled in Model Loading, but shown as a separate step for clarity)
    # The tokenizer is loaded along with the model.

    # 5. Training
    print("5. Training")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can be True for faster training, but might lead to truncation issues
        args = TrainingArguments(
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            warmup_steps = warmup_steps,
            max_steps = max_steps,
            learning_rate = learning_rate,
            fp16 = fp16,
            logging_steps = logging_steps,
            output_dir = output_dir,
            optim = optim,
            seed = seed,
        ),
    )

    trainer.train()
    print("Training complete.")

    # 6. Save LoRA adapter
    print("6. Saving LoRA adapter locally...")
    model.save_pretrained_merged("lora_adapter_merged", tokenizer)
    print("LoRA adapter saved locally.")

    # 7. Upload to Hugging Face Hub
    print(f"7. Uploading LoRA adapter to Hugging Face Hub: {hf_lora_repo}")
    # Push the LoRA adapter to Hugging Face Hub
    # The model will be pushed to the repository specified in hf_lora_repo
    model.push_to_hub(hf_lora_repo, token=True) # token=Trueはnotebook_login()で認証されている前提
    tokenizer.push_to_hub(hf_lora_repo, token=True)
    print("LoRA adapter uploaded to Hugging Face Hub.")

if __name__ == "__main__":
    main()
