"""Compare Qwen2.5-3B-Instruct before and after loading the final LoRA adapter."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from medical_sft.data.medmcqa_training_contract import INSTRUCTION

ADAPTER = ROOT / "release" / "qwen25-3b-final-lora"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="English multiple-choice question")
    parser.add_argument("--options", nargs="+", help="Four or five options in order")
    parser.add_argument("--base-dir", type=Path, help="Local copy of the pinned base model")
    parser.add_argument("--answer-only", action="store_true", help="Use the one-line evaluation prompt")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: CUDA if available, else CPU")
    args = parser.parse_args()
    if (args.question is None) != (args.options is None):
        parser.error("Supply --question and --options together.")
    question = args.question if args.question is not None else "Which organ pumps blood through the body?"
    options = args.options or ["Liver", "Heart", "Kidney", "Lung"]
    if not question.strip() or len(options) not in (4, 5) or any(not option.strip() for option in options):
        parser.error("Supply a nonempty question and four or five nonempty options.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((ADAPTER / "adapter_config.json").read_text(encoding="utf-8"))
    source = args.base_dir or config["base_model_name_or_path"]
    load_options = {"local_files_only": True} if args.base_dir else {"revision": config["revision"]}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable on this machine.")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(source, **load_options)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, **load_options).to(device).eval()
    choices = [f"{letter}. {option}" for letter, option in zip("ABCDE", options)]
    instruction = INSTRUCTION
    max_new_tokens = 512
    if args.answer_only:
        instruction = "Return exactly one line in the format: Answer: X"
        max_new_tokens = 32
    prompt = "\n".join([question, *choices, instruction])
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(device)

    def answer(active_model):
        with torch.inference_mode():
            ids = active_model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
        return tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print("\n".join([question, *choices]), flush=True)
    print("Base:", answer(model), flush=True)
    tuned = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False).eval()
    print("SFT:", answer(tuned), flush=True)


if __name__ == "__main__":
    main()
