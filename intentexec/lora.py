"""Train three named PEFT adapters on the fit split, or serve them on one local base.

Training and inference share the same prompt builder and token budget. Only the
generated answer plus EOS is supervised. A failed training run leaves no published
adapter pool. Use python -m intentexec.lora --help for the training entry point.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
from threading import Lock
from time import perf_counter

from .executors import BASE, EXPERTS, function_catalog, specialist_rows, valid_call


def _json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint_fit(rows):
    parts = [_json([r["id"], r["text"], r["intent"], r["gold"]]) for r in rows]
    return hashlib.sha256("\n".join(sorted(parts)).encode()).hexdigest()


def catalog_text(schema):
    """All methods and argument names, retaining a common execution contract."""
    tools = []
    for (service, method), spec in sorted(function_catalog(schema).items()):
        args = list(spec.get("required_slots", ()))
        args += [name+"?" for name in spec.get("optional_slots", {})]
        tools.append(f"{service}.{method}({','.join(args)})")
    return "\n".join(tools)


def prompt_ids(tokenizer, text, tools, max_length, max_new_tokens):
    allowed = max_length - max_new_tokens
    instruction = ('Return exactly one JSON object {"service":"...","method":"...","arguments":'
                   '{"slot":"value"}} for the next tool call. Use the public tool list below; '
                   'a ? marks an optional argument. Include relevant values from the dialogue.\n'
                   + tools + "\nDialogue:\n")
    def encode(dialogue):
        request = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction + dialogue}],
            tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(request, add_special_tokens=False)
    result = encode(text)
    if len(result) <= allowed:
        return result
    if len(encode("")) >= allowed:
        raise ValueError("Full public tool catalog exceeds the prompt budget; increase max_length")
    # Drop oldest context only. The full catalog and the final user request remain.
    turns = text.splitlines()
    while len(turns) > 1:
        turns.pop(0)
        result = encode("\n".join(turns))
        if len(result) <= allowed:
            return result
    raise ValueError("Latest utterance plus full public catalog exceeds the prompt budget")


def training_example(tokenizer, row, tools, max_length, max_new_tokens):
    prefix = prompt_ids(tokenizer, row["text"], tools, max_length, max_new_tokens)
    target = tokenizer.encode(_json(row["gold"]), add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Shared base must have an EOS token")
    target.append(tokenizer.eos_token_id)
    if len(target) > max_new_tokens:
        raise ValueError(f"Gold call for {row['id']} exceeds max_new_tokens")
    return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


def collate(batch, pad_id):
    import torch
    length = max(len(r["input_ids"]) for r in batch)
    ids, masks, labels = [], [], []
    for row in batch:
        count = len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * (length-count))
        masks.append([1] * count + [0] * (length-count))
        labels.append(row["labels"] + [-100] * (length-count))
    return {k: torch.tensor(v, dtype=torch.long) for k, v in
            (("input_ids", ids), ("attention_mask", masks), ("labels", labels))}


def _runtime():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as e:
        raise RuntimeError("Install requirements-lora.txt for the optional execution backend") from e
    return torch, AutoModelForCausalLM, AutoTokenizer, set_seed, LoraConfig, PeftModel, get_peft_model


def _open_base(path, device):
    torch, model_class, tokenizer_class, *_ = _runtime()
    dtype = torch.float32
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        # BF16 or FP32 keeps the manual training loop free of FP16 loss scaling.
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
    tokenizer = tokenizer_class.from_pretrained(str(path), local_files_only=True)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("Use an instruction checkpoint with a chat template and EOS")
    model = model_class.from_pretrained(str(path), local_files_only=True, torch_dtype=dtype)
    return model.to(device), tokenizer


def check_context(model, max_length, max_new_tokens):
    if max_new_tokens <= 1 or max_length <= max_new_tokens:
        raise ValueError("Require 1 < max_new_tokens < max_length")
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit and max_length > limit:
        raise ValueError(f"max_length {max_length} exceeds base model context {limit}")


class LoRAPool:
    experts = list(EXPERTS)
    base = BASE

    def __init__(self, schema, config, fit_rows):
        torch, _, _, _, _, peft_model, _ = _runtime()
        location = Path(config["adapters_dir"])
        manifest = json.loads((location / "manifest.json").read_text(encoding="utf8"))
        self.tools = catalog_text(schema)
        self.catalog = function_catalog(schema)
        base_path = Path(config["base_path"]).resolve()
        if (manifest["fit_sha256"] != fingerprint_fit(fit_rows)
                or manifest["tools_sha256"] != hashlib.sha256(self.tools.encode()).hexdigest()
                or list(EXPERTS) != manifest["experts"]
                or base_path != Path(manifest["base_path"]).resolve()):
            raise ValueError("Adapter training provenance mismatches this split, tool pool or base")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.get("max_length", manifest["max_length"])
        self.max_new_tokens = config.get("max_new_tokens", manifest["max_new_tokens"])
        base, self.tokenizer = _open_base(base_path, self.device)
        check_context(base, self.max_length, self.max_new_tokens)
        self.model = peft_model.from_pretrained(base, str(location/EXPERTS[0]), adapter_name=EXPERTS[0])
        for name in EXPERTS[1:]:
            self.model.load_adapter(str(location/name), adapter_name=name)
        self.model.eval()
        self._lock = Lock()
        self._torch = torch

    def execute(self, executor_id, text):
        if executor_id not in (*EXPERTS, BASE):
            raise KeyError(executor_id)
        ids = prompt_ids(self.tokenizer, text, self.tools, self.max_length, self.max_new_tokens)
        inputs = self._torch.tensor([ids], device=self.device)
        # set_adapter/disable_adapter mutate the shared model; guard the whole call.
        with self._lock, self._torch.inference_mode():
            if executor_id != BASE:
                self.model.set_adapter(executor_id)
            guard = self.model.disable_adapter() if executor_id == BASE else nullcontext()
            with guard:
                outputs = self.model.generate(input_ids=inputs,
                                              attention_mask=self._torch.ones_like(inputs),
                                              max_new_tokens=self.max_new_tokens, do_sample=False,
                                              num_beams=1, pad_token_id=self.tokenizer.eos_token_id)
        completion = self.tokenizer.decode(outputs[0, len(ids):], skip_special_tokens=True)
        try:
            value = json.loads(completion.strip())
        except (ValueError, TypeError):
            return None
        return value if valid_call(value, self.catalog) else None


def train_pool(fit_rows, known, schema, config):
    """Train from public fit examples only. Caller supplies the canonical split."""
    torch, _, _, set_seed, lora_config, _, get_peft_model = _runtime()
    known = set(known)
    if not fit_rows or any(r["intent"] not in known for r in fit_rows):
        raise ValueError("The LoRA fit partition must contain only known intents")
    out, base_path = Path(config["output"]).resolve(), Path(config["base_path"]).resolve()
    if out.exists():
        raise FileExistsError(f"Use a new adapter output directory: {out}")
    if min(config["epochs"], config["batch_size"], config["rank"], config["learning_rate"]) <= 0:
        raise ValueError("Epochs, batch size, rank and learning rate must be positive")
    out.parent.mkdir(parents=True, exist_ok=True)
    tools = catalog_text(schema)
    manifest = {**config, "base_path": str(base_path), "experts": list(EXPERTS),
                "known": sorted(known), "fit_sha256": fingerprint_fit(fit_rows),
                "tools_sha256": hashlib.sha256(tools.encode()).hexdigest(), "training": {}}
    with tempfile.TemporaryDirectory(prefix=".adapters-",dir=out.parent) as tmp:
        staged = Path(tmp)/"pool"
        staged.mkdir()
        for name in EXPERTS:
            started = perf_counter()
            set_seed(config["seed"])
            base, tokenizer = _open_base(base_path, config["device"])
            check_context(base, config["max_length"], config["max_new_tokens"])
            adapter = get_peft_model(base, lora_config(
                r=config["rank"], lora_alpha=2*config["rank"], lora_dropout=0.0,
                target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
            adapter.train()
            selected = specialist_rows(fit_rows, name, config["seed"], config.get("per_expert"))
            examples = [training_example(tokenizer, r, tools, config["max_length"],
                                         config["max_new_tokens"]) for r in selected]
            optim = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad),
                                      lr=config["learning_rate"])
            total, steps = 0.0, 0
            gen = torch.Generator().manual_seed(config["seed"])
            for epoch in range(config["epochs"]):
                order = torch.randperm(len(examples), generator=gen).tolist()
                for start in range(0, len(order), config["batch_size"]):
                    items = [examples[j] for j in order[start:start+config["batch_size"]]]
                    batch = collate(items, tokenizer.eos_token_id)
                    optim.zero_grad(set_to_none=True)
                    loss = adapter(**{k: v.to(config["device"]) for k,v in batch.items()},
                                   use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss for {name}")
                    loss.backward()
                    optim.step()
                    total += float(loss.detach().cpu())
                    steps += 1
            adapter.save_pretrained(staged/name)
            manifest["training"][name] = {"examples": len(examples), "steps": steps,
                                          "mean_loss": total/steps,
                                          "wall_seconds": perf_counter()-started,
                                          "fit_ids": [r["id"] for r in selected]}
            del loss, batch, optim, adapter, base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        (staged/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        staged.rename(out)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sgd-root", default="data/sgd")
    p.add_argument("--catalog", default="configs/intent_catalog.json")
    p.add_argument("--base-path", required=True, help="Local instruction checkpoint directory")
    p.add_argument("--output", required=True, help="New directory for the trained pool")
    p.add_argument("--device", default="cuda")
    p.add_argument("--per-expert", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--seed", type=int, default=1118)
    args = vars(p.parse_args())
    from .data import load_sgd, public_schema, split_rows
    root = Path(args["sgd_root"])
    rows = load_sgd(root, Path(args["catalog"]))
    splits = split_rows(rows, seed=args["seed"])
    train_pool(splits["fit"], splits["known"], public_schema(root), args)
    print(f"Published trained adapters to {args['output']}")


if __name__ == "__main__":
    main()
