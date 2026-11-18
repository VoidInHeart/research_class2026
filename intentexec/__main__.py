import argparse
import json
from pathlib import Path


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("expected zero or a positive integer")
    return result


def main():
    p = argparse.ArgumentParser(description="Open-set intent recognition and single-executor routing")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="train routers and evaluate a fresh experiment")
    run.add_argument("--sgd-root", default="data/sgd")
    run.add_argument("--catalog", default="configs/intent_catalog.json")
    run.add_argument("--out", required=True)
    run.add_argument("--seed", type=int, default=1118)
    run.add_argument("--encoder", choices=["tfidf","minilm"], default="tfidf")
    run.add_argument("--recognition-window", choices=["recent5", "original"], default="recent5",
                     help="first-layer context; recent5 keeps the latest five utterances")
    run.add_argument("--oos-gate", choices=["centroid", "adaptive-support"], default="centroid",
                     help="adaptive-support uses local recent5 MiniLM training examples")
    run.add_argument("--model-path", help="local MiniLM model snapshot")
    run.add_argument("--backend", choices=["retrieval","lora"], default="retrieval")
    run.add_argument("--base-path")
    run.add_argument("--adapters-dir")
    run.add_argument("--device", default="cpu")
    run.add_argument("--max-length", type=int, help="LoRA total context, defaults to the training manifest")
    run.add_argument("--max-new-tokens", type=int, help="LoRA generation budget, defaults to the training manifest")
    run.add_argument("--limit-test", type=nonnegative, default=0, help="deterministic public-ID sample, 0=all")
    run.add_argument("--limit-router", type=nonnegative, default=0, help="0=all")
    run.add_argument("--live-n", type=nonnegative, default=24, help="live timing requests; 0=skip")
    run.add_argument("--repeats", type=int, default=2)
    predict = sub.add_parser("predict", help="execute a new public dialogue prefix")
    predict.add_argument("--run-dir", required=True)
    predict.add_argument("--policy", choices=["always_base","fixed","lookup","matrix","mlp","embedding","matrix_no_gate"], default="matrix")
    predict.add_argument("--text", required=True)
    jev = sub.add_parser("compare-jev", help="explicitly run the paid hosted Jev benchmark")
    jev.add_argument("--run-dir", required=True)
    jev.add_argument("--out", required=True)
    jev.add_argument("--model", required=True, help="prefer a pinned TypeSafe model version")
    jev.add_argument("--limit", type=int, default=20)
    jev.add_argument("--timeout", type=int, default=30)
    jev.add_argument("--local-usd-hour", type=float, help="reserved instance price per hour")
    jev.add_argument("--input-usd-million", type=float, help="verified Jev price per million input tokens")
    jev.add_argument("--output-usd-million", type=float, help="verified Jev price per million output tokens")
    jev.add_argument("--usd-per-1000-decisions", type=float, help="verified per-1000-call fee, if applicable")
    args = p.parse_args()
    if args.command == "run":
        if args.oos_gate != "centroid" and (args.encoder, args.recognition_window) != ("minilm", "recent5"):
            p.error("support gates require --encoder minilm --recognition-window recent5")
        if args.backend == "lora" and not (args.base_path and args.adapters_dir):
            p.error("--backend lora requires --base-path and --adapters-dir")
        if args.backend == "lora" and (args.max_length is None or args.max_new_tokens is None):
            manifest = json.loads((Path(args.adapters_dir) / "manifest.json").read_text(encoding="utf8"))
            if args.max_length is None:
                args.max_length = manifest["max_length"]
            if args.max_new_tokens is None:
                args.max_new_tokens = manifest["max_new_tokens"]
        if args.repeats < 1 or any(x is not None and x < 1 for x in (args.max_length, args.max_new_tokens)):
            p.error("repeats and generation limits must be positive")
        from .experiment import run as experiment
        experiment(args)
    elif args.command == "predict":
        from .experiment import infer_one, load_run
        bundle, pool = load_run(args.run_dir)
        print(json.dumps(infer_one(bundle, pool, args.policy, args.text), ensure_ascii=False, indent=2))
    else:
        from .compare import compare
        if args.limit < 1 or args.timeout < 1:
            p.error("limit and timeout must be positive")
        if any(x is not None and x < 0 for x in (args.local_usd_hour, args.input_usd_million,
                                                  args.output_usd_million, args.usd_per_1000_decisions)):
            p.error("prices must be nonnegative")
        compare(args)


if __name__ == "__main__":
    main()
