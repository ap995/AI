"""SecUR CLI — collect, train, evaluate, and serve."""

import argparse
import json
import os
import sys
from pathlib import Path


def collect_cmd(args):
    """Run data collection pipelines."""
    from sec_ur.data.cve_collector import CVECollector
    from sec_ur.data.mitre_collector import MITRECollector
    from sec_ur.data.owasp_collector import OWASPCollector, CWECollector

    data_dir = args.data_dir
    os.makedirs(f"{data_dir}/raw", exist_ok=True)

    if args.sources == "all" or "cve" in args.sources:
        print("Collecting CVE data...")
        collector = CVECollector(data_dir=f"{data_dir}/raw/cve", api_key=args.nvd_key)
        cves = collector.fetch_full_history() if args.full else collector.fetch_recent(days_back=args.days)
        out = collector.save(cves, "cve_data.json")
        print(f"  Saved {len(cves)} CVEs to {out}")

    if args.sources == "all" or "mitre" in args.sources:
        print("Collecting MITRE ATT&CK data...")
        collector = MITRECollector(data_dir=f"{data_dir}/raw/mitre")
        data = collector.fetch_all()
        print(f"  Enterprise: {len(data['enterprise'])} objects")
        print(f"  Mobile: {len(data['mobile'])} objects")
        print(f"  ICS: {len(data['ics'])} objects")

    if args.sources == "all" or "owasp" in args.sources:
        print("Collecting OWASP data...")
        collector = OWASPCollector(data_dir=f"{data_dir}/raw/owasp")
        top10 = collector.fetch_top10()
        print(f"  Top 10: {len(top10)} items")
        sheets = collector.fetch_cheatsheets()
        print(f"  Cheatsheets: {len(sheets)} items")

    if args.sources == "all" or "cwe" in args.sources:
        print("CWE collection requires specific IDs. Use --cwe-ids")

    print("Data collection complete.")


def build_tokenizer_cmd(args):
    """Train the cybersecurity tokenizer on collected data."""
    from sec_ur.tokenizer import CyberTokenizer

    data_dir = Path(args.data_dir)
    texts = []

    for json_file in data_dir.rglob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for field in ["description", "content", "name", "title"]:
                                if field in item and isinstance(item[field], str):
                                    texts.append(item[field])
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    print(f"Training tokenizer on {len(texts)} documents...")
    tokenizer = CyberTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts[:args.max_texts])

    out_path = args.output or f"{args.data_dir}/tokenizer.json"
    tokenizer.save(out_path)
    print(f"Tokenizer saved to {out_path}")


def train_cmd(args):
    """Train the SecUR model."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from sec_ur.config import SecURConfig
    from sec_ur.data.dataset import CyberSecurityDataset, collate_cyber_batch
    from sec_ur.model.sec_ur import SecURModel
    from sec_ur.training.trainer import SecURTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    hf_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("Loading dataset...")
    dataset = CyberSecurityDataset(
        data_path=args.data_dir,
        tokenizer_fn=hf_tokenizer,
        max_query_len=args.query_len,
        max_doc_len=args.doc_len,
        num_negatives=args.negatives,
    )
    print(f"Dataset: {len(dataset)} pairs")

    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_ds, eval_ds = torch.utils.data.random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_cyber_batch, num_workers=args.workers,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_cyber_batch, num_workers=args.workers,
    )

    config = SecURConfig(
        vocab_size=len(hf_tokenizer),
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )

    model = SecURModel(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters")

    trainer = SecURTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=device,
    )

    history = trainer.train(num_epochs=args.epochs)

    if args.evaluate:
        from sec_ur.eval.secbench import SecBench
        from sec_ur.eval.runner import SecBenchRunner

        doc_store = {}
        for json_file in Path(args.data_dir).rglob("*.json"):
            with open(json_file) as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "id" in item:
                                doc_store[item["id"]] = item.get("description", item.get("content", ""))
                except Exception:
                    pass

        runner = SecBenchRunner(model, hf_tokenizer, doc_store=doc_store)
        results = runner.run_all()
        runner.secbench.print_summary_table()
        runner.secbench.save_report(f"{args.output_dir}/secbench_report.json")

    print("Training complete.")


def evaluate_cmd(args):
    """Run SecBench evaluation on a trained checkpoint."""
    import torch
    from transformers import AutoTokenizer

    from sec_ur.eval.runner import SecBenchRunner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    doc_store = {}
    if args.data_dir:
        for json_file in Path(args.data_dir).rglob("*.json"):
            with open(json_file) as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "id" in item:
                                doc_store[item["id"]] = item.get("description", item.get("content", ""))
                except Exception:
                    pass

    runner = SecBenchRunner.from_checkpoint(args.checkpoint, tokenizer, doc_store=doc_store)
    runner.run_all()
    runner.secbench.print_summary_table()
    runner.secbench.save_report(f"{Path(args.checkpoint).parent}/secbench_report.json")


def serve_cmd(args):
    """Start the FastAPI server."""
    import uvicorn

    port = int(args.port)
    host = args.host
    checkpoint = args.checkpoint or os.environ.get("SECUR_CHECKPOINT", "")

    if checkpoint:
        os.environ["SECUR_CHECKPOINT"] = checkpoint
        print(f"Model checkpoint: {checkpoint}")

    if args.qdrant_host:
        os.environ["QDRANT_HOST"] = args.qdrant_host

    print(f"Starting SecUR API on {host}:{port}")
    print(f"Docs: http://{host}:{port}/docs")

    uvicorn.run("sec_ur.backend.main:app", host=host, port=port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(description="SecUR — Cybersecurity Unified Retriever-Generator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # collect
    p_collect = subparsers.add_parser("collect", help="Collect cybersecurity data")
    p_collect.add_argument("--data-dir", default="data", help="Data directory")
    p_collect.add_argument("--sources", nargs="+", default=["all"], choices=["all", "cve", "mitre", "owasp", "cwe"])
    p_collect.add_argument("--nvd-key", help="NVD API key")
    p_collect.add_argument("--days", type=int, default=30, help="Days back for recent CVEs")
    p_collect.add_argument("--full", action="store_true", help="Fetch full CVE history")
    p_collect.set_defaults(func=collect_cmd)

    # build-tokenizer
    p_tokenizer = subparsers.add_parser("build-tokenizer", help="Train cybersecurity tokenizer")
    p_tokenizer.add_argument("--data-dir", default="data", help="Data directory")
    p_tokenizer.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    p_tokenizer.add_argument("--max-texts", type=int, default=50000, help="Max training texts")
    p_tokenizer.add_argument("--output", default=None, help="Output path for tokenizer")
    p_tokenizer.set_defaults(func=build_tokenizer_cmd)

    # train
    p_train = subparsers.add_parser("train", help="Train SecUR model")
    p_train.add_argument("--data-dir", default="data", help="Training data directory")
    p_train.add_argument("--output-dir", default="data/checkpoints", help="Checkpoint directory")
    p_train.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p_train.add_argument("--batch-size", type=int, default=4, help="Batch size")
    p_train.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    p_train.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    p_train.add_argument("--d-model", type=int, default=512, help="Model dimension")
    p_train.add_argument("--encoder-layers", type=int, default=6, help="Encoder layers")
    p_train.add_argument("--decoder-layers", type=int, default=6, help="Decoder layers")
    p_train.add_argument("--n-heads", type=int, default=8, help="Attention heads")
    p_train.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    p_train.add_argument("--query-len", type=int, default=128, help="Max query length")
    p_train.add_argument("--doc-len", type=int, default=512, help="Max document length")
    p_train.add_argument("--negatives", type=int, default=7, help="Number of negatives")
    p_train.add_argument("--workers", type=int, default=2, help="Data loader workers")
    p_train.add_argument("--evaluate", action="store_true", help="Run SecBench after training")
    p_train.set_defaults(func=train_cmd)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run SecBench evaluation")
    p_eval.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    p_eval.add_argument("--data-dir", default=None, help="Document store for retrieval eval")
    p_eval.set_defaults(func=evaluate_cmd)

    # serve
    p_serve = subparsers.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address")
    p_serve.add_argument("--port", default="8000", help="Port")
    p_serve.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    p_serve.add_argument("--qdrant-host", default=None, help="Qdrant host")
    p_serve.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    p_serve.set_defaults(func=serve_cmd)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
