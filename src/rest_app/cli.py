"""Command-line trigger client for rest_app.

Usage:
    python -m rest_app.cli trigger \\
        --url https://your-rest-app \\
        --token $ADMIN_TOKEN \\
        --dataset ./data/sales.csv \\
        --params ./configs/params.yaml \\
        --project product_dq \\
        --model-name sales_forecasting \\
        [--category mlops] \\
        [--model-family lightgbm] \\
        [--auto-promote] \\
        [--wait] \\
        [--poll-interval 15] \\
        [--output-format text|json]

Token priority: --token flag → ADMIN_TOKEN env var → REST_APP_ADMIN_TOKEN env var.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rest_app.cli",
        description="rest_app CLI — trigger training runs and retrieve model S3 paths.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("trigger", help="Upload dataset+params and fire a training run.")
    tr.add_argument("--url", required=True, help="rest_app base URL (e.g. https://host:8000).")
    tr.add_argument(
        "--token",
        default=None,
        help="Admin token. Falls back to ADMIN_TOKEN / REST_APP_ADMIN_TOKEN env vars.",
    )
    tr.add_argument("--dataset", required=True, type=Path, help="Dataset file (.csv or .parquet).")
    tr.add_argument("--params", required=True, type=Path, help="params.yaml file.")
    tr.add_argument("--project", required=True, help="Project name.")
    tr.add_argument("--model-name", required=True, dest="model_name", help="Model name.")
    tr.add_argument("--category", default="mlops", help="Category (default: mlops).")
    tr.add_argument(
        "--model-family",
        default="lightgbm",
        dest="model_family",
        choices=[
            "regression",
            "classification",
            "forecasting",
            "clustering",
            "ranking",
            "nlp",
            "vision",
            "other",
        ],
        help="Model family (default: lightgbm).",
    )
    tr.add_argument("--description", default="", help="Optional run description.")
    tr.add_argument(
        "--auto-promote",
        action="store_true",
        dest="auto_promote",
        help="Auto-promote on training success.",
    )
    tr.add_argument(
        "--wait",
        action="store_true",
        help="Block until training completes, then print all version S3 paths.",
    )
    tr.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        dest="poll_interval",
        help="Seconds between status polls when --wait is set (default: 15).",
    )
    tr.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format (default: text).",
    )
    return p


def _resolve_token(flag_token: str | None) -> str:
    t = flag_token or os.environ.get("ADMIN_TOKEN") or os.environ.get("REST_APP_ADMIN_TOKEN") or ""
    if not t:
        print(
            "error: admin token required (--token flag, ADMIN_TOKEN, or REST_APP_ADMIN_TOKEN)",
            file=sys.stderr,
        )
        sys.exit(1)
    return t


def cmd_trigger(args: argparse.Namespace) -> None:
    try:
        import httpx
    except ImportError:
        print("error: httpx is required — pip install httpx", file=sys.stderr)
        sys.exit(1)

    base = args.url.rstrip("/")
    token = _resolve_token(args.token)
    dataset = Path(args.dataset)
    params_file = Path(args.params)

    for path, flag in [(dataset, "--dataset"), (params_file, "--params")]:
        if not path.exists():
            print(f"error: {flag} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    with httpx.Client(timeout=120) as client:
        # Step 1: upload and fire the trigger
        with dataset.open("rb") as ds_fh, params_file.open("rb") as params_fh:
            r = client.post(
                f"{base}/trigger-train",
                headers={"X-Admin-Token": token},
                data={
                    "category": args.category,
                    "project": args.project,
                    "model_name": args.model_name,
                    "model_family": args.model_family,
                    "description": args.description,
                    "auto_promote": str(args.auto_promote).lower(),
                },
                files={
                    "dataset": (dataset.name, ds_fh),
                    "params": (params_file.name, params_fh),
                },
            )

        if r.status_code != 200:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            print(f"error: trigger-train HTTP {r.status_code}: {detail}", file=sys.stderr)
            sys.exit(1)

        body = r.json()
        trigger_id: str = body["trigger_id"]
        status_path: str = body["status_url"]

        if args.output_format == "json":
            print(json.dumps({"trigger_id": trigger_id, "status_url": f"{base}{status_path}"}))
        else:
            print(f"trigger_id:  {trigger_id}")
            print(f"status_url:  {base}{status_path}")

        if not args.wait:
            return

        # Step 2: poll until terminal state
        print("waiting for training to complete…", file=sys.stderr)
        status_url = f"{base}{status_path}"
        while True:
            time.sleep(args.poll_interval)
            sr = client.get(status_url)
            if sr.status_code != 200:
                print(f"error: status poll HTTP {sr.status_code}", file=sys.stderr)
                sys.exit(1)
            status_body = sr.json()
            status = status_body["status"]
            print(f"  [{status}]", file=sys.stderr)
            if status == "failed":
                reason = status_body.get("reason", "")
                print(f"training failed: {reason}", file=sys.stderr)
                sys.exit(1)
            if status == "completed":
                break

        # Step 3: fetch all published versions
        versions_url = (
            f"{base}/projects/{args.category}/{args.project}/models/{args.model_name}/versions"
        )
        vr = client.get(versions_url)
        if vr.status_code != 200:
            print(f"warning: versions fetch HTTP {vr.status_code}", file=sys.stderr)
            if args.output_format == "json":
                print(json.dumps({"status": "completed", "versions": []}))
            else:
                print("training completed (versions unavailable)")
            return

        versions = vr.json().get("versions", [])

        if args.output_format == "json":
            print(json.dumps({"status": "completed", "versions": versions}, indent=2))
        else:
            print(f"\nTraining complete — {len(versions)} version(s) available:\n")
            for v in versions:
                channels = ", ".join(v.get("channels") or []) or "—"
                print(f"  {v['version_id']:6}  [{channels:8}]  {v.get('model_pkl_uri', '')}")
                metrics = v.get("metrics") or {}
                if metrics:
                    metrics_str = "  ".join(
                        f"{k}={val:.4g}" if isinstance(val, float) else f"{k}={val}"
                        for k, val in list(metrics.items())[:4]
                    )
                    print(f"          metrics: {metrics_str}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "trigger":
        cmd_trigger(args)


if __name__ == "__main__":
    main()
