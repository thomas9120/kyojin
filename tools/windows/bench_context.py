#!/usr/bin/env python3
"""Prefill and decode scaling with context length, against a running Kyojin server.

    python tools/windows/bench_context.py                          # 2K..32K, 1 rep
    python tools/windows/bench_context.py --sizes 2048 8192 32768 --reps 2
    python tools/windows/bench_context.py --decode 128 --json

tools/bench.py reports one prefill number (about 3.5K tokens) and one decode number. This one
walks a ladder of prompt sizes, so the cost of the first chunk (a fresh 4096-row chunk reads every
expert) and of long-context attention are visible separately from the steady-state figure.

Every request carries a unique tag, so no cached prefix can answer it. The numbers are the
server's own (usage.prompt_tokens, timings.prompt_ms / timings.predicted_ms), not wall-clock of
the HTTP call, so the client adds no overhead beyond serialisation.
"""

import argparse
import json
import random
import string
import sys
import time
import urllib.error
import urllib.request

FILLER = (
    "The river ran low that summer, and the mill wheel turned only when the wind pushed the water "
    "back up the channel, so the miller spent his days repairing what the drought had broken. "
)


def post(base: str, path: str, payload: dict, timeout: float = 3600.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tag() -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(12))


def token_count(base: str, text: str) -> int:
    try:
        ids = post(base, "/tokenize", {"text": text}, timeout=120.0)
        if isinstance(ids, dict):
            ids = ids.get("tokens") or ids.get("ids") or ids.get("data")
        return len(ids)
    except urllib.error.HTTPError:
        return max(1, len(text) // 4)


def build_prompt(base: str, tokens: int) -> str:
    """Filler text that tokenises to about `tokens` tokens (measured through the server's tokenizer)."""
    head = f"[{tag()}] Summarise the following text in one sentence.\n\n"
    per = max(1.0, token_count(base, head + FILLER) - token_count(base, head))
    reps = max(1, round(tokens / per))
    for _ in range(4):
        text = head + FILLER * reps
        n = token_count(base, text)
        if abs(n - tokens) <= max(16, tokens // 200):
            return text
        reps = max(1, round(reps * tokens / max(n, 1)))
    return head + FILLER * reps


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None, help="model id (default: first id in /v1/models)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--decode", type=int, default=64, help="tokens generated per size after the prefill (0 = skip)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model = args.model
    if model is None:
        with urllib.request.urlopen(args.base + "/models", timeout=60) as r:
            model = json.loads(r.read())["data"][0]["id"]

    rows = []
    for size in args.sizes:
        for rep in range(args.reps):
            prompt = build_prompt(args.base, size)
            t0 = time.time()
            out = post(args.base, "/chat/completions", {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max(1, args.decode),
                "temperature": 0.0,
                "top_k": 1,
            })
            wall = time.time() - t0
            t = out.get("timings") or {}
            pn = t.get("prompt_n") or out["usage"]["prompt_tokens"]
            pm = (t.get("prompt_ms") or 0) / 1000.0
            dn = t.get("predicted_n") or out["usage"]["completion_tokens"]
            dm = (t.get("predicted_ms") or 0) / 1000.0
            row = {
                "target": size, "rep": rep, "prompt_tokens": pn, "prompt_s": round(pm, 3),
                "prefill_tps": round(pn / pm, 1) if pm > 0 else None,
                "decode_tokens": dn, "decode_s": round(dm, 3),
                "decode_tps": round(dn / dm, 2) if dm > 0 else None,
                "wall_s": round(wall, 2),
            }
            rows.append(row)
            print(f"prompt {pn:6d} tok -> prefill {pm:7.2f} s = {row['prefill_tps']:7.1f} tok/s"
                  f"   decode {dn:4d} tok = {row['decode_tps'] or 0:6.2f} tok/s"
                  f"   (wall {wall:6.2f} s)", flush=True)

    if args.json:
        print("\n```json")
        print(json.dumps({"base": args.base, "model": model, "rows": rows}, indent=1))
        print("```")
    return 0


if __name__ == "__main__":
    sys.exit(main())
