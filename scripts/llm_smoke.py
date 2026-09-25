"""Smoke-test an OpenAI-compatible LLM endpoint: is it up, does it answer, how fast.

Works unchanged against the local llama.cpp server and against vLLM on Kaggle, because the harness talks to both
through the same API. Prints one JSON line so the numbers can be pasted into an experiment log.

    uv run python scripts/llm_smoke.py                          # local server on :8080
    uv run python scripts/llm_smoke.py --base-url http://host:8000/v1 --model Qwen/Qwen3.6-27B-FP8
"""
import argparse
import json
import time

from openai import OpenAI

PROMPT = (
    "A 3x3 grid is shown as rows of digits:\n"
    "0 0 0\n0 5 0\n0 0 0\n"
    "The agent pressed ACTION1 (up) and the 5 moved one row up. "
    "Reply with only the new grid, same format."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="none")
    t0 = time.time()
    r = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=args.max_tokens,
        temperature=0.0,
    )
    secs = time.time() - t0
    text = (r.choices[0].message.content or "").strip()
    out_tok = r.usage.completion_tokens if r.usage else None
    print(text)
    print(json.dumps({
        "base_url": args.base_url,
        "model": r.model,
        "secs": round(secs, 2),
        "completion_tokens": out_tok,
        "tok_per_s": round(out_tok / secs, 1) if out_tok else None,
        "correct": text.splitlines()[-3:] == ["0 5 0", "0 0 0", "0 0 0"],
    }))


if __name__ == "__main__":
    main()
