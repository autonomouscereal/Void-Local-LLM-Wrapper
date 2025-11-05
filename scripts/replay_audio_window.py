#!/usr/bin/env python3
import os, sys, json, hashlib, argparse

def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def main():
    ap = argparse.ArgumentParser(description="Replay one audio window from distill pack (sanity)")
    ap.add_argument("job_id", help="Job ID under /srv/film2/distill/jobs/<job_id>")
    ap.add_argument("window_idx", type=int, help="Window index to replay")
    ap.add_argument("--distill_root", default="/srv/film2/distill/jobs", help="Root of distill jobs")
    args = ap.parse_args()

    job_dir = os.path.join(args.distill_root, args.job_id)
    if not os.path.isdir(job_dir):
        print(f"[error] job not found: {job_dir}", file=sys.stderr)
        sys.exit(2)

    # Load capsule (window) and artifacts manifest if present
    caps_dir = os.path.join(job_dir, "capsules")
    win_caps = os.path.join(caps_dir, f"AUDIO_W{args.window_idx:03d}.json")
    if not os.path.exists(win_caps):
        print(f"[warn] capsule not found: {win_caps}")

    # Load scores/deltas lines for context (optional)
    scores = os.path.join(job_dir, "scores.jsonl")
    deltas = os.path.join(job_dir, "deltas.jsonl")
    if os.path.exists(scores):
        print("[info] scores tail:")
        for ln in read_text(scores).splitlines()[-3:]:
            print("  ", ln[:160])
    if os.path.exists(deltas):
        print("[info] deltas tail:")
        for ln in read_text(deltas).splitlines()[-3:]:
            print("  ", ln[:160])

    # Verify an example artifact hash if present
    art_dir = os.path.join(job_dir, "artifacts")
    mix_path = os.path.join(art_dir, f"mix_window_{args.window_idx:03d}.wav")
    if os.path.exists(mix_path):
        h = sha256_bytes(read_bytes(mix_path))
        print("[ok] mix sha256:", h)
    else:
        print("[warn] no mix artifact to verify", mix_path)

    print("[done] replay template; integrate engine calls as needed.")

if __name__ == "__main__":
    main()


