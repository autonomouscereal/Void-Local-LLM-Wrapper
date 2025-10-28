import sys, json as _jsn

def lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            if ln.strip():
                yield _jsn.loads(ln)

def main():
    if len(sys.argv) < 5:
        print('usage: teacher_check.py traces.jsonl sft.jsonl dpo.jsonl toolpolicy.jsonl', file=sys.stderr)
        sys.exit(2)
    traces = list(lines(sys.argv[1]))
    sft    = list(lines(sys.argv[2]))
    dpo    = list(lines(sys.argv[3]))
    tp     = list(lines(sys.argv[4]))
    # Tool call seeds present
    for t in traces:
        for c in t.get('tool_calls', []):
            if 'seed' not in c:
                print('tool call missing seed', file=sys.stderr); sys.exit(1)
    # SFT minimal
    for r in sft:
        if 'input' not in r or 'output' not in r or 'meta' not in r:
            print('sft missing fields', file=sys.stderr); sys.exit(1)
    # DPO minimal
    for r in dpo:
        if 'prompt_hash' not in r or 'chosen' not in r or 'rejected' not in r:
            print('dpo missing fields', file=sys.stderr); sys.exit(1)
    # Toolpolicy minimal
    for r in tp:
        if 'decision' not in r or 'params' not in r:
            print('toolpolicy missing fields', file=sys.stderr); sys.exit(1)
    print('ok')

if __name__ == '__main__':
    main()


