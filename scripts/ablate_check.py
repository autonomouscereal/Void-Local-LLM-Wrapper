import sys, json as _jsn

def lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            if ln.strip():
                yield _jsn.loads(ln)

def main():
    if len(sys.argv) < 3:
        print('usage: ablate_check.py clean.jsonl drops.jsonl', file=sys.stderr)
        sys.exit(2)
    clean = list(lines(sys.argv[1]))
    drops = list(lines(sys.argv[2]))
    # Determinism proxy: ensure scores present and rounded to 1e-6
    for it in clean:
        s = float(it.get('score', 0.0))
        if abs(s - float(f"{s:.6f}")) > 1e-12:
            print('score not rounded', file=sys.stderr); sys.exit(1)
    # Drop reasons enumerated
    for d in drops:
        if not isinstance(d.get('drop_reasons'), list):
            print('drop_reasons missing', file=sys.stderr); sys.exit(1)
    print('ok')

if __name__ == '__main__':
    main()


