import json as _jsn, sys, hashlib

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return _jsn.load(f)

def sha(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def main():
    if len(sys.argv) < 3:
        print('usage: icw_check.py plan1.json plan2.json', file=sys.stderr)
        sys.exit(2)
    a = load(sys.argv[1]); b = load(sys.argv[2])
    # Determinism: sections.PACK identical
    pa = a.get('sections', {}).get('PACK', '')
    pb = b.get('sections', {}).get('PACK', '')
    if pa != pb:
        print('PACK mismatch', file=sys.stderr); sys.exit(1)
    # Budget: estimated_tokens <= budget +/- 3%
    est = int(a.get('estimated_tokens', 0)); bud = int(a.get('budget_tokens', 0))
    if est > int(bud * 1.03):
        print('budget overrun', file=sys.stderr); sys.exit(1)
    print('ok')

if __name__ == '__main__':
    main()


