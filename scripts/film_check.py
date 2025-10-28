import json as _jsn, sys, hashlib

def ndjson_first_line(path):
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            if ln.strip():
                return _jsn.loads(ln)
    return {}

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return _jsn.load(f)

def main():
    if len(sys.argv) < 5:
        print('usage: film_check.py plan.json shots.jsonl edl.json qc_report.json', file=sys.stderr)
        sys.exit(2)
    plan = load(sys.argv[1])
    shot = ndjson_first_line(sys.argv[2])
    edl = load(sys.argv[3])
    qc  = load(sys.argv[4])
    # Seeds present
    if 'seed' not in plan or 'seed' not in shot:
        print('missing seeds', file=sys.stderr); sys.exit(1)
    # QC thresholds
    if qc.get('color', {}).get('deltaE_avg', 99) > 2.0:
        print('color deltaE too high', file=sys.stderr); sys.exit(1)
    # EDL structure
    if 'tracks' not in edl or 'video' not in edl['tracks']:
        print('invalid edl', file=sys.stderr); sys.exit(1)
    print('ok')

if __name__ == '__main__':
    main()


