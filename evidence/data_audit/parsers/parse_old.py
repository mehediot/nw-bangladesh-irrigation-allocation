"""Parse Total Boro, Wheat and Total Potato district tables from BBS yearbooks 2015-2019 (layout text)."""
import re, sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from parse_crops import dist as _dist
def dist(n):
    k = n.lower().replace(' ', '').replace('.', '')
    if 'nawabgonj' in k or 'nawbgonj' in k: return 'Chapainawabganj'
    if k == 'noagaon': return 'Naogaon'
    return _dist(n)
lines = open(sys.argv[1], encoding='utf-8', errors='replace').read().split('\n')
PAT = {'Boro Rice': r'Table.*Estimates.*Total\s*(Boro\s*)?\(Local\+HYV\+\s*Hybrid\)',
       'Wheat': r'Table.*Estimates of Wheat', 'Potato': r'Table.*Estimates\d* of Total Potato'}
out = []
for crop, pat in PAT.items():
    hits = [i for i, l in enumerate(lines) if re.search(pat, l, re.I)]
    seen = set()
    for i in hits:
        hdr = ' '.join(lines[i + 1:i + 3])
        yrs = [int(a) + 1 for a, b in re.findall(r'(20\d\d)\s*-\s*(\d\d)', hdr)]
        if not yrs:
            yrs = [int(a) + 1 for a, b in re.findall(r'(20\d\d)\s*-\s*(\d\d)', lines[i])][-1:]
        for l in lines[i + 1:i + 140]:
            if re.search(r'^\s*Table', l): break
            m = re.match(r'\s*\d+\s+([A-Za-z][A-Za-z ]*?)\s{2,}(.*)$', l)
            if not m: continue
            d = dist(m.group(1))
            if not d: continue
            v = [float(x.replace(',', '')) for x in re.findall(r'\d[\d,]*\.?\d*', m.group(2))]
            for k, y in enumerate(yrs):
                if len(v) >= 5 * (k + 1):
                    a, yt, pr = v[5 * k + 1], v[5 * k + 3], v[5 * k + 4]
                    out.append(dict(crop=crop, district=d, year_end=y, area_ha=a, yield_t_ha=yt, production_t=pr, line=i + 1))
print(json.dumps(out))
