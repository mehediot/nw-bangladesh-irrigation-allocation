import re, sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from parse_old_dist import dist
lines = open(sys.argv[1], encoding='utf-8', errors='replace').read().split('\n')
out = []
for i, l in enumerate(lines):
    m = re.search(r'Area\s*(?:and|&)\s*Production of\s+Rabi\s*Maize\s*by\s*District.*?(20\d\d)-(?:20)?\d\d\s*to\s*(20\d\d)-\d\d', l, re.I)
    if not m: continue
    y0, y1 = int(m.group(1)), int(m.group(2))
    years = [y + 1 for y in range(y0, y1 + 1)]          # year_end convention
    assert len(years) == 3, (l, years)
    for l2 in lines[i + 1:i + 160]:
        if re.search(r'Area\s*(?:and|&)\s*Production of', l2): break
        mm = re.match(r'\s*\d+\s+([A-Za-z][A-Za-z ]*?)\s{2,}(.*)$', l2)
        if not mm: continue
        d = dist(mm.group(1))
        if not d: continue
        v = []
        for x in re.findall(r'\d[\d,]*\.?\.?\d*', mm.group(2)):
            try: v.append(float(x.replace(',', '').replace('..', '.')))
            except ValueError: pass
        if len(v) >= 6:
            for k, y in enumerate(years):
                out.append(dict(district=d, year_end=y, area_acre=v[2 * k], production_t=v[2 * k + 1]))
    break
print(json.dumps(out))
