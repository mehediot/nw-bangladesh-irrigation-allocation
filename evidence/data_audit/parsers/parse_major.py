import re, sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from parse_crops import dist
lines = open(sys.argv[1], encoding='utf-8', errors='replace').read().split('\n')
PAT = {'Boro Rice': r'Estimates? of Total Boro', 'Wheat': r'Estimates? of Wheat', 'Potato': r'Estimates? of Total Potato'}
out = []
for crop, pat in PAT.items():
    hits = [i for i, l in enumerate(lines) if re.search(pat, l, re.I) and 'Table' in l]
    for i in hits[:1]:
        yrs = re.findall(r'(20\d\d)-(\d\d)', lines[i])
        years = [int(a) + 1 for a, b in yrs][:2]
        for l in lines[i:i + 140]:
            m = re.match(r'\s*\d+\s+([A-Za-z][A-Za-z ]*?)\s{2,}(.*)$', l)
            if not m: continue
            d = dist(m.group(1))
            if not d: continue
            v = [float(x.replace(',', '')) for x in re.findall(r'\d[\d,]*\.?\d*', m.group(2))]
            if len(v) >= 10:
                out.append(dict(crop=crop, district=d, year_end=years[0], area_ha=v[1], production_t=v[4]))
                out.append(dict(crop=crop, district=d, year_end=years[1], area_ha=v[6], production_t=v[9]))
print(json.dumps(out))
