"""Parse district area and production of lentil and rape/mustard from one BBS yearbook text file.
Usage: python -I parse_minor.py <yearbook.txt> <edition_year>   (prints JSON)"""
import json
import re
import sys

NWV = {'rajshahi': 'Rajshahi', 'naogaon': 'Naogaon', 'naogoan': 'Naogaon', 'natore': 'Natore', 'pabna': 'Pabna',
       'sirajganj': 'Sirajganj', 'sirajgonj': 'Sirajganj', 'bogura': 'Bogura', 'bogra': 'Bogura', 'joypurhat': 'Joypurhat',
       'joypurhaat': 'Joypurhat', 'rangpur': 'Rangpur', 'gaibandha': 'Gaibandha', 'gaibandah': 'Gaibandha',
       'kurigram': 'Kurigram', 'lalmonirhat': 'Lalmonirhat', 'lalmonirhaat': 'Lalmonirhat', 'nilphamari': 'Nilphamari',
       'nilphamary': 'Nilphamari', 'dinajpur': 'Dinajpur', 'thakurgaon': 'Thakurgaon', 'thakurgoan': 'Thakurgaon',
       'tahkurgaon': 'Thakurgaon', 'panchagarh': 'Panchagarh', 'panchagar': 'Panchagarh'}


def dist(name):
    k = name.lower().replace(' ', '')
    if k.startswith('chapai') or 'nawabganj' in k or 'nawbganj' in k:
        return 'Chapainawabganj'
    return NWV.get(k)


CROPS = {'Lentil': r'Lentil\s*\(?Masur', 'Mustard': r'Rape\s*(?:and|&)\s*Must'}
path, edition = sys.argv[1], int(sys.argv[2])
lines = open(path, encoding='utf-8', errors='replace').read().split('\n')
out = []
for crop, pat in CROPS.items():
    for i, l in enumerate(lines):
        if not (re.search(r'Area\s*(?:and|&)\s*Production\s+of\s+' + pat, l, re.I) and re.search(r'by\s+District', l, re.I)):
            continue
        head = ' '.join(lines[i:i + 3])
        rng = re.findall(r'(20\d\d)\s*-\s*(?:20)?\d\d', head)
        if len(rng) < 2:
            continue
        y0, y1 = int(rng[0]), int(rng[-1])
        years = [y + 1 for y in range(y0, y1 + 1)]
        if len(years) != 3:
            continue
        for l2 in lines[i + 1:i + 200]:
            if re.search(r'Area\s*(?:and|&)\s*Production\s+of', l2, re.I):
                break
            m = re.match(r'\s*\d+\s+([A-Za-z][A-Za-z ]*?)\s{2,}(.*)$', l2)
            if not m:
                continue
            d = dist(m.group(1))
            if not d:
                continue
            v = []
            for x in re.findall(r'\d[\d,]*\.?\.?\d*', m.group(2)):
                try:
                    v.append(float(x.replace(',', '').replace('..', '.')))
                except ValueError:
                    pass
            if len(v) >= 6:
                for k, y in enumerate(years):
                    out.append(dict(crop=crop, edition=edition, line=i + 1, district=d, year_end=y,
                                    area_acre=v[2 * k], production_t=v[2 * k + 1]))
        break
print(json.dumps(out))
