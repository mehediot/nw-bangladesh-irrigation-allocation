import re, sys, json
NWV = {'rajshahi':'Rajshahi','naogaon':'Naogaon','naogoan':'Naogaon','natore':'Natore','pabna':'Pabna','sirajganj':'Sirajganj',
       'bogura':'Bogura','bogra':'Bogura','joypurhat':'Joypurhat','rangpur':'Rangpur','gaibandha':'Gaibandha','gaibandah':'Gaibandha',
       'kurigram':'Kurigram','lalmonirhat':'Lalmonirhat','nilphamari':'Nilphamari','dinajpur':'Dinajpur','thakurgaon':'Thakurgaon',
       'panchagarh':'Panchagarh','panchagar':'Panchagarh','tahkurgaon':'Thakurgaon','thakurgoan':'Thakurgaon','lalmonirhaat':'Lalmonirhat','nilphamary':'Nilphamari','joypurhaat':'Joypurhat','sirajgonj':'Sirajganj','natore ':'Natore'}
def dist(name):
    k = name.lower().replace(' ', '')
    if k.startswith('chapai') or 'nawabganj' in k or 'nawbganj' in k: return 'Chapainawabganj'
    return NWV.get(k)
if __name__ == "__main__":
    lines = open(sys.argv[1], encoding='utf-8', errors='replace').read().split('\n')
    heads = [(i, re.search(r'Area\s*(?:and|&)\s*Production of\s+(.+?)\s+by\s+District', l, re.I)) for i, l in enumerate(lines)]
    heads = [(i, m.group(1).strip()) for i, m in heads if m]
    out = []
    for k, (i, crop) in enumerate(heads):
        j = heads[k + 1][0] if k + 1 < len(heads) else i + 400
        yrs = re.findall(r'(20\d\d-\d\d)', ' '.join(lines[i:i + 4]))
        for l in lines[i:j]:
            m = re.match(r'\s*\d+\s+([A-Za-z][A-Za-z ]*?)\s{2,}(.*)$', l)
            if not m: continue
            d = dist(m.group(1))
            if not d: continue
            nums = re.findall(r'\d[\d,]*\.?\.?\d*', m.group(2))
            vals = []
            for x in nums:
                x = x.replace(',', '').replace('..', '.')
                try: vals.append(float(x))
                except ValueError: pass
            if len(vals) >= 6:
                out.append(dict(table_line=i + 1, crop=crop, years=yrs[:3], district=d, areas_acre=[vals[0], vals[2], vals[4]]))
    print(json.dumps(out))
    