import re, sys, json
NW = {'Rajshahi':'Rajshahi','Naogoan':'Naogaon','Naogaon':'Naogaon','Natore':'Natore','Chapai':'Chapainawabganj','Chapainawabganj':'Chapainawabganj',
      'Nawabganj':'Chapainawabganj','Gaibandah':'Gaibandha','Naogaon':'Naogaon','Chapainababganj':'Chapainawabganj','Pabna':'Pabna','Sirajganj':'Sirajganj','Bogura':'Bogura','Bogra':'Bogura','Joypurhat':'Joypurhat',
      'Rangpur':'Rangpur','Gaibandha':'Gaibandha','Kurigram':'Kurigram','Lalmonirhat':'Lalmonirhat','Nilphamari':'Nilphamari',
      'Dinajpur':'Dinajpur','Thakurgaon':'Thakurgaon','Panchagarh':'Panchagarh'}
COLS = ['total','forest','not_available','culturable_waste','current_fallow','single','double','triple','quadruple','net_cropped','gross_cropped']
txt = open(sys.argv[1], encoding='utf-8', errors='replace').read().split('\n')
start = [i for i, l in enumerate(txt) if re.search(r'Table:?\s*5\.1\b', l)]
end = [i for i, l in enumerate(txt) if re.search(r'Table:?\s*5\.2\b', l) and i > (start[0] if start else 0)]
out = []
if start:
    sec = txt[start[0]: end[0] if end else start[0] + 900]
    year = None
    for l in sec:
        m = re.search(r'Year\s*:?\s*(20\d\d-\d\d)', l)
        if m: year = m.group(1); continue
        m = re.match(r'\s*\d+\s+([A-Za-z]+)(?:\s+[A-Za-z]+)?\s+([\d\s.]+)$', l)
        if m and m.group(1) in NW and 'Division' not in l:
            nums = [float(x) for x in m.group(2).split()]
            if len(nums) == 11:
                out.append(dict(year=year, district=NW[m.group(1)], **dict(zip(COLS, nums))))
print(json.dumps(out))
