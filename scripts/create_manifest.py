"""Explicit capture-level labels: filename inference ONLY under user-approved rules."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--raw-dir',required=True)
p.add_argument('--output',default='data/manifest.json')
p.add_argument('--label-mode',choices=['vpn_binary','directory'],required=True)
a=p.parse_args()
files=sorted([p for p in Path(a.raw_dir).rglob('*') if p.is_file() and (p.suffix.lower() in ('.pcap','.pcapng') or p.name.lower().endswith(('.pcap.gz','.pcapng.gz')))])
rows=[]
for f in files:
    if a.label_mode=='vpn_binary':
        if f.name.lower().startswith('vpn_') or f.name.lower().startswith('vpn-'):label='vpn'
        else:label='nonvpn'
    else:label=f.parent.name
    rows.append({'path':str(f.resolve()),'label':label,'group':f.stem})
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2))
print('Wrote',out,'captures',len(rows),'labels',sorted(set(r['label'] for r in rows)))
print('IMPORTANT: verify each path,label,group manually before processing. Filenames are NOT authoritative labels.')
