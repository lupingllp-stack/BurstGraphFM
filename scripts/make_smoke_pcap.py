"""Create tiny class-labeled captures for executable pipeline checks, NOT accuracy evidence."""
import argparse
import json
import socket
import struct
from pathlib import Path

def packet(src,dst,sport,dport,plen):
    payload=b'X'*plen
    udp=struct.pack('!HHHH',sport,dport,8+plen,0)+payload
    ip=bytearray(20);ip[0]=0x45;struct.pack_into('!H',ip,2,20+len(udp));ip[8]=64;ip[9]=17
    ip[12:16]=socket.inet_aton(src);ip[16:20]=socket.inet_aton(dst)
    eth=b'\x00'*12+b'\x08\x00'
    return eth+bytes(ip)+udp

p=argparse.ArgumentParser();p.add_argument('--out-dir',default='data/smoke');a=p.parse_args()
out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
manifest=[]
for k in range(8):
    f=out/f'{"vpn" if k%2 else "nonvpn"}_{k}.pcap'
    with f.open('wb') as h:
        h.write(struct.pack('<IHHIIII',0xa1b2c3d4,2,4,0,0,65535,1))
        for flow in range(12):
            src=f'10.0.{k}.{flow+1}';dst='172.16.0.2'
            for j in range(9):
                forward=(j//2)%2==0
                wire=packet(src if forward else dst,dst if forward else src,
                            10000+flow if forward else 443,443 if forward else 10000+flow,
                            40+(k%2)*50+(j%3)*20)
                ts=1000+flow*30+j*0.05
                sec=int(ts);usec=int((ts-sec)*1e6)
                h.write(struct.pack('<IIII',sec,usec,len(wire),len(wire)));h.write(wire)
    manifest.append({'path':str(f.resolve()),'label':'vpn' if k%2 else 'nonvpn','group':f.stem})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(out/'manifest.json')
