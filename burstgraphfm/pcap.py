"""Minimal stdlib PCAP reader for Ethernet IPv4 TCP/UDP."""
import gzip
import ipaddress
import struct
from pathlib import Path

def iter_packets(path):
    path=Path(path)
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as fh:
        magic=fh.read(4)
        endian={'\xd4\xc3\xb2\xa1':'<','\xa1\xb2\xc3\xd4':'>','\x4d\x3c\xb2\xa1':'<','\xa1\xb2\x3c\x4d':'>'}.get(magic.decode('latin1'))
        if endian is None:
            raise ValueError(f'Unsupported capture format: {path}; convert PCAPNG to PCAP first')
        nano=magic in (b'\x4d\x3c\xb2\xa1',b'\xa1\xb2\x3c\x4d')
        header=fh.read(20)
        if len(header)!=20:raise ValueError('Truncated PCAP header')
        link=struct.unpack(endian+'HHIIII',header)[-1]
        if link!=1:raise ValueError(f'Only Ethernet DLT=1 supported; got {link}')
        while True:
            h=fh.read(16)
            if not h:break
            if len(h)!=16:break
            sec,sub,incl,_=struct.unpack(endian+'IIII',h)
            if incl>32*1024*1024:raise ValueError('Suspicious record length')
            raw=fh.read(incl)
            if len(raw)!=incl:break
            try:
                if len(raw)<34:continue
                eth_type=int.from_bytes(raw[12:14],'big');off=14
                if eth_type in (0x8100,0x88a8):
                    if len(raw)<18:continue
                    eth_type=int.from_bytes(raw[16:18],'big');off=18
                if eth_type==0x0800:
                    if len(raw)<off+20:continue
                    ihl=(raw[off]&15)*4
                    if ihl<20 or len(raw)<off+ihl+4:continue
                    frag=int.from_bytes(raw[off+6:off+8],'big')
                    if frag&0x3fff:continue
                    proto=raw[off+9];src=str(ipaddress.ip_address(raw[off+12:off+16]));dst=str(ipaddress.ip_address(raw[off+16:off+20]));off+=ihl
                elif eth_type==0x86dd:
                    if len(raw)<off+44:continue
                    proto=raw[off+6];src=str(ipaddress.ip_address(raw[off+8:off+24]));dst=str(ipaddress.ip_address(raw[off+24:off+40]));off+=40
                else:continue
                if proto not in (6,17):continue
                sport,dport=struct.unpack('!HH',raw[off:off+4])
                yield sec+sub/(1e9 if nano else 1e6),(src,sport),(dst,dport),proto,len(raw)
            except (ValueError,IndexError,struct.error):continue
