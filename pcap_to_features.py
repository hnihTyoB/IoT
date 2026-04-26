"""
PCAP to IoT Features Converter
==============================
Converts raw .pcap traffic captures into the 17-feature CSV format 
compatible with the UNSW IoT identification model.

Requirements: pip install scapy pandas numpy
"""

import pandas as pd
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
from collections import defaultdict
from pathlib import Path
import time
import argparse

# The 17 features expected by our model
FEATURE_COLUMNS = [
    "srcNumPackets", "dstNumPackets", "srcPayloadSize", "dstPayloadSize",
    "srcAvgPayloadSize", "dstAvgPayloadSize", "srcMaxPayloadSize", "dstMaxPayloadSize",
    "srcStdDevPayloadSize", "dstStdDevPayloadSize", "flowDuration",
    "srcAvgInterarrivalTime", "dstAvgInterarrivalTime", "avgInterarrivalTime",
    "srcStdDevInterarrivalTime", "dstStdDevInterarrivalTime", "stdDevInterarrivalTime"
]

def pcap_to_csv(pcap_path, output_csv):
    print(f"📖 Reading PCAP: {pcap_path} ... (this may take a while)")
    try:
        packets = rdpcap(str(pcap_path))
    except Exception as e:
        print(f"❌ Error reading PCAP: {e}")
        return

    # Group by Flow (5-tuple: src, dst, sport, dport, proto)
    # We treat bidirectional traffic as a single flow for behavior analysis
    flows = defaultdict(list)
    
    for pkt in packets:
        if IP in pkt:
            src = pkt[IP].src
            dst = pkt[IP].dst
            proto = pkt[IP].proto
            sport = pkt.sport if (TCP in pkt or UDP in pkt) else 0
            dport = pkt.dport if (TCP in pkt or UDP in pkt) else 0
            
            # Key is sorted to ensure bidirectional packets fall into the same bucket
            key = tuple(sorted((src, dst))) + (sport, dport, proto)
            flows[key].append((pkt, src)) # Store packet and who sent it

    features_list = []
    print(f"📊 Processing {len(flows)} detected flows...")

    for key, pkt_data in flows.items():
        if len(pkt_data) < 2: continue # Need at least 2 packets for timing
        
        # Split into src and dst directions based on the first packet's sender
        first_sender = pkt_data[0][1]
        src_pkts = [p[0] for p in pkt_data if p[1] == first_sender]
        dst_pkts = [p[0] for p in pkt_data if p[1] != first_sender]
        
        all_pkts = [p[0] for p in pkt_data]
        all_ts = [float(p.time) for p in all_pkts]
        all_inter = np.diff(all_ts) if len(all_ts) > 1 else [0]
        
        src_ts = [float(p.time) for p in src_pkts]
        src_inter = np.diff(src_ts) if len(src_ts) > 1 else [0]
        src_payloads = [len(p.payload) for p in src_pkts]
        
        dst_ts = [float(p.time) for p in dst_pkts]
        dst_inter = np.diff(dst_ts) if len(dst_ts) > 1 else [0]
        dst_payloads = [len(p.payload) for p in dst_pkts] if dst_pkts else [0]

        feat = {
            "srcNumPackets": len(src_pkts),
            "dstNumPackets": len(dst_pkts),
            "srcPayloadSize": sum(src_payloads),
            "dstPayloadSize": sum(dst_payloads),
            "srcAvgPayloadSize": np.mean(src_payloads),
            "dstAvgPayloadSize": np.mean(dst_payloads),
            "srcMaxPayloadSize": np.max(src_payloads),
            "dstMaxPayloadSize": np.max(dst_payloads),
            "srcStdDevPayloadSize": np.std(src_payloads),
            "dstStdDevPayloadSize": np.std(dst_payloads),
            "flowDuration": all_ts[-1] - all_ts[0],
            "srcAvgInterarrivalTime": np.mean(src_inter),
            "dstAvgInterarrivalTime": np.mean(dst_inter),
            "avgInterarrivalTime": np.mean(all_inter),
            "srcStdDevInterarrivalTime": np.std(src_inter),
            "dstStdDevInterarrivalTime": np.std(dst_inter),
            "stdDevInterarrivalTime": np.std(all_inter)
        }
        features_list.append(feat)

    df = pd.DataFrame(features_list)
    # Ensure columns match training order
    df = df[FEATURE_COLUMNS]
    df.to_csv(output_csv, index=False)
    print(f"✅ Success! Generated {len(df)} flow windows in: {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pcap_to_features.py <input.pcap> <output.csv>")
    else:
        pcap_to_csv(sys.argv[1], sys.argv[2])
