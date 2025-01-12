from scapy.all import rdpcap, IP, TCP, UDP
import networkx as nx
import os
import shutil
import csv
import time
import networkx as nx
import random

def convert_port(port_str):
    port_str = port_str.strip()
    if port_str.startswith('0x') or port_str.startswith('0X'):
        return int(port_str, 16)
    elif port_str.isdigit():
        return int(port_str)
    else:
        return None


def read_sessions_from_csv(csv_file):
    sessions = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            label = row['label']
            if(label != 'BENIGN'):
                continue
            src_ip = row['srcip']
            src_port = convert_port(row['sport'])
            dst_ip = row['dstip']
            dst_port = convert_port(row['dsport'])
            if(src_port is None or dst_port is None):
                continue
            protocol = row['protocol_m'].lower()
            if protocol == 'tcp':
                protocol_num = 6
            elif protocol == 'udp':
                protocol_num = 17
            elif protocol == '6' or protocol == '17':
                protocol_num = int(protocol)
            else:
                continue
            src_tuple = (src_ip, src_port, protocol_num)
            dst_tuple = (dst_ip, dst_port, protocol_num)

            if src_tuple not in sessions:
                sessions[src_tuple] = set()
            sessions[src_tuple].add(dst_tuple)
    return sessions


def read_sessions(pcap_file):
    packets = rdpcap(pcap_file)
    sessions = {}
    for packet in packets:
        if packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP)):
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            src_port = packet[TCP].sport if packet.haslayer(TCP) else packet[UDP].sport
            dst_port = packet[TCP].dport if packet.haslayer(TCP) else packet[UDP].dport
            protocol = packet[IP].proto

            src_tuple = (src_ip, src_port, protocol)
            dst_tuple = (dst_ip, dst_port, protocol)

            if src_tuple not in sessions:
                sessions[src_tuple] = set()
            sessions[src_tuple].add(dst_tuple)
        return sessions


def build_and_eliminate_graph(G):
    degrees = dict(G.degree())
    for node in list(G.nodes()):
        sip, port, o = node
        if degrees[node] == 1:
            continue
        neighbors = list(G.neighbors(node))
        if all(neighbor in degrees and degrees[neighbor] >= degrees[node] for neighbor in neighbors):
            keep_neighbor = neighbors[0]
            for neighbor in neighbors[1:]:
                G.remove_edge(node, neighbor)
            for neighbor in neighbors[1:]:
                src_ip, src_port, protocol = node
                new_src_port = random.randint(49152, 65535)
                new_node = (src_ip, new_src_port, protocol)
                G.add_node(new_node)
                G.add_edge(new_node, neighbor)
    return G


def save_sessions(pcap_dir, G, min_degree):
    degrees = dict(G.degree())
    for filename in os.listdir(pcap_dir):
        if filename.endswith(".pcap"):
            pcap_file = os.path.join(pcap_dir, filename)
            sessions = read_sessions(pcap_file)
            for src, dsts in sessions.items():
                if dsts:
                    dst = next(iter(dsts))
                else:
                    dst = None
                if (src in degrees and degrees[src] > min_degree): 
                    folder_name = f"session_{src[0]}_{src[1]}_{src[2]}_{degrees[src]}"
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    shutil.copy2(pcap_file, os.path.join(folder_name, os.path.basename(pcap_file)))
                for dst in dsts:
                    if (dst in degrees and degrees[dst] > min_degree):
                        folder_name = f"session_{dst[0]}_{dst[1]}_{dst[2]}_{degrees[dst]}"
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        shutil.copy2(pcap_file, os.path.join(folder_name, os.path.basename(pcap_file)))
                        break

def build_graph_from_pcaps(pcap_dir):
    G = nx.Graph()
    for filename in os.listdir(pcap_dir):
        if filename.endswith(".pcap"):
            pcap_file = os.path.join(pcap_dir, filename)
            sessions = read_sessions(pcap_file)
            if len(sessions) > 0:
                for src, dsts in sessions.items():
                    if not G.has_node(src):
                        G.add_node(src)
                    for dst in dsts:
                        if not G.has_node(dst):
                            G.add_node(dst)
                        G.add_edge(src, dst)
    return G

def build_graph_from_csvs(csv_dir):
    G = nx.Graph()
    for filename in os.listdir(csv_dir):
        if filename.endswith("done.csv"):
            print(f'now processing csv file {filename}')
            csv_file = os.path.join(csv_dir, filename)
            sessions = read_sessions_from_csv(csv_file)
            if len(sessions) > 0:
                for src, dsts in sessions.items():
                    if not G.has_node(src):
                        G.add_node(src)
                    for dst in dsts:
                        if not G.has_node(dst):
                            G.add_node(dst)
                        G.add_edge(src, dst)
    return G


def main():
    pcap_dir = "/mnt/Benign"
    csv_dir = "/mnt2/tuple"
    triple_result_dir = "/mnt2/triple.csv"
    print("now processing the csv file to build graph")
    start_time = time.time()
    G= build_graph_from_pcaps(pcap_dir)
    end_time = time.time()
    print(f'time for build graph is {end_time-start_time}')
    start_time = time.time()
    G = build_and_eliminate_graph(G)
    end_time = time.time()
    print(f'time for clip graph is {end_time-start_time}')
    start_time = time.time()
    min_degree = 10
    
    with open(triple_result_dir, 'w', newline='') as csvfile:
        degrees = dict(G.degree())
        writer = csv.writer(csvfile)
        writer.writerow(['ip', 'port', 'protocol', 'degree'])
        for node, degree in degrees.items():
            if degree > min_degree:
                writer.writerow([node[0], node[1], node[2], degree])
    end_time = time.time()
    print(f'time for writing triple file is {end_time-start_time}')


if __name__ == "__main__":
    main()

