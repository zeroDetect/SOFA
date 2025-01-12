import os
import argparse
import copy
import pandas as pd
import numpy as np
import subprocess
from utils_feature import check_path, get_fields_list
import scapy.all as scapy
from scapy.all import rdpcap, Ether, IP, RandMAC, RandIP,RandShort,TCP,UDP
import threading
MAXNUM_PACKETS = 16
MAXBYTE_PER_PACKET = 256
TIME_INTERVAL_INDEX = 1
PACKET_LEN_INDEX = 2
RANDOM_ADDRESS = 0


def merge_pcap_in_dir(pcap_dir, pcap_files, data_name):
    if os.path.exists(os.path.join(pcap_dir, data_name+'.pcap')):
        print('[INFO] Pcap already exists.')
        return

    pcap_files = [os.path.join(pcap_dir, f) for f in pcap_files]
    in_files = ' '.join(pcap_files)
    cmd = 'mergecap -w {0}.pcap {1}'.format(os.path.join(pcap_dir, data_name), in_files)
    ret = os.system(cmd)
    if ret == 0:
        print('[DONE] Merge {0} pcap files.'.format(len(pcap_files)))
    else:
        print('[ERROR] Merge pcap files.')
        exit(1)


def ip_to_float(ip):
    parts = ip.split('.')
    numeric_ip = (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    return float(numeric_ip)


def convert_pcap_to_csv(pcap_path, csv_dir, tshark_path, fields_file, server_ip):
    stats_original = {
        'ip.proto':0, 
        'packet_num':0,
        'duration':0,
        'ip.flags.df':0,
        'ip.flags.mf':0,
        'ip.frag.offset':0,
        'tcp.flags.ack':0,
        'tcp.flags.push':0,
        'tcp.flags.reset':0,
        'tcp.flags.urg':0,
        'tcp.flags.syn':0,
        'tcp.flags.fin':0,
        'max_frame_length': 0,
        'min_frame_length': 0,
        'mean_frame_length': 0, 
        'std_frame_length': 0,
        'max_time_delta':0,
        'min_time_delta':0,
        'mean_time_delta':0,
        'std_time_delta':0,
        'max_calculated_window_size':0,
        'min_calculated_window_size':0,
        'mean_calculated_window_size':0,
        'std_calculated_window_size':0,
        'max_scale_factor':0,
        'min_scale_factor':0,
        'mean_scale_factor':0,
        'std_scale_factor':0,
        'max_window': 0, 
        'min_window': 0, 
        'mean_window': 0, 
        'std_window': 0, 
        'max_ttl': 0,
        'min_ttl': 0,
        'mean_ttl': 0,
        'std_ttl': 0,
    }
    with open(fields_file, 'r') as f:
        fields = f.readlines()
    fields = [line.strip('\n') for line in fields]
    fields = ['-e '+line for line in fields]
    fields = ' '.join(fields)
    data_frame = []
    i = 0
    all_features = np.zeros((len(pcap_path), 17, MAXBYTE_PER_PACKET), dtype='float') 
    Y = []
    for i, pcap_file in enumerate(pcap_path):
        csv_name = os.path.join(csv_dir, os.path.basename(os.path.dirname(pcap_path[0])) + '.csv')
        if(i % 1000 == 0):
            print(f'now processing the {i} files\n')
        tshark_command = ' '.join([tshark_path, '-T', 'fields', fields, '-r', pcap_file, '-E', 'header=y',\
                                   '-E', 'occurrence=f', '-E', 'separator=,'])
        sh_name = os.path.basename(csv_name) + '.sh'
        with open(sh_name, "w") as f:
            f.write("#!/bin/sh\n")
            f.write(tshark_command)
            f.write("\n")
        subprocess.run(["chmod", "+x", sh_name])
        try:
            result = subprocess.run("./"+sh_name, stdout=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the script: {e}")
        try:
            os.remove(sh_name)
        except OSError as e:
            print(f"Error deleting the script file: {e}")
        sessions = result.stdout.strip().split('\n')
        headers = sessions[0].split(',')
        data_rows = [row.split(',') for row in sessions[1:]]
        df = pd.DataFrame(data_rows, columns=headers)
        if(df['ip.src'].iloc[0] == '8.8.8.8' or df['ip.dst'].iloc[0]=='8.8.8.8'):
            continue
        if df['ip.proto'].iloc[0] == '6': 
            df.rename(columns={'tcp.srcport': 'srcport', 'tcp.dstport': 'dstport'}, inplace=True)
            df.drop(columns=['udp.srcport', 'udp.dstport'], inplace=True)
        elif df['ip.proto'].iloc[0] == '17': 
            df.rename(columns={'udp.srcport': 'srcport', 'udp.dstport': 'dstport'}, inplace=True)
            df.drop(columns=['tcp.srcport', 'tcp.dstport'], inplace=True)
        else:
            print(f"Unsupported protocol: {df['ip.proto'].iloc[0]}")
            continue
        df = df.apply(lambda col: col.apply(lambda x: ','.join(['0' if sub_x == '' else sub_x for sub_x in x.split(',')])))
        df = df.apply(lambda col: col.fillna('0'))
        if(server_ip is not None):
            if(df['ip.src'].iloc[0] == server_ip):
                src_ip = df['ip.dst'].iloc[0]
                dst_ip = df['ip.src'].iloc[0]
            else:
                src_ip = df['ip.src'].iloc[0]
                dst_ip = df['ip.dst'].iloc[0]
        else:
            if(df['srcport'].iloc[0] < df['dstport'].iloc[0]) :
                src_ip = df['ip.dst'].iloc[0]
                dst_ip = df['ip.src'].iloc[0]
            else:
                src_ip = df['ip.src'].iloc[0]
                dst_ip = df['ip.dst'].iloc[0]
        if(df['ip.proto'].iloc[0] != '6' and df['ip.proto'].iloc[0] != '17'):
            continue
        stats = copy.deepcopy(stats_original)
        stats_in = copy.deepcopy(stats_original)
        stats_out = copy.deepcopy(stats_original)
        filtered_in_sessions = df[(df['ip.src'] == src_ip) & (df['ip.dst'] == dst_ip)]
        filtered_out_sessions = df[(df['ip.src'] == dst_ip) & (df['ip.dst'] == src_ip)]
        if(len(filtered_in_sessions)):
            session_feature_extract(filtered_in_sessions, stats_in)
        if(len(filtered_out_sessions)):
            session_feature_extract(filtered_out_sessions, stats_out)
        if(len(stats)):
            session_feature_extract(df, stats)
        stats_in_prefixed = {f'in_{k}': v for k, v in stats_in.items()}
        stats_out_prefixed = {f'out_{k}': v for k, v in stats_out.items()}
        combined_stats = {**stats, **stats_in_prefixed, **stats_out_prefixed}
        combined_stats.pop('in_ip.proto', None)
        combined_stats.pop('out_ip.proto', None)
        combined_stats = {'ip.src': ip_to_float(df['ip.src'].iloc[0]), 'ip.dst': ip_to_float(df['ip.dst'].iloc[0]), 'srcport': float(df['srcport'].iloc[0]),\
                          'dstport':float(df['dstport'].iloc[0]), **combined_stats}
        combined_stats = {k: '0' if isinstance(v, float) and np.isnan(v) else v for k, v in combined_stats.items()}
        pkts = scapy.rdpcap(pcap_file)
        flow = np.zeros([MAXNUM_PACKETS, MAXBYTE_PER_PACKET], dtype='float')
        j = 0
        for p in pkts:
            if(RANDOM_ADDRESS == 1):
                p[Ether].src = RandMAC()
                p[Ether].dst = RandMAC()
                if p.haslayer(IP):
                    p[IP].src = RandIP()
                    print(p[IP].src)
                    p[IP].dst = RandIP()
                    if p.haslayer(TCP):
                        p[TCP].sport = RandShort()
                        p[TCP].dport = RandShort()
                    if p.haslayer(UDP):
                        p[UDP].sport = RandShort()
                        p[UDP].dport = RandShort()
                rawpkt = bytes(p)[:MAXBYTE_PER_PACKET]
                if len(rawpkt) < MAXBYTE_PER_PACKET:
                    rawpkt += b'\x00' * (MAXBYTE_PER_PACKET - len(rawpkt))
                rawpkt_dec = [int.from_bytes([b], 'big') for b in rawpkt]
                rawpkt_float = np.array(rawpkt_dec, dtype='float')
                print(rawpkt_float)
            else:

                rawpkt = list(p.original[:MAXBYTE_PER_PACKET])
                if len(rawpkt) < MAXBYTE_PER_PACKET:
                    rawpkt = rawpkt + [0] * (MAXBYTE_PER_PACKET - len(rawpkt))
                rawpkt_float = np.array(rawpkt, dtype='float')
            flow[j] = rawpkt_float
            j = j + 1
            if j == MAXNUM_PACKETS:
                break
        stats_array = np.array(list(combined_stats.values()), dtype='float').reshape(1, -1)
        cols_to_pad = 256 - stats_array.shape[1]
        if cols_to_pad > 0:
            stats_array = np.pad(stats_array, ((0, 0), (0, cols_to_pad)), 'constant', constant_values=0)
        else:
            pass
        merged_features = np.vstack((flow, stats_array))
        all_features[i] = merged_features
        if 'benign' in pcap_file.lower():
            Y.append([os.path.basename(os.path.dirname(pcap_file)), 0])
        else:
            Y.append([os.path.basename(os.path.dirname(pcap_file)), 1])
    npz_filename = os.path.join(csv_dir, os.path.basename(os.path.dirname(pcap_path[0])) + '.npz')
    all_features = all_features[:len(Y)]
    np.savez(npz_filename, X=all_features, Y=np.array(Y))  


def session_feature_extract(df, stats):
    stats['ip.proto'] = df['ip.proto'].iloc[0]
    stats['packet_num'] = len(df)
    
    stats['duration'] = df['frame.time_epoch'].astype(float).max() - df['frame.time_epoch'].astype(float).min()

    keys_to_process = list(stats.keys())[3:12]
    for key in keys_to_process:
        if key in df.columns:
            column_data = df[key].astype(float)
            sum_value = column_data.sum()
            stats[key] = sum_value
    stats['max_frame_length'] = df['frame.len'].astype(float).max()
    stats['min_frame_length'] = df['frame.len'].astype(float).min()
    stats['mean_frame_length'] = df['frame.len'].astype(float).mean()
    stats['std_frame_length'] = df['frame.len'].astype(float).std()

    stats['max_time_delta'] = df['frame.time_delta'].astype(float).max()
    stats['min_time_delta'] = df['frame.time_delta'].astype(float).min()
    stats['mean_time_delta'] = df['frame.time_delta'].astype(float).mean()
    stats['std_time_delta'] = df['frame.time_delta'].astype(float).std()

    stats['max_calculated_window_size'] = df['tcp.window_size'].astype(float).max()
    stats['min_calculated_window_size'] = df['tcp.window_size'].astype(float).min()
    stats['mean_calculated_window_size'] = df['tcp.window_size'].astype(float).mean()
    stats['std_calculated_window_size'] = df['tcp.window_size'].astype(float).std()

    stats['max_scale_factor'] = df['tcp.window_size_scalefactor'].astype(float).max()
    stats['min_scale_factor'] = df['tcp.window_size_scalefactor'].astype(float).min()
    stats['mean_scale_factor'] = df['tcp.window_size_scalefactor'].astype(float).mean()
    stats['std_scale_factor'] = df['tcp.window_size_scalefactor'].astype(float).std()

    stats['max_window'] = df['tcp.window_size_value'].astype(float).max()
    stats['min_window'] = df['tcp.window_size_value'].astype(float).min()
    stats['mean_window'] = df['tcp.window_size_value'].astype(float).mean()
    stats['std_window'] = df['tcp.window_size_value'].astype(float).std()

    stats['max_ttl'] = df['ip.ttl'].astype(float).max()
    stats['min_ttl'] = df['ip.ttl'].astype(float).min()
    stats['mean_ttl'] = df['ip.ttl'].astype(float).mean()
    stats['std_ttl'] = df['ip.ttl'].astype(float).std()


def find_pcap_files(root_dir):
    pcap_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.pcap', '.pcapng')):
                pcap_files.append(os.path.join(dirpath, filename))
    pcap_files.sort()
    return pcap_files


def process_pcap_file(file_path, csv_dir, tshark_path, fields_file, server_ip):
    try:
        convert_pcap_to_csv(file_path, csv_dir, tshark_path, fields_file, server_ip)
        print(f"[INFO] Processed {os.path.dirname(file_path[0])}")
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--pcap_dir', type=str, required=False, \
                      default='/mnt/dataset/Malware/Virut')
    args.add_argument('--csv_dir', type=str, required=False, default=\
                      '/mnt/data_USTC')
    args.add_argument('--tshark_path', type=str, required=False, default='tshark')
    args.add_argument('--fields_file', type=str, required=False, \
                      default = 'ids_fields.txt')
    args.add_argument('--server_ip', type=str, required=False, \
                      default =  None)
    args = args.parse_args()
    print(args)

    if not os.path.exists(args.pcap_dir):
        print('[ERROR] pcap dir is not exist!')
        exit(1)
    if not os.path.exists(args.fields_file):
        print('[ERROR] fields file is not exist!')
        exit(1)
    check_path(args.csv_dir)
    pcap_files = find_pcap_files(args.pcap_dir)
    full_paths = [os.path.join(args.pcap_dir, file) for file in pcap_files]
    print('[INFO] Total {0} pcap files'.format(len(pcap_files)))
    pcap_dirs = set(os.path.dirname(path) for path in full_paths)
     
    threads = []
    for dir in pcap_dirs:
        files_in_dir = [file for file in full_paths if os.path.dirname(file) == dir]
        thread = threading.Thread(target=process_pcap_file, args=(files_in_dir, args.csv_dir, args.tshark_path, args.fields_file, args.server_ip))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print("[INFO] All pcap files have been processed.")
