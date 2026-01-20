import csv
import time
import os
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def update_progress(counter, total_chunks):
    counter.value += 1
    tqdm.write(f"Processed chunk {counter.value}/{total_chunks}")
def read_truth_data(ground_truth):
    truth = pd.read_csv(ground_truth, sep='\t', encoding='utf-8')
    truth['position'] = pd.to_numeric(truth['position'], errors='coerce', downcast='integer')
    return truth


def write_chunk_data(chunk_data, out_file, i):
    if not chunk_data.empty:
        chunk_data.to_feather(f"{out_file}data_processed_{i}.feather")
        print(f'Done {i}')
def combine_data_and_truth(data, truth):
    merged_data = pd.merge(data, truth, on=['contig', 'position'], how='inner')
    return merged_data
def combine_final_data(out_file_folder, final_out_file):
    result = []
    files_list = os.listdir(out_file_folder)
    for file_name in files_list:
        file_path = os.path.join(out_file_folder, file_name)
        if os.path.isfile(file_path):
            data = pd.read_feather(file_path)
            result.append(data)

    if result:
        final_data = pd.concat(result).reset_index(drop=True)
        final_data.to_feather(f"{final_out_file}data_processed.feather")
        print('Done saving data combined with ground truth')
def write_raw_data(f5c_file,chunksize, out_file, final_out_file,ground_truth):
    reader = pd.read_csv(f5c_file, on_bad_lines='skip', sep='\t', quoting=csv.QUOTE_NONE, header=0, chunksize=chunksize)
    truth_data = read_truth_data(ground_truth)
    if not os.path.exists(out_file):
        os.makedirs(out_file)
        print("New folder is created")
    else:
        print("The folder is already have")
    i = 0
    for data in reader:
        #data['contig'] = data['contig'].str.split('.').str[0]
        merged_data = combine_data_and_truth(data, truth_data)
        write_chunk_data(merged_data, out_file, i)
        i += 1

    print('Done saving sub data combined with ground truth')
    combine_final_data(out_file, final_out_file)
def main():
    # 解析命令行参数
    args = parse_args()
    write_raw_data(args.f5c_file, args.chunk, args.out_file_tem_folder, args.final_out_file, args.ground_truth)
    print("Done")
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Align with the label file')
    parser.add_argument('-f', '--f5c_file', type=str,
                        help='f5c_file file from Nanopolish', metavar="character")
    parser.add_argument('-t', '--out_file_tem_folder', type=str,
                        help='folder of the template files', metavar="character")
    parser.add_argument('-o', '--final_out_file', type=str,
                        help='final folder of the out_file', metavar="character")
    parser.add_argument('-c', '--chunk', default=10000000, type=int,
                        help='chunk size', metavar="integer")
    parser.add_argument('-g', '--ground_truth', type=str,
                        help='groundtruth file', metavar="character")
    return parser.parse_args()
if __name__ == "__main__":
    main()
