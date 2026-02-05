import csv
import os
import argparse
import pandas as pd
from tqdm import tqdm
import sys
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
        chunk_data.to_feather(f"{out_file}/data_processed_{i}.feather")
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
        final_data.to_feather(final_out_file)
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
def latest_file_in_dir(directory: str) -> str:
    last_mtime = -1
    latest_path = None
    for name in os.listdir(directory):
        full = os.path.join(directory, name)
    if os.path.isfile(full):
        mtime = os.path.getmtime(full)
    if mtime > last_mtime:
        last_mtime = mtime
    latest_path = full
    return latest_path
def check_suffix_and_continue(path: str) -> None:
    if path is None:
        print("No files were found to be checked in the directory.")
        sys.exit(1)
    _, ext = os.path.splitext(path)
    if ext.lower() != ".feather":
        print("The file extension needs to be set to .feather")
        sys.exit(1)
def main():
    args = parse_args()
    check_suffix_and_continue(args.final_out_file)
    write_raw_data(args.f5c_file, args.chunk, args.out_file_tem_folder, args.final_out_file, args.ground_truth)
    print("Done")
def parse_args():
    parser = argparse.ArgumentParser(description='Align with the label file')
    parser.add_argument('-f', '--f5c_file', type=str,
                        help='f5c_file file from Nanopolish', metavar="character")
    parser.add_argument('-t', '--out_file_tem_folder', type=str,
                        help='folder of the template files', metavar="character")
    parser.add_argument('-o', '--final_out_file', type=str,
                        help='final output file (*.feather)', metavar="character")
    parser.add_argument('-c', '--chunk', default=10000000, type=int,
                        help='chunk size', metavar="integer")
    parser.add_argument('-g', '--ground_truth', type=str,
                        help='groundtruth file', metavar="character")
    return parser.parse_args()
if __name__ == "__main__":
    main()
