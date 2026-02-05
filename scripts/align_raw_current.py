from read5 import read
import os
import sys
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
from multiprocessing import Pool, Manager
from functools import partial
from tqdm import tqdm
import warnings
import glob
warnings.filterwarnings('ignore')

def update_progress(counter, total_chunks):
    counter.value += 1
    tqdm.write(f"Processed chunk {counter.value}/{total_chunks}")
def process_chunk(chunks, new_data, folder_path, final_out_file, counter):
    r5 = read(folder_path)
    chunk_filter = new_data[new_data['read_name'].isin(chunks)]
    data_array = []
    for readid, group in chunk_filter.groupby('read_name'):
        signal = r5.getSignal(readid)
        pA_signal = r5.getpASignal(readid)
        norm_signal = r5.getZNormSignal(readid)
        starts = np.array(group['start_idx'], dtype=int)
        ends = np.array(group['end_idx'], dtype=int)
        group['pA_signal'] = [pA_signal[start:end] for start, end in zip(starts, ends)]
        group['signal'] = [signal[start:end] for start, end in zip(starts, ends)]
        group['norm_signal'] = [norm_signal[start:end] for start, end in zip(starts, ends)]
        data_array.append(group)
    del r5
    if data_array:
        data_array = pd.concat(data_array).reset_index(drop=True)
        data_array.drop_duplicates(subset=['contig', 'position', 'read_name', 'event_index'], keep='first', inplace=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        filename = f"{final_out_file}/final_read_chunk_{timestamp}.feather"
        data_array.reset_index(drop=True).to_feather(filename)
        
        update_progress(counter, total_chunks=len(chunks))
def combine_final_read(out_file_folder, final_out_file):
    csv_files = glob.glob(os.path.join(out_file_folder, '*.feather'))
    combined_df = pd.concat([pd.read_feather(file) for file in csv_files])
    combined_df.reset_index(drop=True).to_feather(final_out_file)
    print('Done saving data combined with ground truth')
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
    check_suffix_and_continue(args.align_label_file)
    check_suffix_and_continue(args.final_out_file)
    manager = Manager()
    counter = manager.Value('i', 0)
    if not os.path.exists(args.out_file_tem_folder):
        os.makedirs(args.out_file_tem_folder)
        print("New folder is created")
    else:
        print("The folder is already have")
        for filename in os.listdir(args.out_file_tem_folder):
            file_path = os.path.join(args.out_file_tem_folder, filename)
            try:
                #
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                #
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    new_data = pd.read_feather(args.align_label_file)
    unique_read_ids = new_data['read_name'].unique()
    # chunk_size = int(len(unique_read_ids) / 100)
    #print(np.ceil(len(unique_read_ids) / 10))
    chunk_size = int(len(unique_read_ids) / np.ceil(len(unique_read_ids) / 100))
    chunks = [unique_read_ids[i:i + chunk_size] for i in range(0, len(unique_read_ids), chunk_size)]
    # r5 = read(args.folder)
    # Use multiprocessing to parallelize chunk processing
    pool = Pool()
    func_partial = partial(process_chunk,new_data=new_data, folder_path=args.blow5_file,
                           final_out_file=args.out_file_tem_folder, counter=counter)
    list(pool.imap(func_partial, chunks))
    print("End reading fast5 files...")
    combine_final_read(args.out_file_tem_folder, args.final_out_file)
    print("Done")
def parse_args():
    parser = argparse.ArgumentParser(description='Align with the raw current file')
    parser.add_argument('-b', '--blow5_file', type=str,
                        help='blow5 file', metavar="character")
    parser.add_argument('-t', '--out_file_tem_folder', type=str,
                        help='folder of the template files', metavar="character")
    parser.add_argument('-a', '--align_label_file', type=str,
                        help='the output file of the align_label', metavar="character")
    parser.add_argument('-c', '--chunk', default=10000000, type=int,
                        help='chunk size', metavar="integer")
    parser.add_argument('-o', '--final_out_file', type=str,
                        help='final output file (*.feather)', metavar="character")
    return parser.parse_args()
if __name__ == "__main__":
    main()