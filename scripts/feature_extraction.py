import numpy as np
import os
import sys
import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
def interpolated_array(dat):
    max_length = 16
    interpolated_arrays = []

    for arr in dat:
        if len(arr) <= max_length:
            # Interpolate to max_length if length is less than or equal to max_length
            interpolated_arr = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(arr)), arr)
        else:
            # Sample down to max_length if length is greater than max_length
            indices = np.linspace(0, len(arr) - 1, max_length, dtype=int)
            interpolated_arr = arr[indices]

        interpolated_arrays.append(interpolated_arr)

    return interpolated_arrays
def combined_same_position_same_read_data(dat):
    result = []
    cols_lower = [c.lower() for c in dat.columns]
    has_mod_rate = 'mod_ratio' in cols_lower
    has_y = 'y' in cols_lower
    if has_mod_rate and has_y:
        agg_dict = {
            'normalized_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'interpolated_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_level_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_length': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'standardized_level': lambda x: np.mean(x, axis=0).astype(np.float32),
            'contig': lambda x: x.unique().tolist()[0],
            'reference_kmer': lambda x: x.unique().tolist()[0],
            'y': lambda x: x.unique().tolist()[0],
            'mod_ratio': lambda x: x.unique().tolist()[0]
        }
    elif (has_mod_rate) and (not has_y):
        agg_dict = {
            'normalized_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'interpolated_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_level_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_length': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'standardized_level': lambda x: np.mean(x, axis=0).astype(np.float32),
            'contig': lambda x: x.unique().tolist()[0],
            'reference_kmer': lambda x: x.unique().tolist()[0],
            'mod_ratio': lambda x: x.unique().tolist()[0]
        }
    elif (not has_mod_rate) and (has_y):
        agg_dict = {
            'normalized_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'interpolated_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_level_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_length': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'standardized_level': lambda x: np.mean(x, axis=0).astype(np.float32),
            'contig': lambda x: x.unique().tolist()[0],
            'reference_kmer': lambda x: x.unique().tolist()[0],
            'y': lambda x: x.unique().tolist()[0]
        }
    else:
        agg_dict = {
            'normalized_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'interpolated_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_level_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_length': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'standardized_level': lambda x: np.mean(x, axis=0).astype(np.float32),
            'contig': lambda x: x.unique().tolist()[0],
            'reference_kmer': lambda x: x.unique().tolist()[0]
        }
    grouped_datas=dat.groupby(["position","read_name"]).agg(agg_dict).reset_index()
    result.append(grouped_datas)
    return pd.concat(result)
def process_interpolated_data(dat):
    result = []
    cols_lower = [c.lower() for c in dat.columns]
    has_mod_rate = 'mod_ratio' in cols_lower
    has_y = 'y' in cols_lower
    if has_mod_rate and has_y:
        agg_dict = {
            'read_name': lambda x: x.unique().tolist()[0],
            'interpolated_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'normalized_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_level_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_length': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'standardized_level': lambda x: np.mean(x, axis=0).astype(np.float32),
            'contig': lambda x: x.unique().tolist()[0],
            'reference_kmer': lambda x: x.unique().tolist()[0],
            'y': lambda x: x.unique().tolist()[0],
            'mod_ratio': lambda x: x.unique().tolist()[0],
            'percentile_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'std_current': lambda x: np.std(x.values).astype(np.float32),
            'var_current': lambda x: np.var(x.values).astype(np.float32),
        }
    elif (has_mod_rate) and (not has_y):
        agg_dict = {
            'read_name': lambda x: x.unique().tolist()[0],
            'interpolated_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'normalized_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_level_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_length': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'standardized_level': lambda x: np.mean(x, axis=0).astype(np.float32),
            'contig': lambda x: x.unique().tolist()[0],
            'reference_kmer': lambda x: x.unique().tolist()[0],
            'mod_ratio': lambda x: x.unique().tolist()[0],
            'percentile_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'std_current': lambda x: np.std(x.values).astype(np.float32),
            'var_current': lambda x: np.var(x.values).astype(np.float32),
        }
    elif (not has_mod_rate) and (has_y):
        agg_dict = {
            'read_name': lambda x: x.unique().tolist()[0],
            'interpolated_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'normalized_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_level_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_length': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'standardized_level': lambda x: np.mean(x, axis=0).astype(np.float32),
            'contig': lambda x: x.unique().tolist()[0],
            'reference_kmer': lambda x: x.unique().tolist()[0],
            'y': lambda x: x.unique().tolist()[0],
            'percentile_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'std_current': lambda x: np.std(x.values).astype(np.float32),
            'var_current': lambda x: np.var(x.values).astype(np.float32),
        }
    else:
        agg_dict = {
            'read_name': lambda x: x.unique().tolist()[0],
            'interpolated_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'normalized_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_level_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'event_length': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_mean': lambda x: np.mean(x, axis=0).astype(np.float32),
            'model_stdv': lambda x: np.mean(x, axis=0).astype(np.float32),
            'standardized_level': lambda x: np.mean(x, axis=0).astype(np.float32),
            'contig': lambda x: x.unique().tolist()[0],
            'reference_kmer': lambda x: x.unique().tolist()[0],
            'percentile_current': lambda x: np.mean(x, axis=0).astype(np.float32),
            'std_current': lambda x: np.std(x.values).astype(np.float32),
            'var_current': lambda x: np.var(x.values).astype(np.float32),
        }
    grouped_datas=dat.groupby(["position"]).agg(agg_dict).reset_index()
    result.append(grouped_datas)
    return pd.concat(result)
def process_data(file,num_parts,template_folder):
    s_data = pd.read_feather(file)
    unique_positions = s_data['position'].unique()
    position_slices = np.array_split(unique_positions, num_parts)
    if not os.path.exists(template_folder):
        os.makedirs(template_folder)
        print("New folder is created")
    else:
        print("The folder is already have")
        for filename in os.listdir(template_folder):
            file_path = os.path.join(template_folder, filename)
            try:
                #
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                #
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    # Iterate over slices and extract corresponding data
    for s, position_slice in enumerate(position_slices):
        data = s_data[s_data['position'].isin(position_slice)]
        X_data = data.groupby('position')
        interpolated_list = []
        interpolated_indices = []
        for _, group in X_data:
            indices = group.index
            interpolated_indices.extend(indices)
            interpolated_list.extend(interpolated_array(group['pA_signal'].values))
        data['interpolated_current'] = np.nan
        interpolated_series = pd.Series(interpolated_list, index=interpolated_indices)
        data['interpolated_current'] = interpolated_series
        data['normalized_current'] = np.nan
        cols_lower = [c.lower() for c in data.columns]
        has_mod_rate = 'mod_ratio' in cols_lower
        has_y = 'y' in cols_lower
        if has_mod_rate and has_y:
            col=20
        elif has_mod_rate or has_y:
            col=19
        else:
            col=18
        new_x = pd.Series([abs((x[col] - x[10])) for x in data.values], index=data.index)
        data['normalized_current'] = new_x
        new_x = pd.Series(
            [(inner_array - inner_array.min()) / (inner_array.max() - inner_array.min()) for inner_array in
             data['normalized_current'].values], index=data.index)
        data['normalized_current'] = new_x
        ####### same positon same read combine
        combined_data = combined_same_position_same_read_data(data)
        num_rows = len(combined_data)
        # Calculate the chunk size for each part
        chunk_size = num_rows // num_parts
        # Split the data into parts and save each part separately
        for i in range(num_parts):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_parts - 1 else num_rows  # For the last chunk
            # Get the chunk of data
            data_chunk = combined_data.iloc[start_idx:end_idx]
            # Save the chunk to a separate feather file
            file_name = os.path.join(template_folder, f"{s}_part_{i + 1}.feather")
            data_chunk.reset_index(drop=True).to_feather(file_name)
def combined_data(num_parts,template_folder,final_out_file):
    result = []
    for i in range(num_parts):
        for j in range(num_parts):
            file_name = os.path.join(template_folder, f"{i}_part_{j + 1}.feather")
            data = pd.read_feather(file_name)
            result.append(data)
    combined_data = pd.concat(result).reset_index(drop=True)
    combined_data["percentile_current"] = combined_data["normalized_current"]
    combined_data["std_current"] = combined_data["normalized_current"]
    combined_data["var_current"] = combined_data["normalized_current"]
    combined_data1 = combined_data[combined_data['y'] == 1]
    combined_data0 = combined_data[combined_data['y'] != 1]
    processed_data = process_interpolated_data(combined_data0)
    processed_data = pd.concat([combined_data1, processed_data])
    #### add 11 features
    processed_data['mean'] = processed_data['interpolated_current'].apply(
        lambda x: np.mean(x) if len(x) > 0 else np.nan)
    processed_data['median'] = processed_data['interpolated_current'].apply(
        lambda x: np.median(x) if len(x) > 0 else np.nan)
    from scipy import stats
    processed_data['min'] = processed_data['interpolated_current'].apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
    processed_data['max'] = processed_data['interpolated_current'].apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
    processed_data['range'] = processed_data['interpolated_current'].apply(
        lambda x: np.ptp(x) if len(x) > 0 else np.nan)
    processed_data['q1'] = processed_data['interpolated_current'].apply(
        lambda x: np.percentile(x, 25) if len(x) > 0 else np.nan)
    processed_data['q3'] = processed_data['interpolated_current'].apply(
        lambda x: np.percentile(x, 75) if len(x) > 0 else np.nan)
    processed_data['iqr'] = processed_data['q3'] - processed_data['q1']
    processed_data['std_dev'] = processed_data['interpolated_current'].apply(
        lambda x: np.std(x) if len(x) > 0 else np.nan)
    processed_data['skewness'] = processed_data['interpolated_current'].apply(
        lambda x: stats.skew(x) if len(x) > 0 else np.nan)
    # processed_data['kurt'] = processed_data['interpolated_current'].apply(lambda x: stats.kurt(x) if len(x) > 0 else np.nan)
    # normalized the feature
    processed_data['mean'] = (processed_data['mean'] - processed_data['mean'].min()) / (
                processed_data['mean'].max() - processed_data['mean'].min())
    processed_data['median'] = (processed_data['median'] - processed_data['median'].min()) / (
                processed_data['median'].max() - processed_data['median'].min())
    processed_data['min'] = (processed_data['min'] - processed_data['min'].min()) / (
                processed_data['min'].max() - processed_data['min'].min())
    processed_data['max'] = (processed_data['max'] - processed_data['max'].min()) / (
                processed_data['max'].max() - processed_data['max'].min())
    processed_data['range'] = (processed_data['range'] - processed_data['range'].min()) / (
                processed_data['range'].max() - processed_data['range'].min())
    processed_data['q1'] = (processed_data['q1'] - processed_data['q1'].min()) / (
                processed_data['q1'].max() - processed_data['q1'].min())
    processed_data['q3'] = (processed_data['q3'] - processed_data['q3'].min()) / (
                processed_data['q3'].max() - processed_data['q3'].min())
    processed_data['iqr'] = (processed_data['iqr'] - processed_data['iqr'].min()) / (
                processed_data['iqr'].max() - processed_data['iqr'].min())
    processed_data['std_dev'] = (processed_data['std_dev'] - processed_data['std_dev'].min()) / (
                processed_data['std_dev'].max() - processed_data['std_dev'].min())
    processed_data['skewness'] = (processed_data['skewness'] - processed_data['skewness'].min()) / (
                processed_data['skewness'].max() - processed_data['skewness'].min())
    # processed_data['kurt'] = (processed_data['kurt'] - processed_data['kurt'].min()) / (processed_data['kurt'].max() - processed_data['kurt'].min())
    processed_data['event_level_mean'] = (processed_data['event_level_mean'] - processed_data[
        'event_level_mean'].min()) / (processed_data['event_level_mean'].max() - processed_data[
        'event_level_mean'].min())
    processed_data['event_stdv'] = (processed_data['event_stdv'] - processed_data['event_stdv'].min()) / (
                processed_data['event_stdv'].max() - processed_data['event_stdv'].min())
    processed_data['event_length'] = (processed_data['event_length'] - processed_data['event_length'].min()) / (
                processed_data['event_length'].max() - processed_data['event_length'].min())
    processed_data['model_mean'] = (processed_data['model_mean'] - processed_data['model_mean'].min()) / (
                processed_data['model_mean'].max() - processed_data['model_mean'].min())
    processed_data['model_stdv'] = (processed_data['model_stdv'] - processed_data['model_stdv'].min()) / (
                processed_data['model_stdv'].max() - processed_data['model_stdv'].min())
    processed_data['standardized_level'] = (processed_data['standardized_level'] - processed_data[
        'standardized_level'].min()) / (processed_data['standardized_level'].max() - processed_data[
        'standardized_level'].min())

    def append_values_to_array(row):
        # 从当前行抓取所有需要添加到数组的值
        values_to_append = np.array([
            row['mean'],
            row['median'],
            row['min'],
            row['max'],
            row['range'],
            row['q1'],
            row['q3'],
            row['iqr'],
            row['std_dev'],
            row['skewness'],
            row['event_level_mean'],
            row['event_stdv'],
            row['event_length'],
            row['model_mean'],
            row['model_stdv'],
            row['standardized_level']
        ])
        # 添加值到 normalized_current 数组，如果 normalized_current 是空的，则创建一个新数组

        return np.concatenate((row['normalized_current'], values_to_append))

    processed_data['normalized_current'] = processed_data.apply(append_values_to_array, axis=1)
    # output_file_name= os.path.join(os.path.dirname(os.path.abspath(template_folder)), "output_feature.feather")
    processed_data.dropna().reset_index(drop=True).to_feather(
        final_out_file)
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
    # 解析命令行参数
    args = parse_args()
    check_suffix_and_continue(args.align_raw_current_file)
    check_suffix_and_continue(args.final_out_file)
    process_data(args.align_raw_current_file,args.chunk,args.out_file_tem_folder)
    combined_data(args.chunk,args.out_file_tem_folder,args.final_out_file)
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Feature extraction')
    parser.add_argument('-a', '--align_raw_current_file',type=str,
                        help='the output file of the align_raw_current', metavar="character")
    parser.add_argument('-t', '--out_file_tem_folder',type=str,
                        help='folder of the template files', metavar="character")
    parser.add_argument('-c', '--chunk', default=2, type=int,
                        help='chunk size', metavar="integer")
    parser.add_argument('-o', '--final_out_file', type=str,
                         help='final output file (*.feather)', metavar="character")
    return parser.parse_args()
if __name__ == "__main__":
    main()