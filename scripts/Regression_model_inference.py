import pandas as pd
import torch
import argparse
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
import sys
from torch.utils.data import Dataset, DataLoader, Sampler
import warnings
warnings.filterwarnings('ignore')

class CustomDataset(Dataset):
    def __init__(self, data, labels,chrs,positions,readnames):
        self.data = data
        self.labels = labels
        self.chrs = chrs
        self.positions = positions
        self.readnames = readnames
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx],self.chrs[idx],self.positions[idx],self.readnames[idx]


class SinglePositionSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.positions = dataset.positions
        self.indices_by_position = self._indices_by_position()

    def _indices_by_position(self):
        #
        indices_by_position = {}
        for i, position in enumerate(self.positions):
            if position not in indices_by_position:
                indices_by_position[position] = []
            indices_by_position[position].append(i)
        return indices_by_position

    def __iter__(self):
        all_positions = list(self.indices_by_position.keys())
        np.random.shuffle(all_positions)  #

        # 为每个位置生成一个batch
        for position in all_positions:
            batch_indices = self.indices_by_position[position]
            np.random.shuffle(batch_indices)  #
            yield batch_indices

    def __len__(self):
        return len(self.indices_by_position)
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        K = self.keys(x)
        Q = self.queries(x)
        V = self.values(x)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.embed_size ** 0.5
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, V)
        return out
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
def data_loader_process(feature_folder,device):
    feature_file = os.path.join(feature_folder, f"classification_inference_result.feather")
    data = pd.read_feather(feature_file)

    contains_only_na = data['normalized_current'].apply(lambda x: any(np.isnan(val) for val in x))
    data = data[~contains_only_na]
    data = data[data['pre_label'].values == 1]
    col_y = data.get('mod_ratio', None)
    if col_y is None:
        data['mod_ratio'] = 0
    position_counts = data.groupby('position')['read_name'].count()
    positions_with_enough_reads = position_counts[position_counts > 10].index
    data = data[data['position'].isin(positions_with_enough_reads)]


    X_test, y_test = data['normalized_current'].values, data['mod_ratio'].values
    chr_list = data['contig'].values.tolist()
    position_list = data['position'].values.tolist()
    read_name_list = data['read_name'].values.tolist()
    X_test = torch.tensor(X_test.tolist()).unsqueeze(1).to(device=device, dtype=torch.float)
    y_test = torch.tensor(y_test).to(device=device, dtype=torch.float)
    #
    test_dataset = CustomDataset(X_test, y_test,chr_list,position_list,read_name_list)
    test_sampler = SinglePositionSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
    return test_loader
def inference(test_loader,model_path,device):
    print("Model inference on " + str(device))
    file = os.path.join(model_path, 'RegressionModel')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("The model file not exist")
        exit()
    model = RegressionModel().to(device)

    state = torch.load(file, map_location=device)
    model.load_state_dict(state)
    model.eval()
    pre_value = []
    chr_list = []
    read_name_list = []
    position_list = []
    count = 0
    predictions = []
    with torch.no_grad():
        for inputs, label,chr,position,read_name in test_loader:
            inputs= inputs.to(device)
            outputs = model(inputs)
            count += 1
            predictions.extend([outputs.mean(dim=0, keepdim=True).cpu().tolist()])
            for i in range(len(label)):
                pre_value.append([outputs.mean(dim=0, keepdim=True).cpu().tolist()][0][0][0][0])
                chr_list.append(chr[i])
                position_list.append(position[i].tolist())
                read_name_list.append(read_name[i])
    return chr_list, position_list, read_name_list, pre_value
def add_the_predict_value(chr_list,position_list,read_name_list,pre_label,feature_file,output_file):
    data = pd.read_feather(feature_file)
    contains_only_na = data['normalized_current'].apply(lambda x: any(np.isnan(val) for val in x))
    data = data[~contains_only_na]
    predicted_df = pd.DataFrame({
        'contig': chr_list,
        'position': position_list,
        'read_name': read_name_list,
        'pre_value': pre_label
    })
    save_file = output_file
    data_with_label = data.merge(predicted_df, how='left', on=['contig', 'position', 'read_name'])
    data_with_label.dropna().reset_index(drop=True).to_feather(
        save_file)
    print("Inference results have saved at " + str(save_file))
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
    check_suffix_and_continue(args.feature_file)
    check_suffix_and_continue(args.output_file)
    test_loader=data_loader_process(args.feature_file,args.device)
    chr_list, position_list, read_name_list, pre_value=inference(test_loader,args.model_saved_folder,args.device)
    add_the_predict_value(chr_list, position_list, read_name_list, pre_value, args.feature_file, args.output_file)
def parse_args():
    parser = argparse.ArgumentParser(description='Regression model inference')
    parser.add_argument('-f', '--feature_file',type=str,
                        help='the folder of the feature extraction output', metavar="character")
    parser.add_argument('-m', '--model_saved_folder', type=str,
                        help='folder of the model saved', metavar="character")
    parser.add_argument('-o', '--output_file', type=str,
                        help='inference result save file', metavar="character")
    parser.add_argument('-d',"--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu).")
    return parser.parse_args()
if __name__ == '__main__':
    main()