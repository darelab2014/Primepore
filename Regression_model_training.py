from torch.optim import AdamW
import pandas as pd
import torch
import argparse
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader, Sampler
import warnings
warnings.filterwarnings('ignore')

class CustomDataset(Dataset):
    def __init__(self, data, labels, positions):
        self.data = data
        self.labels = labels
        self.positions = positions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.positions[idx]


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

        #
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
class SimpleRegressionModel(nn.Module):
    def __init__(self):
        super(SimpleRegressionModel, self).__init__()
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
def data_loader_process(feature_folder,processed_data_template_floder,device):
    feature_file = os.path.join(feature_folder, f"inference_result.feather")
    data = pd.read_feather(feature_file)
    if not os.path.exists(processed_data_template_floder):
        os.makedirs(processed_data_template_floder)
        print("The processed data template folder is created")
    else:
        print("The processed data template folder is already have")
        for filename in os.listdir(processed_data_template_floder):
            file_path = os.path.join(processed_data_template_floder, filename)
            try:
                #
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                #
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    ####
    contains_only_na = data['normalized_current'].apply(lambda x: any(np.isnan(val) for val in x))
    data = data[~contains_only_na]
    data = data[data['pre_label'].values == 1]
    position_counts = data.groupby('position')['read_name'].count()
    positions_with_enough_reads = position_counts[position_counts > 10].index
    data = data[data['position'].isin(positions_with_enough_reads)]

    Pos=data['position'].values
    unique_positions = np.unique(Pos)
    positions_count = len(unique_positions)
    train_count = int(positions_count * 0.8)
    np.random.shuffle(unique_positions)
    train_positions = unique_positions[:train_count]
    val_positions = unique_positions[train_count:]
    train_data = data[data['position'].isin(train_positions)]
    val_data = data[data['position'].isin(val_positions)]
    train_data['normalized_current'] = train_data['normalized_current'].apply(lambda x: ','.join(map(str, x)))
    val_data['normalized_current'] = val_data['normalized_current'].apply(lambda x: ','.join(map(str, x)))
    train_data.to_csv(
        os.path.join(processed_data_template_floder,"train_data.csv"),index=False)
    val_data.to_csv(
        os.path.join(processed_data_template_floder,"val_data.csv"),index=False)
    ####
    train_data['normalized_current'] = train_data['normalized_current'].apply(
        lambda x: np.fromstring(x, sep=','))
    val_data['normalized_current'] = val_data['normalized_current'].apply(
        lambda x: np.fromstring(x, sep=','))
    X_train, y_train, Pos_train = train_data['normalized_current'].values, train_data['mod_ratio'].values, \
    train_data['position'].values
    X_val, y_val, Pos_val = val_data['normalized_current'].values, val_data['mod_ratio'].values, val_data[
        'position'].values
    X_train = torch.tensor(X_train.tolist()).unsqueeze(1).to(device=device, dtype=torch.float)
    y_train = torch.tensor(y_train).to(device=device, dtype=torch.float)
    X_val = torch.tensor(X_val.tolist()).unsqueeze(1).to(device=device, dtype=torch.float)
    y_val = torch.tensor(y_val).to(device=device, dtype=torch.float)
    #
    train_dataset = CustomDataset(X_train, y_train, Pos_train)
    train_sampler = SinglePositionSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_dataset = CustomDataset(X_val, y_val, Pos_val)
    val_sampler = SinglePositionSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
    return train_loader,val_loader
def train(train_loader,val_loader,epochs,model_path,device):
    print("Model training on " + str(device))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("The model folder is created")
    else:
        print("The model folder is already have")
        for filename in os.listdir(model_path):
            file_path = os.path.join(model_path, filename)
            try:
                #
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                #
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    model = SimpleRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_history = []
    val_loss_history = []
    val_rmse_history = []
    best_val_loss = float('inf')
    best_val_rmse = float('inf')
    for epoch in range(epochs):  #
        model.train()
        train_loss_accum = 0
        count = 0
        predictions = []
        targets = []
        for inputs, labels, positions in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            target_ratio = torch.full_like(outputs, float(labels.unique()[0]))
            loss = criterion(outputs.mean(), target_ratio)
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss_accum += loss.item()
            count += 1
            predictions.extend([outputs.mean(dim=0, keepdim=True).cpu().tolist()])
            targets.extend([labels.unique()[0].cpu().tolist()])
        loss_history.append(train_loss_accum / count)
        # Validation
        model.eval()
        val_loss_accum = 0
        val_scores = []
        count = 0
        with torch.no_grad():
            for inputs, labels, positions in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                target_ratio = torch.full_like(outputs, float(labels.unique()[0]))
                val_loss = criterion(outputs.mean(), target_ratio).cpu()
                val_loss_accum += val_loss
                val_scores.append(val_loss)
                count += 1
        val_loss = val_loss_accum / count
        val_loss_history.append(val_loss)
        val_rmse = np.sqrt(np.mean(val_scores))
        val_rmse_history.append(val_rmse)
        # Print results
        print(
            f'Epoch #{epoch}, train loss: {loss_history[-1]:.4f}, '
            f'val loss: {val_loss:.4f}, val RMSE: {val_rmse:.4f}'
        )
        # Update best scores if needed
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            # Save model if this is your best validation score so far
            torch.save(model.state_dict(), os.path.join(model_path, model.__class__.__name__))
def main():
    #
    args = parse_args()
    train_loader,val_loader=data_loader_process(args.feature_folder,args.processed_data_template_floder,args.device)
    train(train_loader,val_loader,args.epochs,args.model_saved_folder,args.device)
def parse_args():
    parser = argparse.ArgumentParser(description='Regression model training')
    parser.add_argument('-f', '--feature_folder',type=str,
                        help='the folder of the feature extraction output', metavar="character")
    parser.add_argument('-t', '--processed_data_template_floder', type=str,
                        help='template folder of the processed data', metavar="character")
    parser.add_argument('-m', '--model_saved_folder', type=str,
                        help='folder of the model saved', metavar="character")
    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help='model training epochs', metavar="integer")
    parser.add_argument('-d',"--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu).")
    return parser.parse_args()
if __name__ == '__main__':
    main()