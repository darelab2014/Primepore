import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Parameter
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 32))
        self.decoder = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 32))
        self.model = nn.Sequential(self.encoder, self.decoder)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.model(x)
        return x
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=2, hidden=4, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            ).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        return t_dist
class single_molecule_model(nn.Module):
    def __init__(self, n_clusters=2, autoencoder=None, hidden=4, cluster_centers=None, alpha=1.0):
        super(single_molecule_model, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = autoencoder
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)
    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
    def forward(self, x):
        x = self.autoencoder.encode(x)
        return self.clusteringlayer(x)
def compute_editing_loss_function(edit_level,out):
    count_sum=len(out)
    positive_label=int(edit_level*count_sum)
    negative_label=count_sum-positive_label
    pre_positive = np.count_nonzero(out == 1)
    pre_negative = np.count_nonzero(out == 0)
    if (edit_level>0.5 and pre_positive<pre_negative) or (edit_level<0.5 and pre_negative<pre_positive):
        pre_positive = np.count_nonzero(out == 0)
        pre_negative = np.count_nonzero(out == 1)
    loss=(pre_positive/count_sum-edit_level)**2
    return loss

def pretrain(data, model
                  , num_epochs,device):
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-5)
    train_loader = DataLoader(dataset=data,
                              batch_size=128,
                              shuffle=True)
    for epoch in range(0, num_epochs):
        for data in train_loader:
            data = data.float().to(device)
            # ===================forward=====================
            output = model(data)
            output = output.squeeze(1)
            output = output.view(output.size(0), 32)
            loss = nn.MSELoss()(output, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
def train(data, pred_mod_ratio,position, model, num_epochs,device):
    features = []
    train_loader = DataLoader(dataset=data,
                              batch_size=128,
                              shuffle=False)
    for i, batch in enumerate(train_loader):
        dat = batch.float().to(device)
        features.append(model.autoencoder.encode(dat).detach().cpu())
    features = torch.cat(features)
    # ============K-means=======================================
    kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
    cluster_centers = kmeans.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
    model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
    # =========================================================
    y_pred = kmeans.predict(features)
    count0 = np.count_nonzero(y_pred == 0)
    count1 = np.count_nonzero(y_pred == 1)
    loss_function = nn.KLDivLoss(size_average=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    row = []
    for epoch in range(0, num_epochs):
        batch = torch.tensor(data)
        dat = batch.float().to(device)
        output = model(dat)
        target = model.target_distribution(output).detach()
        out = output.argmax(1)
        features = model.autoencoder.encode(dat).detach().cpu()
        # ============K-means=======================================
        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        # =========================================================
        y_pred = kmeans.predict(features)
        kld_loss = 0.01*loss_function(output.log(), target) / output.shape[0]
        editing_loss = compute_editing_loss_function(pred_mod_ratio, y_pred)
        loss = editing_loss-kld_loss
        # loss =  editing_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        features = []
        for i, batch in enumerate(train_loader):
            dat = batch.float().to(device)
            features.append(model.autoencoder.encode(dat).detach().cpu())
        features = torch.cat(features)
        # ============K-means=======================================
        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        cluster_centers = kmeans.cluster_centers_
        cluster_centers = torch.tensor(cluster_centers, dtype=torch.float).cuda()
        model.clusteringlayer.cluster_centers = torch.nn.Parameter(cluster_centers)
        # =========================================================
        y_pred = kmeans.predict(features)
        pre_positive = np.count_nonzero(y_pred == 1)
        pre_negative = np.count_nonzero(y_pred == 0)
        if (pred_mod_ratio > 0.5 and pre_positive < pre_negative) or (pred_mod_ratio < 0.5 and pre_negative < pre_positive):
            pre_positive = np.count_nonzero(y_pred == 0)
            pre_negative = np.count_nonzero(y_pred == 1)
            for i in range(len(y_pred)):
                if y_pred[i] == 0:
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
        row.append([abs(pre_positive / len(y_pred) - pred_mod_ratio),y_pred,cluster_centers])
    y_pred_1=row[np.argmin([item[0] for item in row])][1]
    return y_pred_1
def run_single_molecule(feature_file,epochs,device,output_file):
    model = single_molecule_model(n_clusters=2, autoencoder=AutoEncoder().to(device), hidden=4, cluster_centers=None, alpha=1.0).to(device)
    data = pd.read_feather(feature_file)
    contains_only_na = data['normalized_current'].apply(lambda x: any(np.isnan(val) for val in x))
    data = data[~contains_only_na]
    data['normalized_current'] = data['normalized_current'].apply(lambda x: ','.join(map(str, x)))
    data['single_molecule_label'] = data['y']
    position = data['position'].unique()
    for i in tqdm(range(len(position))):
        datass = data[data['position'] == position[i]]
        dataxx = np.array(datass['normalized_current'].values.tolist())
        dataxx=np.array([[float(x) for x in s.split(',')] for s in dataxx], dtype=float)
        pred_mod_ratio = float(datass['pre_value'].unique()[0])
        compute_position = position[i]
        pretrain(data=dataxx, model=AutoEncoder().to(device), num_epochs=epochs,device=device)
        y_pred = train(data=dataxx, pred_mod_ratio=pred_mod_ratio,
                       position=compute_position, model=model, num_epochs=epochs,device=device)
        data.loc[data['position'] == position[i], 'single_molecule_label'] = y_pred

    # data['normalized_current'] = data['normalized_current'].apply(lambda x: ','.join(map(str, x)))
    save_file = output_file
    data.dropna().reset_index(drop=True).to_feather(
        save_file)
    print("Single molecule results have saved at " + str(save_file))
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
    check_suffix_and_continue(args.feature_file)
    check_suffix_and_continue(args.output_file)
    run_single_molecule(args.feature_file,args.epochs,args.device,args.output_file)
def parse_args():
    parser = argparse.ArgumentParser(description='Single molecule results',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--feature_file',type=str,
                        help='the file of the feature extraction output', metavar="character")
    parser.add_argument('-o', '--output_file',type=str,
                        help='single molecule result save file', metavar="character")
    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help='model training epochs', metavar="integer")
    parser.add_argument('-d',"--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu).")
    return parser.parse_args()
if __name__ == '__main__':
    main()