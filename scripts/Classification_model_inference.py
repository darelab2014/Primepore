from functools import partial
import pandas as pd
import argparse
import torch
from torch import nn
import numpy as np
import os
import sys
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class CustomDataset(Dataset):
    def __init__(self, data, labels,chrs,positions,readnames):
        self.data = data
        self.labels = labels
        self.chrs = chrs
        self.positions = positions
        self.readnames= readnames
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx],self.chrs[idx],self.positions[idx],self.readnames[idx]
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=128)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x=x.flatten(2).transpose(1, 2)
        return x
class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class CostSensitiveLoss(nn.Module):
    def __init__(self, weights):
        super(CostSensitiveLoss, self).__init__()
        self.weights = weights
    def forward(self, predictions, targets):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Compute the loss using weighted cross-entropy
        loss = nn.CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))(predictions, targets)
        return loss
class Transformer1D(nn.Module):
    def __init__(self, num_classes=3,
                 embed_dim=128, depth=6, num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0.3, drop_path_ratio=0.3,  norm_layer=None,
                 act_layer=None):
        super(Transformer1D, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = network()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, 17, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 33, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 16, 512]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
class CostSensitiveLoss(nn.Module):
    def __init__(self, weights):
        super(CostSensitiveLoss, self).__init__()
        self.weights = weights
    def forward(self, output, target):
        loss = nn.CrossEntropyLoss(weight=torch.Tensor(self.weights).to(output.device))
        return loss(output, target)
class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, padding=2, dilation=1, stride=1):
        super().__init__()
        if bn:
            self.basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 7, padding=padding, dilation=dilation, stride=stride),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        else:
            self.basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 5, padding=padding, dilation=dilation, stride=stride),
                nn.ReLU()
            )
    def forward(self, x):
        return self.basic_block(x)
def accuracy(logits, y_true):
    _, indices = torch.max(logits, 1)
    correct_samples = torch.sum(indices == y_true)
    return float(correct_samples) / y_true.shape[0]
def data_loader_process(feature_file,batch_size,device):
    data = pd.read_feather(feature_file)
    contains_only_na = data['normalized_current'].apply(lambda x: any(np.isnan(val) for val in x))
    data = data[~contains_only_na]
    col_y = data.get('y', None)
    if col_y is None:
        data['y'] = 0
    data['y'] = data['y'].astype(float)
    data['y'] = data['y'].astype(int)
    normalized_current = data['normalized_current'].values.tolist()
    chr_list = data['contig'].values.tolist()
    position_list = data['position'].values.tolist()
    read_name_list = data['read_name'].values.tolist()
    label_list = [x for x in data['y'].values]
    cost_weights = [label_list.count(1) / label_list.count(0), label_list.count(1) / label_list.count(1)]
    X_test = torch.tensor(normalized_current).unsqueeze(1).to(device=device, dtype=torch.float)
    y_test = torch.tensor(label_list).to(device=device, dtype=torch.int64)
    test_dataset = CustomDataset(X_test, y_test,chr_list,position_list,read_name_list)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return test_loader,cost_weights
def inference(test_loader,model_path,device):
    print("Model inference on "+str(device))
    file = os.path.join(model_path, 'Transformer1D')
    if not os.path.exists(file):
        print("The trained model file not exist")
        exit()
    model = Transformer1D(num_classes=2).to(device)
    state = torch.load(file, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    score_accum = 0
    count = 0
    true_label = []
    pre_label = []
    pre_label1 = []
    chr_list = []
    read_name_list = []
    position_list = []
    for x, y,chr,position,read_name in test_loader:
        logits = model(x.to(device))
        _, indices = torch.max(logits, 1)
        count += 1
        score_accum += accuracy(logits, y.to(device))
        for i in range(len(y)):
            true_label.append(int(y[i]))
            pre_label.append(logits[i].tolist())
            pre_label1.append(int(indices[i]))
            chr_list.append(chr[i])
            position_list.append(position[i].tolist())
            read_name_list.append(read_name[i])
    # acc=float(score_accum / count)
    return chr_list,position_list,read_name_list,pre_label1
def add_the_predict_label(chr_list,position_list,read_name_list,pre_label,feature_file,output_folder):
    data = pd.read_feather(feature_file)
    contains_only_na = data['normalized_current'].apply(lambda x: any(np.isnan(val) for val in x))
    data = data[~contains_only_na]
    predicted_df = pd.DataFrame({
        'contig': chr_list,
        'position': position_list,
        'read_name': read_name_list,
        'pre_label': pre_label
    })
    save_file=os.path.join(output_folder,'classification_inference_result.feather')
    data_with_label = data.merge(predicted_df, how='left', on=['contig','position','read_name'])
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
    args = parse_args()
    check_suffix_and_continue(args.feature_file)
    test_loader,cost_weights=data_loader_process(args.feature_file,args.batch_size,args.device)
    chr_list,position_list,read_name_list,pre_label1=inference(test_loader,args.model_saved_folder,args.device)
    add_the_predict_label(chr_list,position_list,read_name_list,pre_label1, args.feature_file,args.output_folder)
def parse_args():
    parser = argparse.ArgumentParser(description='Classification model inference')
    parser.add_argument('-f', '--feature_file',type=str,
                        help='the file of the feature extraction output', metavar="character")
    parser.add_argument('-m', '--model_saved_folder',type=str,
                        help='folder of the model saved', metavar="character")
    parser.add_argument('-o', '--output_folder', type=str,
                        help='inference result save folder', metavar="character")
    parser.add_argument('-b', '--batch_size', default=512, type=int,
                        help='model training batch size', metavar="integer")
    parser.add_argument('-d',"--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu).")
    return parser.parse_args()
if __name__ == '__main__':
    main()

