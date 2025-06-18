import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import argparse

# -------- 自注意力层 --------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output

# -------- 模型定义 --------
class PTModel(nn.Module):
    def __init__(self, input_dim=20):
        super(PTModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(0.4)
        self.self_attention = SelfAttention(128)
        self.fc3 = nn.Linear(128, 512)
        self.dropout3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(512, 128)
        self.dropout4 = nn.Dropout(0.4)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = x.unsqueeze(1)
        x = self.self_attention(x)
        x = x.squeeze(1)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc5(x))
        return x

def load_test_data():
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'outcome', 'level'
    ]
    data_test = pd.read_csv("datasets/KDDTest+.txt", header=None, names=columns)
    # outcome 列先转为二分类
    data_test.loc[data_test['outcome'] == "normal", "outcome"] = 0
    data_test.loc[data_test['outcome'] != 0, "outcome"] = 1
    data_test['outcome'] = data_test['outcome'].astype(int)
    # 先提取标签
    y = data_test['outcome'].values
    # 再做后续特征处理
    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_guest_login', 'level', 'outcome']
    df_num = data_test.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaler = joblib.load("datasets/scaler.pkl")
    pca = joblib.load("datasets/pca.pkl")
    scaled_df = scaler.transform(df_num)
    scaled_df = pd.DataFrame(scaled_df, columns=num_cols)
    data_test.drop(labels=num_cols, axis=1, inplace=True)
    data_test[num_cols] = scaled_df[num_cols]
    data_test = pd.get_dummies(data_test, columns=['protocol_type', 'service', 'flag'])
    all_features_columns = np.load("datasets/all_features_columns.npy", allow_pickle=True)
    for col in all_features_columns:
        if col not in data_test.columns:
            data_test[col] = 0
    data_test = data_test[all_features_columns]
    X = data_test.values
    X_reduced = pca.transform(X)
    x_tensor = torch.tensor(X_reduced, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=32, shuffle=False)

def evaluate(model_path, device):
    model = PTModel(input_dim=20).to(device)
    state_dict = torch.load(model_path, map_location=device)
    # 去除 _module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_module."):
            new_state_dict[k[8:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # load_test_data()为KDDTest+.txt测试集，load_val_data()为KDDTrain+.txt数据集前20%
    # test_loader = load_test_data()
    from pt_client import load_val_data
    test_loader = load_val_data()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            preds = torch.clamp(preds, min=0.0, max=1.0)
            predicted = (preds > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)
    print(f"测试集准确率: {correct / total:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="模型权重文件路径，如 client_model_alice.pth")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args.model_path, device)