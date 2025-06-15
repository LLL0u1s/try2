import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from opacus import PrivacyEngine  # 新增
from opacus.validators import ModuleValidator  # 新增

# -------- 自定义自注意力层 --------
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

# -------- 数据准备函数 --------
def prepare_and_save_data():
    print("读取原始数据...")
    data_train = pd.read_csv("KDDTrain+.txt")

    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'outcome', 'level'])
    data_train.columns = columns

    data_train.loc[data_train['outcome'] == "normal", "outcome"] = 0
    data_train.loc[data_train['outcome'] != 0, "outcome"] = 1
    data_train['outcome'] = data_train['outcome'].astype(int)

    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_guest_login', 'level', 'outcome']

    df_num = data_train.drop(cat_cols, axis=1)
    num_cols = df_num.columns

    print("归一化数值特征...")
    scaler = RobustScaler()
    scaled_df = scaler.fit_transform(df_num)
    scaled_df = pd.DataFrame(scaled_df, columns=num_cols)

    data_train.drop(labels=num_cols, axis=1, inplace=True)
    data_train[num_cols] = scaled_df[num_cols]

    print("One-hot编码类别特征...")
    data_train = pd.get_dummies(data_train, columns=['protocol_type', 'service', 'flag'])

    print("准备X、y矩阵...")
    X = data_train.drop(['outcome', 'level'], axis=1).values
    y = data_train['outcome'].values

    print("PCA降维到20维...")
    pca = PCA(n_components=20)
    X_reduced = pca.fit_transform(X)

    val_split = int(0.2 * X_reduced.shape[0])
    train_split = (X_reduced.shape[0] - val_split) // 2

    val_x, val_y = X_reduced[:val_split], y[:val_split]
    alice_x, alice_y = X_reduced[val_split:val_split + train_split], y[val_split:val_split + train_split]
    bob_x, bob_y = X_reduced[val_split + train_split:], y[val_split + train_split:]

    print("保存数据为 numpy 文件...")
    np.savez("data_val.npz", x=val_x, y=val_y)
    np.savez("data_alice.npz", x=alice_x, y=alice_y)
    np.savez("data_bob.npz", x=bob_x, y=bob_y)

    print("数据准备完成！")

# -------- 加载数据 --------
def load_data(client_name):
    if client_name == "alice":
        data = np.load("data_alice.npz")
    elif client_name == "bob":
        data = np.load("data_bob.npz")
    else:
        raise ValueError("客户端名称必须为 'alice' 或 'bob'")

    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def load_val_data():
    data = np.load("data_val.npz")
    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=False)

# -------- Flower 客户端 --------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCELoss()

        self.privacy_engine = PrivacyEngine()

        self.model = ModuleValidator.fix(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            target_epsilon=5.0,
            target_delta=1e-5,
            epochs=1,
            max_grad_norm=1.0
        )
        print(f"差分隐私已启用，目标 ε = 5.0，δ = 1e-5")

    def get_parameters(self):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        params = [torch.tensor(p).to(self.device) for p in parameters]
        for p, new_p in zip(self.model.parameters(), params):
            p.data = new_p.data.clone()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(x)
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

        # -------- 修复点 --------
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        print(f"当前隐私预算: ε = {epsilon:.2f}, δ = 1e-5")

        return self.get_parameters(), len(self.train_loader.dataset), {"epsilon": epsilon}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_total = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x)
                loss = self.criterion(preds, y)
                loss_total += loss.item() * y.size(0)
                predicted = (preds > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)
        return loss_total / total, total, {"accuracy": correct / total}

# -------- 主程序 --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_name", type=str, required=True, help="alice or bob")
    parser.add_argument("--server_address", type=str, default="localhost:8080", help="Flower server address")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prepare_and_save_data()

    train_loader = load_data(args.client_name)
    val_loader = load_val_data()

    model = PTModel(input_dim=20)

    client = FlowerClient(model, train_loader, val_loader, device)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()
