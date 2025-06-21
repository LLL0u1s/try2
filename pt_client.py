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
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import joblib
import pickle
from phe import paillier
import sys
import traceback
import concurrent.futures
import os

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
    if os.path.exists("datasets/data_alice.npz") and os.path.exists("datasets/data_bob.npz"):
        return
    print("读取原始数据...")
    data_train = pd.read_csv("datasets/KDDTrain+.txt")
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
    np.savez("datasets/data_val.npz", x=val_x, y=val_y)
    np.savez("datasets/data_alice.npz", x=alice_x, y=alice_y)
    np.savez("datasets/data_bob.npz", x=bob_x, y=bob_y)
    print("数据准备完成！")
    joblib.dump(scaler, "datasets/scaler.pkl")
    joblib.dump(pca, "datasets/pca.pkl")
    all_features_columns = list(data_train.drop(['outcome', 'level'], axis=1).columns)
    np.save("datasets/all_features_columns.npy", all_features_columns)

# -------- 加载数据 --------
def load_data(client_name):
    if client_name == "alice":
        data = np.load("datasets/data_alice.npz")
    elif client_name == "bob":
        data = np.load("datasets/data_bob.npz")
    else:
        raise ValueError("客户端名称必须为 'alice' 或 'bob'")
    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def load_val_data():
    data = np.load("datasets/data_val.npz")
    x = torch.tensor(data["x"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=False)

def paillier_encrypt(pubkey, x):
    return pubkey.encrypt(int(x)).ciphertext()

# -------- Flower 客户端 --------
class FlowerClient(fl.client.Client):
    def __init__(self, model, train_loader, val_loader, device, client_name):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.client_name = client_name
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
        self.scale = 1e6  # 量化因子
        self.param_shapes = None
        self._cached_enc_params = None
        self._cached_param_state = None

        # 密钥管理：只有alice生成密钥对，其它客户端只用公钥
        self.pubkey = None
        self.privkey = None
        self.key_file = f"paillier_key_{client_name}.pkl"
        self.pubkey_file = "paillier_pubkey.pkl"

        if client_name == "alice":
            # alice生成密钥对并保存
            if not os.path.exists(self.key_file):
                self.pubkey, self.privkey = paillier.generate_paillier_keypair()
                with open(self.key_file, "wb") as f:
                    pickle.dump((self.pubkey, self.privkey), f)
                with open(self.pubkey_file, "wb") as f:
                    pickle.dump(self.pubkey, f)
            else:
                with open(self.key_file, "rb") as f:
                    self.pubkey, self.privkey = pickle.load(f)
        else:
            # 其它客户端只加载公钥
            while not os.path.exists(self.pubkey_file):
                print("等待公钥文件生成...")
                import time
                time.sleep(1)
            with open(self.pubkey_file, "rb") as f:
                self.pubkey = pickle.load(f)
            self.privkey = None

    def get_parameters(self, ins):
        """
        只加密最后一层参数，其余参数明文传递。
        对最后一层参数做量化和稀疏化（只加密非零参数），并用多线程加速加密。
        """
        try:
            current_state = [val.cpu().detach().numpy().copy() for val in self.model.parameters()]
            if self._cached_param_state is not None and all(
                np.array_equal(a, b) for a, b in zip(self._cached_param_state, current_state)
            ):
                return self._cached_enc_params

            params = current_state
            self.param_shapes = [p.shape for p in params]

            # 1. 只加密最后一层参数，其余参数明文传递
            last_idx = len(params) - 1
            # 量化所有参数
            params_int = [np.round(p * self.scale).astype(np.int64) for p in params]

            # 2. 对最后一层参数做稀疏化（只加密非零参数）
            last_param = params_int[last_idx]
            last_param_flat = last_param.flatten()
            nonzero_indices = np.nonzero(last_param_flat)[0]
            nonzero_values = last_param_flat[nonzero_indices]

            # 3. 并行加密非零参数
            with concurrent.futures.ThreadPoolExecutor() as executor:
                enc_nonzero = list(executor.map(lambda x: paillier_encrypt(self.pubkey, x), nonzero_values))

            # 只加密最后一层的非零参数，其余参数明文传递
            # 明文参数直接序列化
            plain_params = [params_int[i] if i != last_idx else None for i in range(len(params))]
            # 密文参数只存储非零位置和密文
            enc_last_layer = {
                "indices": nonzero_indices,
                "enc_values": enc_nonzero,
                "shape": last_param.shape
            }

            pubkey_bytes = pickle.dumps(self.pubkey)
            result = fl.common.Parameters(tensors=[
                pickle.dumps(plain_params),         # 其余层明文参数（最后一层为None）
                pickle.dumps(enc_last_layer),       # 最后一层密文参数（非零位置和密文）
                pickle.dumps(self.param_shapes),    # 所有层的shape
                pubkey_bytes
            ], tensor_type="mixed")
            self._cached_enc_params = result
            self._cached_param_state = [p.copy() for p in params]
            return result
        except Exception as e:
            print("get_parameters 异常:", e)
            traceback.print_exc()
            sys.exit(1)

    def set_parameters(self, parameters):
        try:
            plain_params = pickle.loads(parameters.tensors[0])
            enc_last_layer = pickle.loads(parameters.tensors[1])
            if not parameters.tensors[2]:
                print("收到的参数 shapes 为空，跳过 set_parameters。")
                return
            shapes = pickle.loads(parameters.tensors[2])
            if len(parameters.tensors) > 3 and parameters.tensors[3]:
                self.pubkey = pickle.loads(parameters.tensors[3])
            # 只有alice有privkey，其它客户端privkey为None
            params = []
            for i, shape in enumerate(shapes):
                if i != len(shapes) - 1:
                    arr = plain_params[i].reshape(shape) / self.scale
                    params.append(torch.tensor(arr, dtype=torch.float32).to(self.device))
                else:
                    arr = np.zeros(np.prod(enc_last_layer["shape"]), dtype=np.float32)
                    if self.privkey is not None:
                        # 只有alice能解密
                        dec = [self.privkey.decrypt(paillier.EncryptedNumber(self.pubkey, c)) for c in enc_last_layer["enc_values"]]
                        arr[enc_last_layer["indices"]] = np.array(dec, dtype=np.float32)
                    # 其它客户端无法解密密文参数，保持为零
                    arr = arr.reshape(enc_last_layer["shape"]) / self.scale
                    params.append(torch.tensor(arr, dtype=torch.float32).to(self.device))
            for p, new_p in zip(self.model.parameters(), params):
                p.data = new_p.data.clone()
            torch.save(self.model.state_dict(), "models/global_model.pth")
            print("全局模型权重已保存到 models/global_model.pth")
        except Exception as e:
            print("set_parameters 异常:", e)
            traceback.print_exc()
            sys.exit(1)

    def fit(self, ins):
        try:
            self.set_parameters(ins.parameters)
            self.model.train()
            for epoch in range(1):
                for x, y in self.train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    preds = self.model(x)
                    preds = torch.clamp(preds, min=0.0, max=1.0)
                    loss = self.criterion(preds, y)
                    loss.backward()
                    self.optimizer.step()
            epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
            print(f"当前隐私预算: ε = {epsilon:.2f}, δ = 1e-5")
            save_path = f"models/client_model_{self.client_name}.pth"
            torch.save(self.model.state_dict(), save_path)
            print(f"模型权重已保存到 {save_path}")
            return fl.common.FitRes(
                parameters=self.get_parameters(ins),
                num_examples=len(self.train_loader.dataset),
                metrics={"epsilon": epsilon},
                status=fl.common.Status(code=fl.common.Code.OK, message="Success")
            )
        except Exception as e:
            print("fit 异常:", e)
            traceback.print_exc()
            sys.exit(1)

    def evaluate(self, ins):
        try:
            self.set_parameters(ins.parameters)
            self.model.eval()
            loss_total = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    preds = self.model(x)
                    preds = torch.clamp(preds, min=0.0, max=1.0)
                    loss = self.criterion(preds, y)
                    loss_total += loss.item() * y.size(0)
                    predicted = (preds > 0.5).float()
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
            acc = correct / total
            avg_loss = loss_total / total
            # 写入本地结果文件
            with open(f"res/res_{self.client_name}.txt", "a") as f:
                f.write(f"{acc},{avg_loss}\n")
            return fl.common.EvaluateRes(
                loss=avg_loss,
                num_examples=total,
                metrics={"accuracy": acc},
                status=fl.common.Status(code=fl.common.Code.OK, message="Success")
            )
        except Exception as e:
            print("evaluate 异常:", e)
            traceback.print_exc()
            sys.exit(1)

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
    client = FlowerClient(model, train_loader, val_loader, device, args.client_name)
    fl.client.start_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()