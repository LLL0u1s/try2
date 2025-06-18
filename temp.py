

以下是一个联邦学习模型分为pt_server.py和pt_client.py ，请增加同态加密功能
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
from datetime import datetime

# ---------------- 日志初始化函数 ----------------
def init_logger():
    log_dir = "server_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(log_dir, f"{log_filename}.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("日志系统初始化完成")

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

# ---------------- 模型参数工具 ----------------
model = PTModel()

def get_model_parameters():
    return [val.cpu().detach().numpy() for val in model.parameters()]

def set_model_parameters(model, parameters):
    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_param).float()

# ---------------- 配置函数 ----------------
def fit_config(server_round):
    return {"epochs": 1}

def evaluate_config(server_round):
    return {}

def get_evaluate_fn():
    loss_fn = nn.BCELoss()
    data_val = np.load("data_val.npz")
    x_val = torch.tensor(data_val["x"]).float()
    y_val = torch.tensor(data_val["y"]).float().view(-1, 1)

    def evaluate(server_round, parameters, config):
        set_model_parameters(model, parameters)
        model.eval()
        with torch.no_grad():
            preds = model(x_val)
            loss = loss_fn(preds, y_val).item()
            predicted = (preds > 0.5).float()
            acc = (predicted == y_val).float().mean().item()
        return loss, {"accuracy": acc}

    return evaluate

# ---------------- 自定义FedAvg策略，支持日志和提前停止 ----------------
class MyFedAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_rounds=10, threshold_acc=0.90, **kwargs):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.threshold_acc = threshold_acc
        self.stop_training = False

    def aggregate_fit(self, server_round, results, failures):
        """直接使用父类实现，不要自己处理 metrics"""
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self,server_round: int,results,failures):
        """聚合评估结果并检查是否达到目标准确率"""
        if not results:
            return 0.0, {}

        # 计算平均准确率
        accuracies = [r[1].metrics["accuracy"] for r in results]
        accuracy_aggregated = sum(accuracies) / len(accuracies)
        
        logging.info(f"第 {server_round} 轮平均准确率: {accuracy_aggregated:.4f}")
        
        # # 如果达到目标准确率，返回None触发停止
        # if accuracy_aggregated >= self.threshold_acc:
        #     logging.info(f"准确率达到目标 {self.threshold_acc}，停止训练")
        #     self.stop_training = 1
        #     return 0.0, {}
        
        # 否则返回聚合结果
        loss_aggregated = sum(r[1].loss for r in results) / len(results)
        return loss_aggregated, {"accuracy": accuracy_aggregated}

    def evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
    ):
        """评估当前模型"""
        # if self.stop_training:
        #     logging.info(f"训练提前停止，当前轮次: {server_round}")
        #     return None

        results = super().evaluate(server_round, parameters)
        if results is None:
            logging.info("results is None")
            return None  
        loss, metrics = results
        return loss, metrics

    def configure_fit(self, server_round, parameters, client_manager):
        logging.info(f"配置第 {server_round} 轮训练客户端")
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        logging.info(f"配置第 {server_round} 轮评估客户端")
        return super().configure_evaluate(server_round, parameters, client_manager)


# ---------------- 启动服务器 ----------------
if __name__ == "__main__":
    init_logger()

    logging.info("联邦学习服务器启动")

    strategy = MyFedAvgStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters()),
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        num_rounds=50,
        threshold_acc=0.80,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )

    logging.info("联邦学习服务器已关闭")


pt_client.py
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
