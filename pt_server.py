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
