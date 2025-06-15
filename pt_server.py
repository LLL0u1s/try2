import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

# ---------------- 自定义FedAvg策略，控制训练是否继续 ----------------
class MyFedAvgStrategy(fl.server.strategy.FedAvg):
    def __init__(self, max_rounds=3, threshold_acc=0.90, **kwargs):
        super().__init__(**kwargs)
        self.max_rounds = max_rounds  # 最大训练轮数
        self.threshold_acc = threshold_acc  # 准确率阈值
        self.stop_training = False

    def configure_next_round(
        self,
        rnd: int,
        results,
        failures,
    ):
        if failures:
            print(f"第{rnd}轮有失败客户端，继续训练")
            self.stop_training = False
        else:
            accuracies = []
            for _, metrics in results:
                if metrics and "accuracy" in metrics:
                    accuracies.append(metrics["accuracy"])
            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"第{rnd}轮平均准确率: {avg_acc:.4f}")
                if avg_acc >= self.threshold_acc:
                    print(f"准确率达到{avg_acc:.4f}，停止训练")
                    self.stop_training = True

        if self.stop_training or rnd >= self.max_rounds:
            return None, {}  # None表示停止训练
        else:
            return super().configure_next_round(rnd, results, failures)

# ---------------- 启动服务器 ----------------
if __name__ == "__main__":
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
        max_rounds=3,
        threshold_acc=0.97,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),  # 这个最大轮数是防止服务端强制停止
        strategy=strategy,
    )
