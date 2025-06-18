import flwr as fl
import numpy as np
import tenseal as ts
import base64
import pickle
import logging
import os
from datetime import datetime

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

class EncryptedFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # 取第一个客户端的 context 公钥
        pub_context_b64 = results[0][1].parameters.tensors[3]
        if isinstance(pub_context_b64, bytes):
            pub_context_b64 = pub_context_b64.decode("utf-8")
        pub_context_bytes = base64.b64decode(pub_context_b64)
        context = ts.context_from(pub_context_bytes)
        context.global_scale = 2 ** 60

        # 明文参数聚合
        num_clients = len(results)
        params_list = []
        shapes = None
        for _, fitres in results:
            params = pickle.loads(fitres.parameters.tensors[0])
            shapes = pickle.loads(fitres.parameters.tensors[1])
            params_list.append(params)
        agg_params = []
        for idx in range(len(params_list[0])):
            agg = sum(p[idx] for p in params_list) / num_clients
            agg_params.append(agg)

        # 加密均值聚合
        enc_means_list = []
        for _, fitres in results:
            enc_means = pickle.loads(fitres.parameters.tensors[2])
            enc_means_list.append(enc_means)
        agg_enc_means = []
        for idx in range(len(enc_means_list[0])):
            enc_vecs = []
            for enc_means in enc_means_list:
                enc_bytes = base64.b64decode(enc_means[idx].encode("utf-8"))
                enc_vec = ts.ckks_vector_from(context, enc_bytes)
                enc_vecs.append(enc_vec)
            agg_vec = enc_vecs[0]
            for v in enc_vecs[1:]:
                agg_vec += v
            agg_vec = agg_vec * (1.0 / num_clients)
            agg_bytes = agg_vec.serialize()
            agg_b64 = base64.b64encode(agg_bytes).decode("utf-8")
            agg_enc_means.append(agg_b64)

        agg_parameters = fl.common.Parameters(
            tensors=[
                pickle.dumps(agg_params),
                pickle.dumps(shapes),
                pickle.dumps(agg_enc_means),
                results[0][1].parameters.tensors[3]
            ],
            tensor_type="mixed"
        )
        return agg_parameters, {}

    def aggregate_evaluate(self, server_round, results, failures):
        # 可选：聚合评估指标
        if not results:
            return 0.0, {}
        accuracies = [r[1].metrics["accuracy"] for r in results]
        accuracy_aggregated = sum(accuracies) / len(accuracies)
        loss_aggregated = sum(r[1].loss for r in results) / len(results)
        logging.info(f"第 {server_round} 轮平均准确率: {accuracy_aggregated:.4f}")
        return loss_aggregated, {"accuracy": accuracy_aggregated}

if __name__ == "__main__":
    init_logger()
    logging.info("联邦学习服务器启动")
    strategy = EncryptedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=fl.common.Parameters(
            tensors=[pickle.dumps([]), pickle.dumps([]), b""],
            tensor_type="encrypted"
        ),
        on_fit_config_fn=lambda r: {"epochs": 1},
        on_evaluate_config_fn=lambda r: {},
        # evaluate_fn 可选
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
    )
    logging.info("联邦学习服务器已关闭")