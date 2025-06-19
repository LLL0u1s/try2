import flwr as fl
import numpy as np
import pickle
import logging
import os
from datetime import datetime
from phe import paillier

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
        try:
            if not results:
                return
            # 获取公钥
            pubkey_bytes = results[0][1].parameters.tensors[3]
            pubkey = pickle.loads(pubkey_bytes)
            num_clients = len(results)
            # 收集所有客户端参数
            plain_params_list = []
            enc_last_layer_list = []
            shapes = None
            for _, fitres in results:
                plain_params = pickle.loads(fitres.parameters.tensors[0])
                enc_last_layer = pickle.loads(fitres.parameters.tensors[1])
                shapes = pickle.loads(fitres.parameters.tensors[2])
                plain_params_list.append(plain_params)
                enc_last_layer_list.append(enc_last_layer)
            # 1. 明文参数直接平均
            agg_plain_params = []
            for i in range(len(plain_params_list[0])):
                if plain_params_list[0][i] is None:
                    agg_plain_params.append(None)
                else:
                    arrs = [client_params[i] for client_params in plain_params_list]
                    agg = sum(arrs) // num_clients  # 用整数平均，保持量化一致
                    agg_plain_params.append(agg)
            # 2. 密文参数（最后一层）同态加密聚合
            # 假设所有客户端最后一层 shape、非零索引一致
            indices = enc_last_layer_list[0]["indices"]
            shape = enc_last_layer_list[0]["shape"]
            agg_enc_values = []
            for idx in range(len(indices)):
                # 收集每个客户端该位置的密文
                enc_nums = [
                    paillier.EncryptedNumber(pubkey, enc_last_layer["enc_values"][idx])
                    for enc_last_layer in enc_last_layer_list
                ]
                agg_num = enc_nums[0]
                for n in enc_nums[1:]:
                    agg_num += n
                agg_enc_values.append(agg_num.ciphertext())
            agg_enc_last_layer = {
                "indices": indices,
                "enc_values": agg_enc_values,
                "shape": shape
            }
            # 3. 返回聚合参数
            agg_parameters = fl.common.Parameters(
                tensors=[
                    pickle.dumps(agg_plain_params),
                    pickle.dumps(agg_enc_last_layer),
                    pickle.dumps(shapes),
                    pubkey_bytes
                ],
                tensor_type="mixed"
            )
            logging.info(f"第 {server_round} 轮参数聚合完成")
            return agg_parameters, {}
        except Exception as e:
            logging.error(f"aggregate_fit 异常: {e}", exc_info=True)
            raise

    def aggregate_evaluate(self, server_round, results, failures):
        try:
            if not results:
                return 0.0, {}
            accuracies = [r[1].metrics["accuracy"] for r in results]
            accuracy_aggregated = sum(accuracies) / len(accuracies)
            loss_aggregated = sum(r[1].loss for r in results) / len(results)
            logging.info(f"第 {server_round} 轮平均准确率: {accuracy_aggregated:.4f}")
            return loss_aggregated, {"accuracy": accuracy_aggregated}
        except Exception as e:
            logging.error(f"aggregate_evaluate 异常: {e}", exc_info=True)
            raise

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
            tensor_type="mixed"
        ),
        on_fit_config_fn=lambda r: {"epochs": 1},
        on_evaluate_config_fn=lambda r: {},
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=25),
        strategy=strategy,
    )
    logging.info("联邦学习服务器已关闭")