import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import torch

from pt_client import PTModel  # 你已有模型定义
# 你可能还需要一个函数剥离 _module 前缀，参考之前的代码

def preprocess_test_data(test_file_path, scaler, pca):
    print("读取测试数据...")
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'outcome', 'level'])
    
    data_test = pd.read_csv(test_file_path, header=None)  # 这里指定没有表头
    data_test.columns = columns
    
    print(f"列数: {len(columns)}, 数据列数: {data_test.shape[1]}")
    print("测试数据列名:", data_test.columns.tolist())
    
    data_test.loc[data_test['outcome'] == "normal", "outcome"] = 0
    data_test.loc[data_test['outcome'] != 0, "outcome"] = 1
    data_test['outcome'] = data_test['outcome'].astype(int)
    
    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_guest_login', 'level', 'outcome']
    
    df_num = data_test.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    
    print("归一化数值特征...")
    scaled_df = scaler.transform(df_num)
    scaled_df = pd.DataFrame(scaled_df, columns=num_cols)
    
    data_test.drop(labels=num_cols, axis=1, inplace=True)
    data_test[num_cols] = scaled_df[num_cols]
    
    print("One-hot编码类别特征...")
    data_test = pd.get_dummies(data_test, columns=['protocol_type', 'service', 'flag'])
    print("One-hot之后的列名：", data_test.columns.tolist())

    print("准备X、y矩阵...")

    # y 必须在重新排列列之前取出来
    y = data_test['outcome'].values

    # 重新排列列，和训练时一致
    data_test = data_test[all_features_columns]

    X = data_test.values

    print("PCA降维到20维...")
    X_reduced = pca.transform(X)

    return X_reduced, y



def test_model(test_file_path, model_path, scaler, pca):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PTModel(input_dim=20).to(device)

    # 加载模型权重，并剥离 _module 前缀（如果有）
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_module."):
            new_state_dict[k[len("_module."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    X_test, y_test = preprocess_test_data(test_file_path, scaler, pca)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = (outputs.squeeze() > 0.5).int().cpu().numpy()

    accuracy = (preds == y_test).mean()
    print(f"测试集准确率: {accuracy:.4f}")

# 运行示例（你需先训练时保存scaler、pca和特征列名）：
if __name__ == "__main__":
    import joblib

    # 加载训练时保存的scaler、pca、one-hot列名（你需自己保存）
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    all_features_columns = np.load("all_features_columns.npy", allow_pickle=True)

    test_model("KDDTest+.txt", "client_model_alice.pth", scaler, pca)
