import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

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

    # 标签二值化 normal->0，其他->1
    data_train.loc[data_train['outcome'] == "normal", "outcome"] = 0
    data_train.loc[data_train['outcome'] != 0, "outcome"] = 1
    data_train['outcome'] = data_train['outcome'].astype(int)

    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_guest_login', 'level', 'outcome']

    # 数值列
    df_num = data_train.drop(cat_cols, axis=1)
    num_cols = df_num.columns

    print("归一化数值特征...")
    scaler = RobustScaler()
    scaled_df = scaler.fit_transform(df_num)
    scaled_df = pd.DataFrame(scaled_df, columns=num_cols)

    # 合并回原数据
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

    # 切分：20%验证，剩下80%平分给alice和bob
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

if __name__ == "__main__":
    prepare_and_save_data()
