import pandas as pd
def fun1(filename="KDDTest+.txt"):
    print(f"{filename}")
    with open(filename) as f:
        print(len(f.readline().strip().split(',')))

    data_test = pd.read_csv("KDDTest+.txt", header=None)
    print(data_test.shape)
    print(data_test.head())

fun1("KDDTrain+.txt")
fun1()
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
def fun2(filename="KDDTest+.txt"):
    data_test = pd.read_csv(filename, header=None, names=columns, sep=',')
    print("columns\n",data_test.columns)
    print("head\n",data_test.head())

fun2("KDDTrain+.txt")
fun2()