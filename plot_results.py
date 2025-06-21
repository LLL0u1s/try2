import matplotlib.pyplot as plt
import numpy as np
def plot_acc_loss_curve():
    def load_results(filename):
        accs, losses = [], []
        with open(filename, "r") as f:
            for line in f:
                acc, loss = map(float, line.strip().split(","))
                accs.append(acc)
                losses.append(loss)
        return accs, losses

    alice_acc, alice_loss = load_results("res/res_alice.txt")
    bob_acc, bob_loss = load_results("res/res_bob.txt")
    total_acc, total_loss = load_results("res/res_total.txt")

    max_len = max(len(alice_acc), len(bob_acc), len(total_acc))

    def pad_front(data, target_len, pad_value=np.nan):
        return [pad_value] * (target_len - len(data)) + data

    alice_acc = pad_front(alice_acc, max_len)
    alice_loss = pad_front(alice_loss, max_len)
    bob_acc = pad_front(bob_acc, max_len)
    bob_loss = pad_front(bob_loss, max_len)
    total_acc = pad_front(total_acc, max_len)
    total_loss = pad_front(total_loss, max_len)
    rounds = np.arange(1, max_len + 1)

    # 只显示有效数据段（去掉前面全是nan的区间）
    def first_valid_idx(*datas):
        idx = max([next((i for i, v in enumerate(d) if not np.isnan(v)), len(d)) for d in datas])
        return idx

    start_idx = first_valid_idx(alice_acc, bob_acc, total_acc)
    rounds = rounds[start_idx:]
    alice_acc = alice_acc[start_idx:]
    bob_acc = bob_acc[start_idx:]
    total_acc = total_acc[start_idx:]
    alice_loss = alice_loss[start_idx:]
    bob_loss = bob_loss[start_idx:]
    total_loss = total_loss[start_idx:]

    # 计算准确率的最大最小值
    acc_all = np.array([alice_acc, bob_acc, total_acc])
    acc_valid = acc_all[~np.isnan(acc_all)]
    acc_max = np.nanmax(acc_all)
    acc_min = np.nanmin(acc_all)
    acc_ylim = (max(acc_min - 0.1, 0), min(acc_max + 0.1, 1.0))

    plt.figure(figsize=(10,5))
    plt.plot(rounds, alice_acc, label="Alice Accuracy", linewidth=2, marker='o')
    plt.plot(rounds, bob_acc, label="Bob Accuracy", linewidth=2, marker='s')
    plt.plot(rounds, total_acc, label="Total Accuracy", linewidth=3, linestyle='--')
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Learning Accuracy per Round")
    plt.ylim(acc_ylim)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("res/accuracy_curve.png")
    plt.show()

    # 计算loss的最大最小值
    loss_all = np.array([alice_loss, bob_loss, total_loss])
    loss_valid = loss_all[~np.isnan(loss_all)]
    loss_max = np.nanmax(loss_all)
    loss_min = np.nanmin(loss_all)
    loss_ylim = (max(loss_min - 0.1, 0), loss_max + 0.1)

    plt.figure(figsize=(10,5))
    plt.plot(rounds, alice_loss, label="Alice Loss", linewidth=2, marker='o')
    plt.plot(rounds, bob_loss, label="Bob Loss", linewidth=2, marker='s')
    plt.plot(rounds, total_loss, label="Total Loss", linewidth=3, linestyle='--')
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Federated Learning Loss per Round")
    plt.ylim(loss_ylim)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("res/loss_curve.png")
    plt.show()

def plot_train_time(client_names=["alice", "bob"]):
    plt.figure(figsize=(10,5))
    for cname in client_names:
        try:
            with open(f"res/res_time_{cname}.txt") as f:
                times = [float(line.strip()) for line in f if line.strip()]
            rounds = np.arange(1, len(times)+1)
            plt.plot(rounds, times, marker='o', label=f"{cname.capitalize()} Train Time")
        except FileNotFoundError:
            continue
    plt.xlabel("Round")
    plt.ylabel("Train Time (s)")
    plt.title("Training Time per Round")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("res/train_time_curve.png")
    plt.show()

if __name__ == "__main__":
    plot_acc_loss_curve()
    plot_train_time(["alice", "bob"])
    print("所有图表已生成并保存。")