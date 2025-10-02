# -*- coding: utf-8 -*-
"""
MS-GWCN (with Chebyshev Polynomial Bases)
- 保持整体功能/流程不变
- 引入 Chebyshev 多项式基作为多尺度谱滤波 (ChebConv)
- 训练/评估/可视化接口与打印保持一致
"""

import os
import scipy.io
import seaborn as sns  # 保留（未用也不影响）
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.utils import to_undirected
from torch.optim import Adam
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from matplotlib.colors import ListedColormap
from torch.optim.lr_scheduler import StepLR

# -----------------------------
# 0. Reproducibility
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了确定性（可能会轻微降速）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed(42)  # Moved inside multi-runs for variation per run

############################################
# 1. Data Loading and Standardization
############################################

def _safe_loadmat(path_candidates):
    """尝试多路径加载 .mat 数据，方便在不同机器上运行。"""
    for p in path_candidates:
        if os.path.exists(p):
            return scipy.io.loadmat(p)
    # 若失败则抛错，便于用户自行调整路径
    raise FileNotFoundError(f"Could not find any of: {path_candidates}")

# 原路径（你的环境）
base_dir = r'E:/PythonProject/HSI-Datas'
# 备选相对路径（当前目录）
rel_dir  = '.'

data_dict = _safe_loadmat([os.path.join(base_dir, 'PaviaU.mat'),
                           os.path.join(rel_dir,  'PaviaU.mat')])

corrected_data = _safe_loadmat([os.path.join(base_dir, 'PaviaU.mat'),
                                os.path.join(rel_dir,  'PaviaU.mat')])

gt_data = _safe_loadmat([os.path.join(base_dir, 'PaviaU_gt.mat'),
                         os.path.join(rel_dir,  'PaviaU_gt.mat')])

# Extract hyperspectral data and ground truth
X = corrected_data['paviaU']  # (145, 145, 200)
y = gt_data['paviaU_gt']               # (145, 145)

height, width, num_bands = X.shape
# 假定 0 为背景，实际类别为 1..16
num_classes = 9

############################################
# 2. Graph Construction (valid pixels only)
############################################
# 仅保留有标签的像素（>0），并将标签从 1..16 映射到 0..15
y_flat = y.flatten()
mask = y_flat > 0
y_tensor = torch.tensor(y_flat[mask] - 1, dtype=torch.long)  # [N_valid]

# 特征也仅保留有效像素
X_flat_np = X.reshape(-1, num_bands)[mask]                   # [N_valid, C]
X_flat = torch.tensor(X_flat_np, dtype=torch.float32)

# 记录有效像素在原图中的索引，并构造映射
valid_indices = np.where(mask)[0]
index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

# 采用 8 邻接，且仅在有效像素之间连边
edges = []
for i in range(height):
    for j in range(width):
        node_idx = i * width + j
        if node_idx in index_mapping:
            cur = index_mapping[node_idx]
            # 上下左右
            if i > 0 and (node_idx - width) in index_mapping:
                edges.append((cur, index_mapping[node_idx - width]))
            if i < height - 1 and (node_idx + width) in index_mapping:
                edges.append((cur, index_mapping[node_idx + width]))
            if j > 0 and (node_idx - 1) in index_mapping:
                edges.append((cur, index_mapping[node_idx - 1]))
            if j < width - 1 and (node_idx + 1) in index_mapping:
                edges.append((cur, index_mapping[node_idx + 1]))
            # 四个对角
            if i > 0 and j > 0 and (node_idx - width - 1) in index_mapping:
                edges.append((cur, index_mapping[node_idx - width - 1]))
            if i > 0 and j < width - 1 and (node_idx - width + 1) in index_mapping:
                edges.append((cur, index_mapping[node_idx - width + 1]))
            if i < height - 1 and j > 0 and (node_idx + width - 1) in index_mapping:
                edges.append((cur, index_mapping[node_idx + width - 1]))
            if i < height - 1 and j < width - 1 and (node_idx + width + 1) in index_mapping:
                edges.append((cur, index_mapping[node_idx + width + 1]))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_index = to_undirected(edge_index)

data_obj = Data(x=X_flat, edge_index=edge_index, y=y_tensor)

############################################
# 3. Train-Test Split (Stratified) + 标准化（仅训练集）
############################################
# Note: Moved split inside run_experiment for per-run variation

# 标准化仅在训练子集上拟合，避免泄漏 (done per run)

# 类别不均衡权重 (done per run)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# 4. Graph Chebyshev Multi-Scale Wavelet Block
############################################
class GraphChebMultiScale(nn.Module):
    """
    使用 Chebyshev 多项式基 (ChebConv) 的多尺度卷积块。
    不同的 K 表示不同的“尺度”，各尺度输出在通道维拼接。
    """
    def __init__(self, in_channels, out_channels, cheb_k_list=(2, 4, 6)):
        super().__init__()
        self.k_list = tuple(cheb_k_list)
        self.convs = nn.ModuleList([
            ChebConv(in_channels, out_channels, K=k) for k in self.k_list
        ])

    def forward(self, x, edge_index):
        outs = []
        for conv in self.convs:
            # ChebConv 支持 edge_weight/lambda_max，默认内部会估计
            o = conv(x, edge_index)
            outs.append(o)
        x_cat = torch.cat(outs, dim=-1)  # [N, out_channels * num_scales]
        return x_cat


class MS_GWCN(nn.Module):
    """
    多层多尺度 Chebyshev 卷积网络：
    - 每层都是 GraphChebMultiScale
    - 每层后 ReLU
    - 最后 Dropout + FC 得到 logits
    """
    def __init__(self, in_channels, num_classes, cheb_k_list=(2, 4, 6), hidden_dims=(64, 128, 256, 256), dropout=0.5):  # Reduced dims, increased dropout
        super().__init__()
        self.k_list = tuple(cheb_k_list)
        scales = len(self.k_list)

        c1, c2, c3, c4 = hidden_dims
        self.layer1 = GraphChebMultiScale(in_channels, c1, cheb_k_list=self.k_list)
        self.layer2 = GraphChebMultiScale(c1 * scales,   c2, cheb_k_list=self.k_list)
        self.layer3 = GraphChebMultiScale(c2 * scales,   c3, cheb_k_list=self.k_list)
        self.layer4 = GraphChebMultiScale(c3 * scales,   c4, cheb_k_list=self.k_list)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c4 * scales, num_classes)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index); x = F.relu(x)
        x = self.layer2(x, edge_index); x = F.relu(x)
        x = self.layer3(x, edge_index); x = F.relu(x)
        x = self.layer4(x, edge_index); x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x  # raw logits


############################################
# 5. Metrics
############################################
def compute_metrics(true, pred, num_classes):
    oa = accuracy_score(true, pred)
    conf_mat = confusion_matrix(true, pred, labels=np.arange(0, num_classes))
    class_acc = conf_mat.diagonal() / (conf_mat.sum(axis=1) + 1e-8)
    aa = np.mean(class_acc)
    kappa = cohen_kappa_score(true, pred)
    return oa, aa, kappa, conf_mat, class_acc

############################################
# 6. Training / Evaluation
############################################
def run_experiment(run_id, num_epochs=400, lr=5e-4, step_size=100, gamma=0.9, weight_decay=1e-4, cheb_k_list=(2,4,6)):  # Added weight_decay
    set_seed(42 + run_id)  # Vary seed per run
    indices = np.arange(len(y_tensor))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.8, random_state=42 + run_id, stratify=y_tensor.numpy()  # Changed to 0.8 (20% train); vary random_state
    )
    train_mask = torch.zeros(len(y_tensor), dtype=torch.bool)
    test_mask  = torch.zeros(len(y_tensor), dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices]   = True

    # 标准化仅在训练子集上拟合，避免泄漏
    scaler = StandardScaler()
    scaler.fit(X_flat_np[train_mask.numpy()])
    X_flat_std = torch.from_numpy(scaler.transform(X_flat_np)).float()
    data_obj.x = X_flat_std  # Update per run (though features fixed, scaler varies slightly)

    # 类别不均衡权重
    class_counts = np.bincount(y_tensor[train_mask].cpu().numpy(), minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = MS_GWCN(in_channels=num_bands, num_classes=num_classes, cheb_k_list=cheb_k_list).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    epoch_list, oa_curve, aa_curve, kappa_curve = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data_obj.x.to(device), data_obj.edge_index.to(device))
        loss = criterion(out[train_mask.to(device)], data_obj.y.to(device)[train_mask.to(device)])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                out_eval = model(data_obj.x.to(device), data_obj.edge_index.to(device))
                pred = out_eval.argmax(dim=1).cpu().numpy()
                true = data_obj.y.cpu().numpy()
                test_idx = np.where(test_mask.cpu().numpy())[0]
                oa, aa, kappa, _, _ = compute_metrics(true[test_idx], pred[test_idx], num_classes)
                epoch_list.append(epoch)
                oa_curve.append(oa); aa_curve.append(aa); kappa_curve.append(kappa)
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, OA: {oa * 100:.2f}%, AA: {aa * 100:.2f}%, Kappa: {kappa:.4f}")

    model.eval()
    with torch.no_grad():
        out = model(data_obj.x.to(device), data_obj.edge_index.to(device))
        final_pred = out.argmax(dim=1).cpu().numpy()
        true = data_obj.y.cpu().numpy()
        test_idx = np.where(test_mask.cpu().numpy())[0]
        final_oa, final_aa, final_kappa, conf_mat, per_class_acc = compute_metrics(true[test_idx], final_pred[test_idx], num_classes)

    return model, final_pred, final_oa, final_aa, final_kappa, conf_mat, per_class_acc, epoch_list, oa_curve, aa_curve, kappa_curve


############################################
# 7. Multi-Runs & Summary
############################################
num_runs = 4
oa_runs, aa_runs, kappa_runs = [], [], []
per_class_acc_runs = []
all_epoch_list, all_oa_curves, all_aa_curves, all_kappa_curves = [], [], [], []

for run in range(num_runs):
    print(f"\n==== Run {run + 1} ====")
    # 你可以调整 cheb_k_list，例如 (1,2,3,4) 以获得更细尺度
    model, final_pred, oa_val, aa_val, kappa_val, conf_mat, per_class_acc, epoch_list, oa_curve, aa_curve, kappa_curve = \
        run_experiment(run_id=run, num_epochs=400, cheb_k_list=(2,4,6))
    oa_runs.append(oa_val); aa_runs.append(aa_val); kappa_runs.append(kappa_val)
    per_class_acc_runs.append(per_class_acc)
    all_epoch_list.append(epoch_list); all_oa_curves.append(oa_curve)
    all_aa_curves.append(aa_curve);   all_kappa_curves.append(kappa_curve)

    print(f"Run {run + 1}: OA: {oa_val * 100:.2f}%, AA: {aa_val * 100:.2f}%, Kappa: {kappa_val:.4f}")
    for i in range(num_classes):
        print(f"Class {i+1} Accuracy: {per_class_acc[i] * 100:.2f}%")

oa_mean, oa_std = np.mean(oa_runs), np.std(oa_runs)
aa_mean, aa_std = np.mean(aa_runs), np.std(aa_runs)
kappa_mean, kappa_std = np.mean(kappa_runs), np.std(kappa_runs)
per_class_acc_runs = np.array(per_class_acc_runs)  # shape: (num_runs, num_classes)
per_class_mean = np.mean(per_class_acc_runs, axis=0)
per_class_std  = np.std(per_class_acc_runs, axis=0)

print("\n=== Evaluation Metrics ===")
print("{:<10} {:>30}".format("Metric", "Value (mean ± std)"))
print("-" * 42)
print("{:<10} {:>30}".format("OA",    f"{oa_mean * 100:.2f}% ± {oa_std * 100:.2f}%"))
print("{:<10} {:>30}".format("AA",    f"{aa_mean * 100:.2f}% ± {aa_std * 100:.2f}%"))
print("{:<10} {:>30}".format("Kappa", f"{kappa_mean:.2f} ± {kappa_std:.2f}"))

print("\n=== Per-Class Accuracy (mean ± std) ===")
for i in range(num_classes):
    print(f"Class {i+1}: {per_class_mean[i] * 100:.2f}% ± {per_class_std[i] * 100:.2f}%")

############################################
# 8. Plot Curves (first run as example)
############################################
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.plot(all_epoch_list[0], all_oa_curves[0], marker='o')
plt.xlabel('Epoch'); plt.ylabel('Overall Accuracy (OA)'); plt.title('OA vs Epoch')

plt.subplot(1, 3, 2)
plt.plot(all_epoch_list[0], all_aa_curves[0], marker='o')
plt.xlabel('Epoch'); plt.ylabel('Average Accuracy (AA)'); plt.title('AA vs Epoch')

plt.subplot(1, 3, 3)
plt.plot(all_epoch_list[0], all_kappa_curves[0], marker='o')
plt.xlabel('Epoch'); plt.ylabel('Kappa Coefficient'); plt.title('Kappa vs Epoch')
plt.tight_layout()
plt.show()

############################################
# 9. Final Map
############################################
model.eval()
with torch.no_grad():
    out = model(data_obj.x.to(device), data_obj.edge_index.to(device))
    predicted = out.argmax(dim=1).cpu().numpy()
true = data_obj.y.cpu().numpy()

# 将预测映射回原图尺寸（背景=0，类别=1..9）
predicted_image = np.zeros(height * width, dtype=np.int32)
predicted_image[valid_indices] = predicted + 1
predicted_image = predicted_image.reshape(height, width)

# 自定义颜色（0 背景 + 16 类）
colors = [
    '#000000',  # 0: Background
    '#FF3333',  # 1: Asphal
    '#DAA520',  # 2: Meadows
    '#66FF66',  # 3: Gravel
    '#33CC33',  # 4: Trees
    '#00FFFF',  # 5: Painted metal sheets
    '#0099FF',  # 6: Bare Soil
    '#7B68EE',  # 7: Bitumen
    '#BA55D3',  # 8: Self-Blocking Bricks
    '#EE82EE',  # 9: Shadows
]

cmap = ListedColormap(colors[:10])
cmap.set_under('k')

# 非有效区域设为 0（背景）
full_mask = y.flatten() > 0
predicted_image_flat = predicted_image.flatten()
predicted_image_flat[~full_mask] = 0
predicted_image = predicted_image_flat.reshape(height, width)

plt.figure(figsize=(10, 10))
plt.imshow(predicted_image, cmap=cmap, interpolation='nearest', vmin=0, vmax=10)
plt.title('Classification Result by MS-GWCN (Cheb)')
plt.axis('on')
plt.savefig('Classification_Result_MS-GWCN-Cheb.png')

fig = plt.gcf(); ax = plt.gca()
fig.set_facecolor('#000000'); ax.set_facecolor('#000000')
plt.tight_layout(pad=0)
plt.show()