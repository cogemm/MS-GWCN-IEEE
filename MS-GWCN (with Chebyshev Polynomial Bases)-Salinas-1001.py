# -*- coding: utf-8 -*-
"""
MS-GWCN (with Chebyshev Polynomial Bases)
- 保持整体功能/流程不变
- 引入 Chebyshev 多项式基作为多尺度谱滤波 (ChebConv)
- 训练/评估/可视化接口与打印保持一致
- 修改: 正确计算lambda_max, 添加验证集和早停, 调整训练比例等
"""

import os
import scipy.io
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.utils import to_undirected, get_laplacian
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


############################################
# 1. Data Loading and Standardization
############################################

def _safe_loadmat(path_candidates):
    """尝试多路径加载 .mat 数据，方便在不同机器上运行。"""
    for p in path_candidates:
        if os.path.exists(p):
            return scipy.io.loadmat(p)
    raise FileNotFoundError(f"Could not find any of: {path_candidates}")


# 原路径（你的环境）
base_dir = r'E:/PythonProject/HSI-Datas'
# 备选相对路径（当前目录）
rel_dir = '.'

data_dict = _safe_loadmat([os.path.join(base_dir, 'salinas.mat'),
                           os.path.join(rel_dir, 'Salinas.mat')])

corrected_data = _safe_loadmat([os.path.join(base_dir, 'salinas_corrected.mat'),
                                os.path.join(rel_dir, 'salinas_corrected.mat')])

gt_data = _safe_loadmat([os.path.join(base_dir, 'salinas_gt.mat'),
                         os.path.join(rel_dir, 'salinas_gt.mat')])

# Extract hyperspectral data and ground truth
X = corrected_data['salinas_corrected']  # (145, 145, 200)
y = gt_data['salinas_gt']  # (145, 145)

height, width, num_bands = X.shape
num_classes = 16

############################################
# 2. Graph Construction (valid pixels only)
############################################
y_flat = y.flatten()
mask = y_flat > 0
y_tensor = torch.tensor(y_flat[mask] - 1, dtype=torch.long)

X_flat_np = X.reshape(-1, num_bands)[mask]
X_flat = torch.tensor(X_flat_np, dtype=torch.float32)

valid_indices = np.where(mask)[0]
index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

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

# 计算拉普拉斯矩阵的最大特征值 lambda_max
edge_index_lap, edge_weight_lap = get_laplacian(
    data_obj.edge_index, normalization='sym'
)
L = torch.sparse_coo_tensor(
    edge_index_lap,
    edge_weight_lap,
    torch.Size([data_obj.num_nodes, data_obj.num_nodes])
)

try:
    eigenvalues, _ = torch.lobpcg(L, k=1, largest=True)
    lambda_max = eigenvalues[0].item()
except:
    lambda_max = 2.0
    print("Warning: lobpcg failed, using lambda_max=2.0")

print(f"Computed lambda_max: {lambda_max}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_obj = data_obj.to(device)


############################################
# 3. Graph Chebyshev Multi-Scale Wavelet Block
############################################
class GraphChebMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels, cheb_k_list=(2, 4, 6)):
        super().__init__()
        self.k_list = tuple(cheb_k_list)
        self.convs = nn.ModuleList([
            ChebConv(in_channels, out_channels, K=k) for k in self.k_list
        ])

    def forward(self, x, edge_index, lambda_max):
        outs = []
        for conv in self.convs:
            o = conv(x, edge_index, lambda_max=lambda_max)
            outs.append(o)
        x_cat = torch.cat(outs, dim=-1)
        return x_cat


class MS_GWCN(nn.Module):
    def __init__(self, in_channels, num_classes, cheb_k_list=(2, 4, 6), hidden_dims=(64, 128, 256, 256), dropout=0.5):
        super().__init__()
        self.k_list = tuple(cheb_k_list)
        scales = len(self.k_list)

        c1, c2, c3, c4 = hidden_dims
        self.layer1 = GraphChebMultiScale(in_channels, c1, cheb_k_list=self.k_list)
        self.layer2 = GraphChebMultiScale(c1 * scales, c2, cheb_k_list=self.k_list)
        self.layer3 = GraphChebMultiScale(c2 * scales, c3, cheb_k_list=self.k_list)
        self.layer4 = GraphChebMultiScale(c3 * scales, c4, cheb_k_list=self.k_list)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c4 * scales, num_classes)

    def forward(self, x, edge_index, lambda_max):
        x = self.layer1(x, edge_index, lambda_max);
        x = F.relu(x)
        x = self.layer2(x, edge_index, lambda_max);
        x = F.relu(x)
        x = self.layer3(x, edge_index, lambda_max);
        x = F.relu(x)
        x = self.layer4(x, edge_index, lambda_max);
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


############################################
# 4. Metrics
############################################
def compute_metrics(true, pred, num_classes):
    oa = accuracy_score(true, pred)
    conf_mat = confusion_matrix(true, pred, labels=np.arange(0, num_classes))
    class_acc = conf_mat.diagonal() / (conf_mat.sum(axis=1) + 1e-8)
    aa = np.mean(class_acc)
    kappa = cohen_kappa_score(true, pred)
    return oa, aa, kappa, conf_mat, class_acc


############################################
# 5. Training / Evaluation with Early Stopping
############################################
def run_experiment(run_id, num_epochs=400, lr=5e-4, step_size=100, gamma=0.9, weight_decay=1e-4, cheb_k_list=(2, 4, 6),
                   patience=30):
    set_seed(42 + run_id)

    indices = np.arange(len(y_tensor))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.9, random_state=42 + run_id, stratify=y_tensor.numpy()
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.2, random_state=42 + run_id, stratify=y_tensor.numpy()[train_indices]
    )

    train_mask = torch.zeros(len(y_tensor), dtype=torch.bool)
    val_mask = torch.zeros(len(y_tensor), dtype=torch.bool)
    test_mask = torch.zeros(len(y_tensor), dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    scaler = StandardScaler()
    scaler.fit(X_flat_np[train_mask.cpu().numpy()])
    X_flat_std = torch.from_numpy(scaler.transform(X_flat_np)).float().to(device)
    data_obj.x = X_flat_std

    class_counts = np.bincount(y_tensor[train_mask.cpu()].cpu().numpy(), minlength=num_classes)
    weights = 1.0 / (class_counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = MS_GWCN(in_channels=num_bands, num_classes=num_classes, cheb_k_list=cheb_k_list).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    epoch_list, oa_curve, aa_curve, kappa_curve = [], [], [],[]
    train_loss_list, val_loss_list = [], []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data_obj.x, data_obj.edge_index, lambda_max)
        loss = criterion(out[train_mask], data_obj.y[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(data_obj.x, data_obj.edge_index, lambda_max)
            val_loss = criterion(out_eval[val_mask], data_obj.y[val_mask])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            train_loss_list.append(loss.item())
            val_loss_list.append(val_loss.item())

        if epoch % 10 == 0:
            with torch.no_grad():
                pred = out_eval.argmax(dim=1).cpu().numpy()
                true = data_obj.y.cpu().numpy()
                test_idx = np.where(test_mask.cpu().numpy())[0]
                oa, aa, kappa, _, _ = compute_metrics(true[test_idx], pred[test_idx], num_classes)
                epoch_list.append(epoch)
                oa_curve.append(oa);
                aa_curve.append(aa);
                kappa_curve.append(kappa)
                print(
                    f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, OA: {oa * 100:.2f}%, AA: {aa * 100:.2f}%, Kappa: {kappa:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        out = model(data_obj.x, data_obj.edge_index, lambda_max)
        final_pred = out.argmax(dim=1).cpu().numpy()
        true = data_obj.y.cpu().numpy()
        test_idx = np.where(test_mask.cpu().numpy())[0]
        final_oa, final_aa, final_kappa, conf_mat, per_class_acc = compute_metrics(true[test_idx], final_pred[test_idx],
                                                                                   num_classes)

    return model, final_pred, final_oa, final_aa, final_kappa, conf_mat, per_class_acc, epoch_list, oa_curve, aa_curve, kappa_curve, train_loss_list, val_loss_list


############################################
# 6. Multi-Runs & Summary
############################################
num_runs = 4
oa_runs, aa_runs, kappa_runs = [], [], []
per_class_acc_runs = []
all_epoch_list, all_oa_curves, all_aa_curves, all_kappa_curves = [], [], [], []
all_train_loss, all_val_loss = [], []

for run in range(num_runs):
    print(f"\n==== Run {run + 1} ====")
    model, final_pred, oa_val, aa_val, kappa_val, conf_mat, per_class_acc, epoch_list, oa_curve, aa_curve, kappa_curve, train_loss, val_loss = \
        run_experiment(run_id=run, num_epochs=400, cheb_k_list=(2, 4, 6))

    oa_runs.append(oa_val);
    aa_runs.append(aa_val);
    kappa_runs.append(kappa_val)
    per_class_acc_runs.append(per_class_acc)
    all_epoch_list.append(epoch_list);
    all_oa_curves.append(oa_curve)
    all_aa_curves.append(aa_curve);
    all_kappa_curves.append(kappa_curve)
    all_train_loss.append(train_loss);
    all_val_loss.append(val_loss)

    print(f"Run {run + 1}: OA: {oa_val * 100:.2f}%, AA: {aa_val * 100:.2f}%, Kappa: {kappa_val:.4f}")
    for i in range(num_classes):
        print(f"Class {i + 1} Accuracy: {per_class_acc[i] * 100:.2f}%")

oa_mean, oa_std = np.mean(oa_runs), np.std(oa_runs)
aa_mean, aa_std = np.mean(aa_runs), np.std(aa_runs)
kappa_mean, kappa_std = np.mean(kappa_runs), np.std(kappa_runs)
per_class_acc_runs = np.array(per_class_acc_runs)
per_class_mean = np.mean(per_class_acc_runs, axis=0)
per_class_std = np.std(per_class_acc_runs, axis=0)

print("\n=== Evaluation Metrics ===")
print("{:<10} {:>30}".format("Metric", "Value (mean ± std)"))
print("-" * 42)
print("{:<10} {:>30}".format("OA", f"{oa_mean * 100:.2f}% ± {oa_std * 100:.2f}%"))
print("{:<10} {:>30}".format("AA", f"{aa_mean * 100:.2f}% ± {aa_std * 100:.2f}%"))
print("{:<10} {:>30}".format("Kappa", f"{kappa_mean:.2f} ± {kappa_std:.2f}"))

print("\n=== Per-Class Accuracy (mean ± std) ===")
for i in range(num_classes):
    print(f"Class {i + 1}: {per_class_mean[i] * 100:.2f}% ± {per_class_std[i] * 100:.2f}%")

############################################
# 7. Plot Curves (first run as example)
############################################
plt.figure(figsize=(14, 10))
plt.subplot(2, 3, 1)
plt.plot(all_epoch_list[0], all_oa_curves[0], marker='o')
plt.xlabel('Epoch');
plt.ylabel('Overall Accuracy (OA)');
plt.title('OA vs Epoch')

plt.subplot(2, 3, 2)
plt.plot(all_epoch_list[0], all_aa_curves[0], marker='o')
plt.xlabel('Epoch');
plt.ylabel('Average Accuracy (AA)');
plt.title('AA vs Epoch')

plt.subplot(2, 3, 3)
plt.plot(all_epoch_list[0], all_kappa_curves[0], marker='o')
plt.xlabel('Epoch');
plt.ylabel('Kappa Coefficient');
plt.title('Kappa vs Epoch')

plt.subplot(2, 3, 4)
plt.plot(all_train_loss[0], label='Train Loss')
plt.plot(all_val_loss[0], label='Validation Loss')
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.title('Loss vs Epoch')
plt.legend()

plt.subplot(2, 3, 5)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted');
plt.ylabel('True');
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

############################################
# 8. Final Map
############################################
model.eval()
with torch.no_grad():
    out = model(data_obj.x, data_obj.edge_index, lambda_max)
    predicted = out.argmax(dim=1).cpu().numpy()
true = data_obj.y.cpu().numpy()

predicted_image = np.zeros(height * width, dtype=np.int32)
predicted_image[valid_indices] = predicted + 1
predicted_image = predicted_image.reshape(height, width)

colors = [
    '#000000',
    "#0000FF",
    "#FF0000",
    "#00FF00",
    "#FFFF00",
    "#98F5FF",
    "#FF00FF",
    "#006400",
    "#A52A2A",
    "#FFEC8B",
    "#8470FF",
    "#EEAD0E",
    "#1E90FF",
    "#00FFFF",
    "#9A32CD",
    "#EE6AA7",
    "#D3D3D3"
]
cmap = ListedColormap(colors[:17])
cmap.set_under('k')

full_mask = y.flatten() > 0
predicted_image_flat = predicted_image.flatten()
predicted_image_flat[~full_mask] = 0
predicted_image = predicted_image_flat.reshape(height, width)

plt.figure(figsize=(10, 10))
plt.imshow(predicted_image, cmap=cmap, interpolation='nearest', vmin=0, vmax=16)
plt.title('Classification Result by MS-GWCN (Cheb)')
plt.axis('on')
plt.savefig('Classification_Result_MS-GWCN-Cheb.png')

fig = plt.gcf();
ax = plt.gca()
fig.set_facecolor('#000000');
ax.set_facecolor('#000000')
plt.tight_layout(pad=0)
plt.show()