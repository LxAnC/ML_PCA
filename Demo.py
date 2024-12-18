import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
# 加载手写数字数据集
digits = load_digits()
data = digits.data
labels = digits.target

print("数据形状：", data.shape)  # 输出数据维度
# 数据标准化
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

# PCA 降维至 2 维
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_std)

print("降维后数据形状：", data_pca.shape)
# 绘制降维后的数据
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.scatter(data_pca[labels == i, 0], data_pca[labels == i, 1], label=str(i), s=10)
plt.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on MNIST Dataset")
plt.show()

# 输出方差贡献率
print("方差贡献率：", pca.explained_variance_ratio_)
print("累计方差贡献率：", np.cumsum(pca.explained_variance_ratio_))