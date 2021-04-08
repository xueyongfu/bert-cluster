import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pandas as pd
from bert_encoder import encode


def elbow_fun_for_test():
    x = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
    y = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
    data = np.array(list(zip(x, y)))

    # 肘部法则 求解最佳分类数
    # K-Means参数的最优解也是以成本函数最小化为目标
    # 成本函数是各个类畸变程度（distortions）之和。每个类的畸变程度等于该类重心与其内部成员位置距离的平方和
    aa = []
    K = range(1, 10)
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        aa.append(sum(np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])
    plt.figure()
    plt.plot(np.array(K), aa, 'bx-')
    plt.show()
    # 绘制散点图及聚类结果中心点
    plt.figure()
    plt.axis([0, 10, 0, 10])
    plt.grid(True)
    plt.plot(x, y, 'k.')
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    plt.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 'r.')
    plt.show()


def text_encode(lines):
    return encode(lines)


def cluster_use_elbow(weights, search_range):
    scores = []
    for k in search_range:
        clf = KMeans(n_clusters=k)
        clf.fit(weights)
        # score = clf.inertia_
        # scores.append(score)
        scores.append(sum(np.min(cdist(weights, clf.cluster_centers_, 'euclidean'), axis=1)) / len(weights))
    plt.figure()
    plt.plot(list(search_range), scores, 'bx-')
    plt.show()


def cluster_k(weights, lines, k=6):
    clf = KMeans(n_clusters=k, max_iter=300, n_init=20, init='k-means++', n_jobs=-1)
    y = clf.fit_predict(weights)

    res = pd.DataFrame({'line': lines, 'cluster': y})
    sdf = []
    for _, d in res.groupby('cluster'):
        sdf.append(d)
    res = pd.concat(sdf)
    res.to_excel('data/entity_data/cluster_res.xlsx', index=False)


def main():
    df = pd.read_csv('data/entity_data/final_results.csv')
    df = df.dropna(subset=['address']).head(100)
    lines = [str(line) for line in df['address'].to_list()]

    search_range = range(3, 25)
    weights = encode(lines)

    # 获取k值
    cluster_use_elbow(weights, search_range)

    # 指定k值聚类
    cluster_k(weights, lines, k=5)


if __name__ == '__main__':
    main()
    pass
