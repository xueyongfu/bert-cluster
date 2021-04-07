from sklearn.cluster import KMeans
from bert_serving.client import BertClient
import pandas as pd
from bert_encoder import encode


# def text_encode(lines):
#     bc = BertClient(check_length=False)
#     vecs = bc.encode(lines)  # （1273,768） np.ndarray数组
#     return vecs

def text_encode(lines):
    return encode(lines)

def cluster(lines):
    num_clusters = 50
    # 构造聚类器
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
    vecs = text_encode(lines)
    km_cluster.fit(vecs)
    label_sentence = km_cluster.labels_  # 获取聚类标签
    center_array = km_cluster.cluster_centers_  # 获取聚类中心
    return label_sentence

def main():
    df = pd.read_excel('data/all_data.xlsx')
    lines = [ str(line) for line in df['评论内容'].to_list()]
    labels = cluster(lines)
    df['cluster'] = labels
    sdf = []
    for _, s in df.groupby('cluster'):
        sdf.append(s)
    pd.concat(sdf)[['评论内容','cluster']].to_excel('data/cluster_res_new.xlsx', index=False)


if __name__ == '__main__':
    main()
    pass


