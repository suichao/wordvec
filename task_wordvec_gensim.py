from gensim.models import Word2Vec
import jieba
import os
model_path = "output/word2vec.model"


def read_file(path):
    return open(path, "r", encoding='utf-8')


def train():
    sentence = read_file("data/passage.txt").readlines()
    stopwords = read_file("data/stopwords.txt").readlines()
    sentence_cut = [" ".join(jieba.lcut(line)) for line in sentence]
    sentence_no_stopwords = [[word for word in line.split() if word not in stopwords] for line in sentence_cut]
    # min_count是最低出现数，默认数值是5；
    # size是gensim Word2Vec将词汇映射到的N维空间的维度数量（N）默认的size数是100；
    # iter是模型训练时在整个训练语料库上的迭代次数，假如参与训练的文本量较少，就需要把这个参数调大一些。iter的默认值为5；
    # sg是模型训练所采用的的算法类型：1 代表 skip-gram，0代表 CBOW，sg的默认值为0；
    # window控制窗口，如果设得较小，那么模型学习到的是词汇间的组合性关系（词性相异）；如果设置得较大，会学习到词汇之间的聚合性关系（词性相同）。模型默认的window数值为5；
    model = Word2Vec(sentences=sentence_no_stopwords, vector_size=100, window=5, min_count=1, workers=4)
    model.save(model_path)


if __name__ == '__main__':
    force_retrain = False
    if not os.path.exists(model_path) or force_retrain:
        train()
    wvm = Word2Vec.load(model_path)
    # 词汇token
    # print(wvm.wv.key_to_index)

    # 查看词向量的维度,109个词汇，维度为100
    print(wvm.wv.vectors.shape)

    # 词汇相似度
    sim = wvm.wv.similarity('上帝', '玉帝')
    print("相似度：", sim)

    # 召回相似词汇
    most_sim = wvm.wv.most_similar("黄河")
    print("most_sim:", most_sim)

    # 类似于女人+先生-男人的结果
    compute_vec = wvm.wv.most_similar(positive=["女人", "先生"], negative=["女人"], topn=10)
    print("compute_vec:", compute_vec)

    # 找出不太合群的词
    outliers = wvm.wv.doesnt_match(["疯狂", "痛苦", "包含"])
    print("outliers:", outliers)

    #  返回与爱情最近的词和相似度
    sim_word = wvm.wv.similar_by_word("爱情", topn=10, restrict_vocab=30)
    print("sim_word:", sim_word)

    # 接近词汇A更甚于词汇B接近词汇A的【所有】词汇,按相似度由高到低降序排列
    btw_word = wvm.wv.closer_than('迷恋', '爱情')
    print("between word", btw_word[:3])

    # 给定上下文词汇作为输入，可以获得中心词汇的概率分布
    cent_word = wvm.predict_output_word(['痛苦', '疯狂', '狂热'], topn=10)
    print("centers word", cent_word)
