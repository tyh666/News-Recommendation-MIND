from .base_config import BaseConfig
class config(BaseConfig):
    scale = 'demo'
    mode = 'train'
    batch_size = 10
    title_size = 20
    abs_size = 40
    his_size = 50
    learning_rate = 1e-3
    vert_num = 18
    subvert_num = 293
    npratio = 4
    dropout_p = 0.2
    embedding_dim = 300
    hidden_dim = 150
    query_dim = 200
    head_num = 16
    epochs = 8
    metrics = 'auc,mean_mrr,ndcg@5,ndcg@10'
    device = 'cpu'
    k = 0
    step = [0]
    interval = 10
    val_freq = 2
    schedule = None
    embedding = 'glove'