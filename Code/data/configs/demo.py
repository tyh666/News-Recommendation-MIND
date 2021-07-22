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
    query_dim = 200
    embedding_dim = 300
    filter_num = 150
    head_num = 16
    epochs = 8
    metrics = 'auc,mean_mrr,ndcg@5,ndcg@10'
    device = 'cpu'
    attrs = ['title']
    k = 0
    save_step = [0]
    validate = False
    interval = 10
    spadam = False
    val_freq = 2
    schedule = None

    # deprecated
    multiview = False
    onehot = False