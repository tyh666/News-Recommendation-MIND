class config:
    scale = 'demo'
    mode = 'tune'
    epochs = 8

    batch_size = 10

    k = 3
    threshold = 0
    title_length = 20
    abs_length = 40
    signal_length = 50

    npratio = 4
    his_size = 50

    dropout_p = 0.2
    device = 'cpu'
    learning_rate = 3e-5
    metrics = 'auc,mean_mrr,ndcg@5,ndcg@10'
    # vert_num = 18
    # subvert_num = 293

    embedding = 'glove'
    embedding_dim = 300
    hidden_dim = 150
    # query_dim = 200
    # head_num = 16

    rank = 0
    world_size = 0
    step = [0]
    seeds = 42
    interval = 10
    val_freq = 2
    schedule = None
    path = "../../Data/"
    tb = False

    bert = 'bert-base-uncased'
