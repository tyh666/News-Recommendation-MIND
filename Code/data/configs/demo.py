class config:
    scale = 'demo'
    mode = 'tune'
    epochs = 8
    batch_size = 5
    batch_size_news = 100
    batch_size_history = 100

    k = 5
    threshold = -float('inf')
    title_length = 20
    abs_length = 40
    signal_length = 100

    npratio = 4
    his_size = 50
    cdd_size = 5
    impr_size = 10

    dropout_p = 0.2
    device = 'cpu'
    lr = 1e-4
    bert_lr = 3e-5
    metrics = 'auc,mean_mrr,ndcg@5,ndcg@10'
    # vert_num = 18
    # subvert_num = 293

    embedding = 'bert'
    encoderN = 'cnn'
    encoderU = 'rnn'
    selector = 'sfi'
    reducer = 'matching'
    ranker = 'onepass'

    embedding_dim = 768
    hidden_dim = 384
    head_num = 12

    rank = 0
    base_rank = 0
    world_size = 0
    step = 0
    seed = 42
    interval = 10

    granularity = 'avg'
    debias = True
    full_attn = True
    descend_history = False
    shuffle_pos = False
    save_pos = False
    sep_his = False
    diversify = False
    no_dedup = False
    no_order_embed = False
    no_rm_punc = False

    fast = False
    scheduler = 'linear'
    warmup = 100
    pin_memory = False
    shuffle = False
    bert = 'bert-base-uncased'
    num_workers = 0
    smoothing = 0.3

    path = "../../../Data/"

    tb = False