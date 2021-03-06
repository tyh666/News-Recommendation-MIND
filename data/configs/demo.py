class config:
    scale = 'demo'
    mode = 'train'
    epochs = 8
    batch_size = 5
    batch_size_news = 500

    k = 3
    threshold = -float('inf')
    title_length = 20
    abs_length = 40
    signal_length = 100
    news = None
    case = False

    npratio = 4
    his_size = 50
    cdd_size = 5
    impr_size = 2000

    dropout_p = 0.2
    device = 0
    lr = 1e-4
    bert_lr = 3e-5
    metrics = 'auc,mean_mrr,ndcg@5,ndcg@10'
    # vert_num = 18
    # subvert_num = 293

    embedding = 'bert'
    encoderN = 'cnn'
    encoderU = 'lstm'
    selector = 'sfi'
    reducer = 'matching'
    ranker = 'onepass'
    pooler = "attn"
    aggregator = None

    bert_dim = 768
    embedding_dim = 768
    hidden_dim = 384
    head_num = 12

    rank = 0
    base_rank = 0
    world_size = 0
    step = 0
    seed = 42
    interval = 10

    debias = False
    full_attn = True
    descend_history = False
    shuffle_pos = False
    save_pos = False
    sep_his = False
    diversify = False
    no_dedup = False
    segment_embed = False
    no_rm_punc = False

    fast = False
    scheduler = 'linear'
    warmup = 100
    pin_memory = False
    shuffle = False
    bert = 'bert'
    num_workers = 0
    smoothing = 0.3

    path = "../../../Data/"
    unilm_path = path + 'bert_cache/UniLM/unilm2-base-uncased.bin'
    unilm_config_path = path + 'bert_cache/UniLM/unilm2-base-uncased-config.json'


    path = "../../../Data/"

    tb = False