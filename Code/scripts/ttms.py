import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import prepare, load_manager, setup, cleanup
from models.TTMS import TTMS

def main(rank, manager, dist=False):
    """ train/dev/test/tune the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
        conig
    """
    setup(rank, manager)
    loaders = prepare(manager)

    from models.Embeddings.BERT import BERT_Embedding
    embedding = BERT_Embedding(manager)

    if manager.encoderN == 'cnn':
        from models.Encoders.CNN import CNN_Encoder
        encoderN = CNN_Encoder(manager)
    elif manager.encoderN == 'bert':
        from models.Encoders.BERT import BERT_Encoder
        encoderN = BERT_Encoder(manager)

    if manager.encoderU == 'rnn':
        from models.Encoders.RNN import RNN_User_Encoder
        encoderU = RNN_User_Encoder(manager)
    elif manager.encoderU == 'avg':
        from models.Encoders.Pooling import Average_Pooling
        encoderU = Average_Pooling(manager)
    elif manager.encoderU == 'attn':
        from models.Encoders.Pooling import Attention_Pooling
        encoderU = Attention_Pooling(manager)

    if manager.reducer in ['matching', 'bow']:
        from models.Modules.DRM import Matching_Reducer
        reducer = Matching_Reducer(manager)
    elif manager.reducer == 'bm25':
        from models.Modules.DRM import BM25_Reducer
        reducer = BM25_Reducer(manager)

    # if manager.aggregator == 'rnn':
    #     from models.Encoders.RNN import RNN_User_Encoder
    #     aggregator = RNN_User_Encoder(manager)
    # elif manager.aggregator == 'avg':
    #     from models.Encoders.Pooling import Average_Pooling
    #     aggregator = Average_Pooling(manager)
    # elif manager.aggregator == 'attn':
    #     from models.Encoders.Pooling import Attention_Pooling
    #     aggregator = Attention_Pooling(manager)
    # else:
    #     aggregator = None

    ttms = TTMS(manager, embedding, encoderN, encoderU, reducer).to(rank)

    if dist:
        ttms = DDP(ttms, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if manager.mode == 'dev':
        manager.evaluate(ttms, loaders[0], load=True)

    elif manager.mode == 'train':
        manager.train(ttms, loaders)

    elif manager.mode == 'tune':
        manager.tune(ttms, loaders)

    elif manager.mode == 'test':
        manager.test(ttms, loaders[0])

    elif manager.mode == 'inspect':
        manager.inspect(ttms, loaders[0])

    if dist:
        cleanup()

if __name__ == "__main__":
    manager = load_manager()
    manager.hidden_dim = 768
    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager, True),
            nprocs=manager.world_size
        )
    else:
        main(manager.device, manager, dist=False)