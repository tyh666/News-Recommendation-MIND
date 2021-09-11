import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import prepare, setup, cleanup
from utils.Manager import Manager
from models.TTM import TTM

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

    # if manager.reducer == 'matching':
    #     from models.Modules.DRM import Matching_Reducer
    #     docReducer = Matching_Reducer(manager)
    # elif manager.reducer == 'bm25':
    #     from models.Modules.DRM import BM25_Reducer
    #     docReducer = BM25_Reducer(manager)
    ttm = TTM(manager, embedding, encoderN, encoderU).to(rank)

    if dist:
        ttm = DDP(ttm, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if manager.mode == 'dev':
        manager.evaluate(ttm, loaders[0], load=True)

    elif manager.mode == 'train':
        manager.train(ttm, loaders)

    elif manager.mode == 'tune':
        manager.tune(ttm, loaders)

    elif manager.mode == 'test':
        manager.test(ttm, loaders[0])

    if dist:
        cleanup()

if __name__ == "__main__":
    manager = Manager()
    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager, True),
            nprocs=manager.world_size
        )
    else:
        main(manager.device, manager, dist=False)