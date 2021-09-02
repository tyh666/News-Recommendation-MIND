import torch.multiprocessing as mp
from torch.nn.modules.rnn import RNN
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import prepare, load_manager, setup, cleanup
from models.ESM import ESM

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
    elif manager.encoderN == 'rnn':
        from models.Encoders.RNN import RNN_Encoder
        encoderN = RNN_Encoder(manager)
    elif manager.encoderN == 'mha':
        from models.Encoders.MHA import MHA_Encoder
        encoderN = MHA_Encoder(manager)

    if manager.encoderU == 'avg':
        from models.Encoders.Pooling import Average_Pooling
        encoderU = Average_Pooling(manager)
    elif manager.encoderU == 'cnn':
        from models.Encoders.CNN import CNN_User_Encoder
        encoderU = CNN_User_Encoder(manager)
    elif manager.encoderU == 'rnn':
        from models.Encoders.RNN import RNN_User_Encoder
        encoderU = RNN_User_Encoder(manager)
    elif manager.encoderU == 'mha':
        from models.Encoders.MHA import MHA_User_Encoder
        encoderU = MHA_User_Encoder(manager)

    if manager.reducer in ['matching', 'bow']:
        from models.Modules.DRM import Matching_Reducer
        reducer = Matching_Reducer(manager)
    elif manager.reducer in ['bm25', "entity", "first"]:
        from models.Modules.DRM import Slicing_Reducer
        reducer = Slicing_Reducer(manager)


    # if manager.fuser == 'union':
    #     from models.Modules.TFM import Union_Fuser
    #     termFuser = Union_Fuser(manager)

    if manager.ranker == 'onepass':
        from models.Rankers.BERT import BERT_Onepass_Ranker
        ranker = BERT_Onepass_Ranker(manager)
    elif manager.ranker == 'original':
        from models.Rankers.BERT import BERT_Original_Ranker
        ranker = BERT_Original_Ranker(manager)
    elif manager.ranker == 'cnn':
        from models.Rankers.CNN import CNN_Ranker
        ranker = CNN_Ranker(manager)

    esm = ESM(manager, embedding, encoderN, encoderU, reducer, None, ranker).to(rank)

    if dist:
        esm = DDP(esm, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if manager.mode == 'dev':
        manager.evaluate(esm, loaders[0], load=True)

    elif manager.mode == 'train':
        manager.train(esm, loaders)

    elif manager.mode == 'tune':
        manager.tune(esm, loaders)

    elif manager.mode == 'test':
        manager.test(esm, loaders[0])

    elif manager.mode == 'inspect':
        manager.inspect(esm, loaders[0])

    if dist:
        cleanup()

if __name__ == "__main__":
    manager = load_manager()
    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager, True),
            nprocs=manager.world_size
        )
    else:
        main(manager.device, manager, dist=False)