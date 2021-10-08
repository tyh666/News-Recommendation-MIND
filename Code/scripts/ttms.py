import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.Manager import Manager
from models.TTMS import TTMS

def main(rank, manager):
    """ train/dev/test/tune the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
        conig
    """
    manager.setup(rank)
    loaders = manager.prepare()

    from models.Embeddings.BERT import BERT_Embedding
    embedding = BERT_Embedding(manager)

    if manager.encoderN == 'cnn':
        from models.Encoders.CNN import CNN_Encoder
        encoderN = CNN_Encoder(manager)
    elif manager.encoderN == 'bert':
        from models.Encoders.BERT import BERT_Encoder
        encoderN = BERT_Encoder(manager)
    elif manager.encoderN == 'mha':
        from models.Encoders.MHA import MHA_Encoder
        encoderN = MHA_Encoder(manager)

    if manager.encoderU == 'rnn':
        from models.Encoders.RNN import RNN_User_Encoder
        encoderU = RNN_User_Encoder(manager)
    elif manager.encoderU == 'avg':
        from models.Encoders.Pooling import Average_Pooling
        encoderU = Average_Pooling(manager)
    elif manager.encoderU == 'attn':
        from models.Encoders.Pooling import Attention_Pooling
        encoderU = Attention_Pooling(manager)
    elif manager.encoderU == 'mha':
        from models.Encoders.MHA import MHA_User_Encoder
        encoderU = MHA_User_Encoder(manager)
    elif manager.encoderU == 'lstur':
        from models.Encoders.RNN import LSTUR
        encoderU = LSTUR(manager)

    if manager.reducer in ['matching', 'bow']:
        from models.Modules.DRM import Matching_Reducer
        reducer = Matching_Reducer(manager)
    elif manager.reducer in ['bm25', "entity", "first"]:
        from models.Modules.DRM import Slicing_Reducer
        reducer = Slicing_Reducer(manager)

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

    if manager.world_size > 1:
        ttms = DDP(ttms, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if manager.mode == 'dev':
            manager.evaluate(ttms, loaders, load=True)

    elif manager.mode == 'train':
        manager.train(ttms, loaders)

    elif manager.mode == 'test':
        manager.test(ttms, loaders)

    elif manager.mode == 'inspect':
        manager.inspect(ttms, loaders)


if __name__ == "__main__":
    manager = Manager()
    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager,),
            nprocs=manager.world_size,
            join=True
        )
    else:
        main(manager.device, manager)