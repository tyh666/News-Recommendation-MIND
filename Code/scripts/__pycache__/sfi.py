import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.Manager import Manager
from models.SFI import SFI

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

    if manager.selector == 'recent':
        from models.Modules.HSM import Recent_Selector
        selector = Recent_Selector(manager)
    else:
        from models.Modules.HSM import SFI_Selector
        selector = SFI_Selector(manager)

    from models.Rankers.BERT import BERT_Original_Ranker
    ranker = BERT_Original_Ranker(manager)

    sfi = SFI(manager, embedding, encoderN, selector, ranker).to(manager.device)

    if manager.world_size > 1:
        sfi = DDP(sfi, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if manager.mode == 'dev':
        manager.evaluate(sfi,loaders[0],load=True)

    elif manager.mode == 'train':
        manager.train(sfi, loaders)

    elif manager.mode == 'tune':
        manager.tune(sfi, loaders)

    elif manager.mode == 'test':
        manager.test(sfi, loaders[0])


if __name__ == "__main__":
    manager = Manager()
    manager.reducer = 'none'

    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager,),
            nprocs=manager.world_size,
            join=True
        )
    else:
        main(manager.device, manager)