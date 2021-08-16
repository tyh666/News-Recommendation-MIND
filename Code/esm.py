from utils.utils import prepare, load_manager, setup, cleanup

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# from models.Interactors.BERT_Overlook import BERT_Interactor

from models.Modules.DRM import Document_Reducer
# from models.Modules.TFM import TFM
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
    if manager.encoderU == 'rnn':
        from models.Encoders.RNN import RNN_User_Encoder
        encoderU = RNN_User_Encoder(manager)

    docReducer = Document_Reducer(manager)
    # termFuser = TFM(manager.his_size, manager.k)
    # interactor = CNN_Interactor(manager)
    if manager.interactor == 'onepass':
        from models.Interactors.BERT import BERT_Onepass_Interactor
        interactor = BERT_Onepass_Interactor(manager)

    elif manager.interactor == 'cnn':
        from models.Interactors.CNN import CNN_Interactor
        interactor = CNN_Interactor(manager)

    esm = ESM(manager, embedding, encoderN, encoderU, docReducer, None, interactor).to(rank)

    if dist:
        esm = DDP(esm, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if manager.mode == 'dev':
        manager.evaluate(esm, loaders[0], load=True)

    elif manager.mode == 'train':
        manager.train(esm, loaders)

    elif manager.mode == 'tune':
        manager.tune(esm, loaders)

    elif manager.mode == 'test':
        manager.test(esm, loaders[0])

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