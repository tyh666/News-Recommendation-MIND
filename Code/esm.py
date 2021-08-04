from models.Encoders.FIM import FIM_Encoder
from utils.utils import prepare,load_config, setup, cleanup

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from models.Embeddings.BERT import BERT_Embedding
from models.Interactors.BERT import BERT_Interactor
from models.Encoders.CNN import CNN_Encoder
from models.Encoders.FIM import FIM_Encoder
from models.Encoders.RNN import RNN_User_Encoder
from models.Interactors.CNN import CNN_Interactor
from models.Modules.DRM import DRM_Matching
# from models.Modules.TFM import TFM
from models.ESM import ESM

def main(rank, config, dist=False):
    """ train/dev/test/tune the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
        conig
    """
    if dist:
        setup()

    vocab, loaders = prepare(config)

    embedding = BERT_Embedding(config)
    encoderN = CNN_Encoder(config)
    encoderU = RNN_User_Encoder(encoderN.hidden_dim)
    docReducer = DRM_Matching(config.k)
    # termFuser = TFM(config.his_size, config.k)
    # interactor = CNN_Interactor(config)
    interactor = BERT_Interactor(config)
    esm = ESM(config, embedding, encoderN, encoderU, docReducer, None, interactor).to(config.device)

    if dist:
        esm = DDP(esm, device_ids=[rank])

    if config.mode == 'dev':
        esm.evaluate(config,loaders[0],loading=True)

    elif config.mode == 'train':
        esm.fit(config, loaders)

    elif config.mode == 'tune':
        esm.tune(config, loaders)

    elif config.mode == 'test':
        esm.test(config, loaders[0])

    if dist:
        cleanup()

if __name__ == "__main__":
    config = load_config()
    if config.world_size > 0:
        mp.spawn(
            main,
            args=(config, True),
            nprocs=config.world_size
        )
    else:
        main(config.device, config)