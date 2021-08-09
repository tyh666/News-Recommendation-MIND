import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import prepare, load_manager, setup, cleanup

from models.Interactors.BERT_Overlook import BERT_Interactor

def main(rank, manager, dist=False):
    """ train/dev/test/tune the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
        conig
    """
    setup(rank, manager)

    if manager.mode == 'encode':
        vocab, loaders = prepare(manager, news=True)
    else:
        vocab, loaders = prepare(manager)

    if manager.embedding == 'glove':
        from models.Embeddings.GLOVE import GLOVE_Embedding
        embedding = GLOVE_Embedding(manager, vocab)

    elif manager.embedding == 'bert':
        from models.Embeddings.BERT import BERT_Embedding
        embedding = BERT_Embedding(manager)

    if manager.encoderN == 'fim':
        from models.Encoders.FIM import FIM_Encoder
        encoder = FIM_Encoder(manager)

    # elif manager.encoderN == 'mha':
    #     from models.Encoders.MHA import MHA_Encoder
    #     encoder = MHA_Encoder(manager)

    # elif manager.encoderN == 'npa':
    #     manager.user_dim = 200
    #     manager.query_dim = 200
    #     manager.filter_num = 400
    #     from models.Encoders.NPA import NPA_Encoder
    #     encoder = NPA_Encoder(manager, vocab, len(loaders[0].dataset.uid2index))

    # elif manager.encoderN == 'nrms':
    #     manager.value_dim = 16
    #     manager.query_dim = 200
    #     manager.head_num = 16
    #     from models.Encoders.MHA import NRMS_Encoder
    #     encoder = NRMS_Encoder(manager, vocab)

    elif manager.encoderN == 'cnn':
        from models.Encoders.CNN import CNN_Encoder
        encoder = CNN_Encoder(manager)

    elif manager.encoderN == 'rnn':
        from models.Encoders.RNN import RNN_Encoder
        encoder = RNN_Encoder(manager)

    elif manager.encoderN == 'pipeline':
        from models.Encoders.Pipeline import Pipeline_Encoder
        encoder = Pipeline_Encoder(manager)

    else:
        raise ValueError("Undefined Encoder:{}".format(manager.encoderN))

    if manager.interactor == 'fim':
        from models.Interactors.FIM import FIM_Interactor
        interactor = FIM_Interactor(manager)

    # elif manager.interactor == 'knrm':
    #     from models.Interactors.KNRM import KNRM_Interactor
    #     interactor = KNRM_Interactor()

    # elif manager.interactor == '2dcnn':
    #     from models.Interactors.CNN import CNN_Interator
    #     interactor = CNN_Interator(manager.k)

    # elif manager.interactor == 'mha':
    #     from models.Interactors.MHA import MHA_Interactor
    #     interactor = MHA_Interactor(encoder.hidden_dim)

    else:
        raise ValueError("Undefined Interactor:{}".format(manager.interactor))

    # if manager.multiview:
    #     if manager.coarse:
    #         from models.SFI import SFI_unified_MultiView
    #         sfi = SFI_unified_MultiView(manager, encoder, interactor).to(manager.device)

    #     else:
    #         from models.SFI import SFI_MultiView
    #         sfi = SFI_MultiView(manager, encoder, interactor).to(manager.device)

    # else:
    if manager.coarse:
        from models.SFI import SFI_unified
        sfi = SFI_unified(manager, embedding, encoder, interactor).to(manager.device)

    else:
        from models.SFI import SFI
        sfi = SFI(manager, embedding, encoder, interactor).to(manager.device)

    if dist:
        sfi = DDP(sfi, device_ids=[rank], output_device=rank)

    if manager.mode == 'dev':
        manager.evaluate(sfi,loaders[0],loading=True)

    elif manager.mode == 'train':
        manager.train(sfi, loaders)

    elif manager.mode == 'tune':
        manager.tune(sfi, loaders)

    elif manager.mode == 'test':
        manager.test(sfi, loaders[0])

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