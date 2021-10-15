import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.Manager import Manager
from models.PLM import PLM

def main(rank, manager):
    """ train/dev/test/tune the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
        conig
    """
    manager.setup(rank)
    loaders = manager.prepare()

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

    plm = PLM(manager, encoderU).to(rank)

    if manager.world_size > 1:
        plm = DDP(plm, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if manager.mode == 'dev':
        manager.evaluate(plm, loaders, load=True)

    elif manager.mode == 'train':
        manager.train(plm, loaders)

    elif manager.mode == 'test':
        manager.test(plm, loaders)

    elif manager.mode == 'inspect':
        manager.inspect(plm, loaders)


if __name__ == "__main__":
    manager = Manager()
    manager.hidden_dim = 768

    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager,),
            nprocs=manager.world_size,
            join=True
        )
    else:
        main(manager.device, manager)