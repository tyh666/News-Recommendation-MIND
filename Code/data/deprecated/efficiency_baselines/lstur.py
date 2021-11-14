import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.Manager import Manager
from models.LSTUR import LSTUR

def main(rank, manager):
    """ train/dev/test/tune the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    manager.setup(rank)
    loaders = manager.prepare()

    lstur = LSTUR(manager).to(rank)

    if manager.world_size > 1:
        lstur = DDP(lstur, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if manager.mode == 'dev':
        manager.evaluate(lstur, loaders, load=True)

    elif manager.mode == 'train':
        manager.train(lstur, loaders)

    elif manager.mode == 'test':
        manager.test(lstur, loaders)

    elif manager.mode == 'inspect':
        manager.inspect(lstur, loaders)

    elif manager.mode == 'encode':
        manager.encode(lstur, loaders)

    elif manager.mode == 'recall':
        manager.recall(lstur, loaders)


if __name__ == "__main__":
    manager = Manager()

    # default settings
    manager.reducer = 'none'
    manager.hidden_dim = 150
    manager.embedding_dim = 300

    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager,),
            nprocs=manager.world_size,
            join=True
        )
    else:
        main(manager.device, manager)