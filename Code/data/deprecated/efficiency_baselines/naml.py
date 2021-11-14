import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.Manager import Manager
from models.NAML import NAML

def main(rank, manager):
    """ train/dev/test/tune the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    manager.setup(rank)
    loaders = manager.prepare()

    model = NAML(manager).to(rank)

    if manager.world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if manager.mode == 'dev':
        manager.evaluate(model, loaders, load=True)

    elif manager.mode == 'train':
        manager.train(model, loaders)

    elif manager.mode == 'test':
        manager.test(model, loaders)

    elif manager.mode == 'inspect':
        manager.inspect(model, loaders)

    elif manager.mode == 'encode':
        manager.encode(model, loaders)

    elif manager.mode == 'recall':
        manager.recall(model, loaders)


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