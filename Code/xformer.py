import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.Manager import Manager
from models.XFormer import XFormer

def main(rank, manager):
    """ train/dev/test/tune the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    manager.setup(rank)
    loaders = manager.prepare()

    xformer = XFormer(manager).to(rank)

    if manager.world_size > 1:
        xformer = DDP(xformer, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if manager.mode == 'dev':
        manager.evaluate(xformer, loaders, load=True)

    elif manager.mode == 'train':
        manager.train(xformer, loaders)

    elif manager.mode == 'test':
        manager.test(xformer, loaders)

    elif manager.mode == 'inspect':
        manager.inspect(xformer, loaders)

    elif manager.mode == 'encode':
        manager.encode(xformer, loaders)


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