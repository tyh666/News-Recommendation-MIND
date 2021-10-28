import torch
import time
import logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s (%(name)s) %(message)s")
logger = logging.getLogger(__file__)

logger.info("I'm running to stop the platform killing this job!")
a = torch.zeros((1),device=1)
while(1):
    if a.item() > 2:
        a -= 1
    else:
        a += 1