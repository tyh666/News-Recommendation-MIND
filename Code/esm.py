from models.Encoders.FIM import FIM_Encoder
from utils.utils import prepare,load_config
from data.configs.drm import config

from models.Encoders.CNN import CNN_Encoder
from models.Encoders.FIM import FIM_Encoder
from models.Encoders.RNN import RNN_User_Encoder
from models.Interactors.CNN import CNN_Interactor
from models.Modules.DRM import DRM_Matching
# from models.Modules.TFM import TFM
from models.ESM import ESM

if __name__ == "__main__":

    config = load_config(config)
    vocab, loaders = prepare(config)

    # encoderN = CNN_Encoder(config, vocab)
    encoderN = FIM_Encoder(config,vocab)
    encoderU = RNN_User_Encoder(encoderN.hidden_dim)
    docReducer = DRM_Matching(config.k)
    # termFuser = TFM(config.his_size, config.k)
    interactor = CNN_Interactor(config.title_size, config.k * config.his_size, encoderN.level, encoderN.hidden_dim)

    esm = ESM(config, encoderN, encoderU, docReducer, None, interactor).to(config.device)

    if config.mode == 'dev':
        esm.evaluate(config,loaders[0],loading=True)

    elif config.mode == 'train':
        esm.fit(config, loaders)

    elif config.mode == 'tune':
        esm.tune(config, loaders)

    elif config.mode == 'test':
        esm.test(config, loaders[0])
