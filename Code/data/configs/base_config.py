class BaseConfig:
    @classmethod
    def info(self):
        return ", ".join(["{}:{}".format(k,v) for k,v in vars(self).items() if not k.startswith('__')])