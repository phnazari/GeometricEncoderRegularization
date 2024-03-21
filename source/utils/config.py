import os
import yaml
import datetime


class Config:
    """
    Config class that merges base and personal config file.
    Has some useful functions for ease of use.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, file=None):
        if hasattr(self, "_initialized"):
            return
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

        if file is None:
            file = "conf.yml"
        else:
            file = f"{file}.yml"

        def load_recursive(config, stack):
            if config in stack:
                raise AssertionError("Attempting to build recursive configuration.")

            config_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            config_path = os.path.join(config_path, "configs", config)
            with open(config_path, "r", encoding="UTF-8") as file_handle:
                cfg = yaml.safe_load(file_handle)

            base = (
                {}
                if "extends" not in cfg
                else load_recursive(cfg["extends"], stack + [config])
            )
            base = _recursive_update(base, cfg)
            return base

        self._config = load_recursive(file, [])
        self.initialized = True

    def __getitem__(self, item):
        return self._config.__getitem__(item)


def _recursive_update(base: dict, cfg: dict) -> dict:
    for k, v in cfg.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base
