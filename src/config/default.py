from typing import Optional

from yacs.config import CfgNode as CN

_C = CN()


def get_cfg(config_file: Optional[str] = None, new_allowed: bool = True) -> CN:
    cfg = _C.clone()
    cfg.set_new_allowed(new_allowed)
    if config_file:
        cfg.merge_from_file(config_file)
    return cfg


def merge_cfg(cfg_1: CN, cfg_2: CN) -> CN:
    cfg_1 = cfg_1.clone()
    cfg_1.merge_from_other_cfg(cfg_2)
    return cfg_1
