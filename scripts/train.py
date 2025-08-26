import logging

import comet_ml
import hydra
from omegaconf import DictConfig

from experimenting.utils.trainer import HydraTrainer

logging.basicConfig(level=logging.INFO)


# @hydra.main(config_path='../confs/train/config.yaml')
# @hydra.main(
#     version_base=None,                # 去掉那个警告
#     config_path="../confs/train",     # 只写目录
#     config_name="config"              # 配置文件名（不带 .yaml 扩展名）
# )

@hydra.main(config_path="../confs/train", config_name="config")


def main(cfg: DictConfig) -> None:
    trainer = HydraTrainer(cfg)
    trainer.fit()
    trainer.test()


if __name__ == '__main__':
    main()


