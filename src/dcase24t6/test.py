#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_OFFLINE"] = "FALSE"
os.environ["HF_HUB_OFFLINE"] = "FALSE"

import multiprocessing as mp

import hydra
from omegaconf import DictConfig

from dcase24t6.train import train

# Set multiprocessing start method to 'spawn'
mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base=None,
    config_path=os.environ.get("HYDRA_CONFIG_PATH"),
    config_name="test",
)
def test(cfg: DictConfig) -> None | float:
    return train(cfg)


if __name__ == "__main__":
    test()
