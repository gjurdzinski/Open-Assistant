#!/bin/bash

python \
    model/reward/instructor/trainer.py\
    model/reward/instructor/configs/multi/reward-model-deberta-v3-small-newsroom-inf-multi-train.yml
python \
    model/reward/instructor/trainer.py\
    model/reward/instructor/configs/multi/reward-model-reward-deberta-v3-base-newsroom-inf-multi-train.yml
python \
    model/reward/instructor/trainer.py\
    model/reward/instructor/configs/multi/reward-model-reward-deberta-v3-large-newsroom-inf-multi-train.yml