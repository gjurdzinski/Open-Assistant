#!/bin/bash

python \
    model/reward/instructor/trainer.py\
    model/reward/instructor/configs/multi/deberta-base-newsroom-inf.yml
python \
    model/reward/instructor/trainer.py\
    model/reward/instructor/configs/multi/deberta-large-newsroom-inf.yml