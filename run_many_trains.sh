#!/bin/bash

python \
    model/reward/instructor/trainer.py\
    model/reward/instructor/configs/multi/temp.yml
python \
    model/reward/instructor/trainer.py\
    model/reward/instructor/configs/multi/deberta-base-newsroom-inf.yml
