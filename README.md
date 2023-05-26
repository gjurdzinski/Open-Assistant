Forked from https://github.com/LAION-AI/Open-Assistant.

Run example:
```bash
python \
   training/trainer.py \
   training/configs/multi/deberta-small-newsroom-inf.yml
```

## Configs for _multi training_
Training configs allow to run multiple fine-tunings of the same base model on different splits. This is to allow training on 0%, 10%, ..., 100% of the training data, if the dataset is prepared to have training set split into subsets.

Eg. [`training/configs/multi/deberta-small-newsroom-inf.yml`](https://github.com/gjurdzinski/Open-Assistant/blob/314f0e4deab1fcc656e8e8eb3f42c38654594a34/training/configs/multi/deberta-small-newsroom-inf.yml) utilises the fact that for Newsroom we have training set split into `train_0`, `train_1`, ..., `train_9`, each containing ~10% of the training data. Each run is given a list of splits which are together used for training.
