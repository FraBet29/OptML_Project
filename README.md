# [CS-439] Comparison of second-order optimizers (???)

This repository contains implementations of two second-order optimizers - AdaHessian and AdaSub - and the utilities to train them on the SQuAD v2.0 dataset for extractive question answering. All the runs are using the [ALBERT](https://arxiv.org/abs/1909.11942) model with base configuration (~11M params).

Our best result on AdaHessian can be reproduced by calling

```
python train.py --optimizer adahessian \
                --hessian_power 0.5 \
                --lr 0.003
```

The repository is organized as follows:

```
./
├── src/                    # Project source code
│   ├── optimizers/         # Optimizer implementations
|   │   ├── adahessian.py   
|   │   ├── adasub.py       
│   ├── utils/              # Utility functions
|   │   ├── data_utils.py   
|   │   ├── train_utils.py  
|   │   ├── eval_utils.py
├── data/                   # Raw SQuAD v2.0 data
│   ├── raw_data/
|   │   ├── train-v2.0.json
|   │   ├── dev-v2.0.json
├── train.py                # Training script
├── README.md
└── requirements.txt
```

### `src/optimizers/`

Contains source code for the two second-order optimizers. Both implementations are adapted from the [AdaSub repository](https://github.com/Jvictormata/adasub).

- `adahessian.py`: Implements the AdaHessian optimizer, as described in the [original paper](https://arxiv.org/abs/2006.00719). It is taken from the AdaSub repository, which modified the [original source code](https://github.com/amirgholami/adahessian) for training a custom CNN. 
- `adasub.py`: Implements the [AdaSub optimizer](https://arxiv.org/abs/2310.20060). We restructured the original code for faster computation of Hessian-vector products. Namely, the original AdaSub code called `torch.autograd.grad` for each model parameter to obtain the products. Similarly to AdaHessian implementation, we call the function once for a group of parameters, speeding the algorithm up to 10 times.


### `utils/`

Contains the utility functions for model training.

- `data_utils.py`: Consists of the `load_and_cache_examples` function. It takes raw .json SQuAD data, converts it to a torch `Dataset` instance and a collection of `SquadExample`s and `SquadFeature`s, and saves them to disk. The latter two types are integrated into `HuggingFace` and allow for easier evaluation on SQuAD.
- `train_utils.py`: Contains the function for model training.
- `eval_utils.py`: Contains the function for model evaluation. It leverages the `squad_evaluate` function from `HuggingFace` which implements advanced methods for robust prediction of the answer to the SQuAD question.

### `data/raw_data/`

Contains the raw train and evaluation splits of the SQuAD v2.0 data, used by the `load_and_cache_examples` method to transforming to the appropriate form. 

### `train.py`

The script invoking the full training pipeline, including data processing.