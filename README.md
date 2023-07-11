<div align="center">
<h1> Neural Source-Filter BigVGAN </h1>

code is adapted for PlayVoice/lora-svc [WIP]

</div>

## Dataset preparation

Put the dataset into the data_raw directory according to the following file structure
```shell
data_raw
├───speaker0
│   ├───000001.wav
│   ├───...
│   └───000xxx.wav
└───speaker1
    ├───000001.wav
    ├───...
    └───000xxx.wav
```

## Install dependencies

- 1 software dependency
  
  > pip install -r requirements.txt

## Data preprocessing

- 1， re-sampling: 32kHz

    > python prepare/preprocess_a.py -w ./data_raw -o ./data_bigvgan/waves-32k

- 3， extract pitch

    > python prepare/preprocess_f0.py -w data_bigvgan/waves-32k/ -p data_bigvgan/pitch

- 4， extract mel: [100, length]

    > python prepare/preprocess_spec.py -w data_bigvgan/waves-32k/ -s data_bigvgan/mel

- 5， generate training index

    > python prepare/preprocess_train.py

- 6， training file debugging

    > python prepare/preprocess_zzz.py -c configs/maxgan.yaml

```shell
data_bigvgan/
│
└── waves-32k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── pitch
│    └── speaker0
│    │      ├── 000001.pit.npy
│    │      └── 000xxx.pit.npy
│    └── speaker1
│           ├── 000001.pit.npy
│           └── 000xxx.pit.npy
└── mel
     └── speaker0
     │      ├── 000001.mel.pt
     │      └── 000xxx.mel.pt
     └── speaker1
            ├── 000001.mel.pt
            └── 000xxx.mel.pt

```

## Train

- 1， start training

    > python nsf_bigvgan_trainer.py -c configs/nsf_bigvgan.yaml -n nsf_bigvgan

- 2， resume training

    > python nsf_bigvgan_trainer.py -c configs/nsf_bigvgan.yaml -n nsf_bigvgan -p chkpt/nsf_bigvgan/***.pth

- 3， view log

    > tensorboard --logdir logs/

## Inference


## Source of code and References

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/NVIDIA/BigVGAN [[paper]](https://arxiv.org/abs/2206.04658)


## Encouragement
If you adopt the code or idea of this project, please list it in your project, which is the basic criterion for the continuation of the open source spirit.

如果你采用了本项目的代码或创意，请在你的项目中列出，这是开源精神得以延续的基本准则。

このプロジェクトのコードやアイデアを採用した場合は、オープンソースの精神が続く基本的なガイドラインであるプロジェクトにリストしてください。
