<div align="center">
<h1> Neural Source-Filter BigVGAN </h1>

code is adapted for PlayVoice/lora-svc [WIP]

</div>

## Dataset preparation

Necessary pre-processing:
- 1 accompaniment separation, [UVR](https://github.com/Anjok07/ultimatevocalremovergui)
- 2 cut audio, less than 30 seconds for whisper, [slicer](https://github.com/flutydeer/audio-slicer)

then put the dataset into the data_raw directory according to the following file structure
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

- 4， extract mel

    > python prepare/preprocess_spec.py -w data_bigvgan/waves-32k/ -s data_bigvgan/mel

- 5， generate training index

    > python prepare/preprocess_train.py

- 6， training file debugging

    > python prepare/preprocess_zzz.py -c configs/maxgan.yaml

```shell
data_bigvgan/
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
- 0， if fine-tuning based on the pre-trained model, you need to download the pre-trained model: [maxgan_pretrain_32K.pth](https://github.com/PlayVoice/lora-svc/releases/download/v_final/maxgan_pretrain_32K.pth)

    > set pretrain: "./maxgan_pretrain_32K.pth" in configs/maxgan.yaml，and adjust the learning rate appropriately, eg 1e-5

- 1， set working directory

    > export PYTHONPATH=$PWD

- 2， start training

    > python svc_trainer.py -c configs/maxgan.yaml -n svc

- 3， resume training

    > python svc_trainer.py -c configs/maxgan.yaml -n svc -p chkpt/svc/***.pth

- 4， view log

    > tensorboard --logdir logs/


## Inference

- 0, use this command if you want a GUI that does all the commands below:

    > python3 svcgui.py

- 1， set working directory

    > export PYTHONPATH=$PWD

- 2， export inference model

    > python svc_export.py --config configs/maxgan.yaml --checkpoint_path chkpt/svc/***.pt

- 3， use whisper to extract content encoding, without using one-click reasoning, in order to reduce GPU memory usage

    > python whisper/inference.py -w test.wav -p test.ppg.npy

    generate test.ppg.npy; if no ppg file is specified in the next step, generate it automatically

- 4， extract the F0 parameter to the csv text format, open the csv file in Excel, and manually modify the wrong F0 according to Audition or SonicVisualiser

    > python pitch/inference.py -w test.wav -p test.csv

- 5，specify parameters and infer

    > python svc_inference.py --config configs/maxgan.yaml --model maxgan_g.pth --spk ./data_bigvgan/singers/your_singer.npy --wave test.wav --ppg test.ppg.npy --pit test.csv

    when --ppg is specified, when the same audio is reasoned multiple times, it can avoid repeated extraction of audio content codes; if it is not specified, it will be automatically extracted;

    when --pit is specified, the manually tuned F0 parameter can be loaded; if not specified, it will be automatically extracted;

    generate files in the current directory:svc_out.wav

    | args |--config | --model | --spk | --wave | --ppg | --pit | --shift |
    | :---:  | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | name | config path | model path | speaker | wave input | wave ppg | wave pitch | pitch shift |

#

<div align="center">
<h1> Singing Voice Conversion based on Whisper & neural source-filter BigVGAN </h1>

<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/lora-svc">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/lora-svc">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/lora-svc">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/lora-svc">
</div>

```
Black technology based on the three giants of artificial intelligence:

OpenAI's whisper, 680,000 hours in multiple languages

Nvidia's bigvgan, anti-aliasing for speech generation

Microsoft's adapter, high-efficiency for fine-tuning
```

## Source of code and References

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/mindslab-ai/univnet [[paper]](https://arxiv.org/abs/2106.07889)

https://github.com/NVIDIA/BigVGAN [[paper]](https://arxiv.org/abs/2206.04658)


## Encouragement
If you adopt the code or idea of this project, please list it in your project, which is the basic criterion for the continuation of the open source spirit.

如果你采用了本项目的代码或创意，请在你的项目中列出，这是开源精神得以延续的基本准则。

このプロジェクトのコードやアイデアを採用した場合は、オープンソースの精神が続く基本的なガイドラインであるプロジェクトにリストしてください。
