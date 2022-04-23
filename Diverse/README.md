## Generating Smooth Pose Sequences for Diverse Human Motion Prediction
![Loading GSPS Overview](data/pipeline.png "GSPS overview")

This is official implementation for the paper

[_Generating Smooth Pose Sequences for Diverse Human Motion Prediction_](https://arxiv.org/abs/2108.08422). In ICCV 21.

Wei Mao, Miaomiao Liu, Mathieu Salzmann. 

[[paper](https://arxiv.org/abs/2108.08422)] [[talk](https://www.youtube.com/watch?v=IDu08Lh-qPU&ab_channel=anucvml)]

### Dependencies
* Python >= 3.8
* [PyTorch](https://pytorch.org) >= 1.8
* Tensorboard

tested on pytorch == 1.8.1

### Datasets
* We follow the data preprocessing steps ([DATASETS.md](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md)) inside the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repo.
* Given the processed dataset, we further compute the multi-modal future for each motion sequence. All data needed can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1sb1n9l0Na5EqtapDVShOJJ-v6o-GZrIJ?usp=sharing) and place all the dataset in ``data`` folder inside the root of this repo.

### Training and Evaluation
* We provide 4 YAML configs inside ``motion_pred/cfg``: `[dataset].yml` and `[dataset]_nf.yml` for training generator and normalizing flow respectively. These configs correspond to pretrained models inside ``results``.
* The training and evaluation command is included in ``run.sh`` file.

### Citing

If you use our code, please cite our work

```
@inproceedings{mao2021generating,
  title={Generating Smooth Pose Sequences for Diverse Human Motion Prediction},
  author={Mao, Wei and Liu, Miaomiao and Salzmann, Mathieu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13309--13318},
  year={2021}
}
```

### Acknowledgments

The overall code framework (dataloading, training, testing etc.) is adapted from [DLow](https://github.com/Khrylx/DLow). 

### Licence
MIT
