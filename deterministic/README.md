 # Deterministic Human Motion Forecasting on Human3.6M
![Loading Architecture Overview](../images/deterministic.png "Architecture")
---
### Dependencies
* Python >= 3.8
* [PyTorch](https://pytorch.org) >= 1.9
* Tensorboard
* matplotlib
* tqdm
* argparse

 ### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).
 
Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```

Put the all downloaded datasets in ./data directory.

### Train
The arguments for running the code are defined in [parser.py](utils/parser.py). We have used the following commands for training the network on Human3.6M with skeleton representation:
 
```bash
python main.py --input_n [number of historical sequence frames] --output_n [maximum number of predicted frames] --skip_rate [sampling rate] --n_pre [number of dct coefficients] --data_dir ./data --version [long / short]
 ```

It's able to predict the future motion considering the global translation

```bash
python main.py --input_n [number of historical sequence frames] --output_n [maximum number of predicted frames] --skip_rate [sampling rate] --n_pre [number of dct coefficients] --global_translation --data_dir ./data --version [long / short]
 ```

We provide the pretrained model with 10 historical sequence frames and 25 future predicted frames following the literature.
 ### Test
 To test on the pretrained model, we have used the following commands:
 ```bash
 python main.py --input_n [number of historical sequence frames] --output_n [maximum number of predicted frames] --test_output_n [index of the test frame] --skip_rate [sampling rate] --n_pre [number of dct coefficients] --mode test --model_path ./checkpoints/CKPT_3D_H36M --data_dir ./data --version [long / short]
  ```

 With global translation, we have the following commands:
  ```bash
 python main.py --input_n [number of historical sequence frames] --output_n [maximum number of predicted frames] --test_output_n [index of the test frame] --skip_rate [sampling rate] --n_pre [number of dct coefficients] --mode test --model_path ./checkpoints/CKPT_3D_H36M --global_translation --data_dir ./data --version [long / short]
  ```

### Visualization
 For visualizing from a pretrained model, we have used the following commands:
 ```bash
 python main.py --input_n [number of historical sequence frames] --output_n [maximum number of predicted frames] --skip_rate [sampling rate] --n_pre [number of dct coefficients] --mode viz --model_path ./checkpoints/CKPT_3D_H36M --n_viz 5 --data_dir ./data --version [long / short]
 ```
With global translation, we have the following commands:
 ```bash
 python main.py --input_n [number of historical sequence frames] --output_n [maximum number of predicted frames] --skip_rate [sampling rate] --n_pre [number of dct coefficients] --mode viz --model_path ./checkpoints/CKPT_3D_H36M --n_viz 5 --global_translation --data_dir ./data --version [long / short]
 ```
 
 ### Acknowledgments
 
 This code is based on the implementations of [STSGCN](https://github.com/FraLuca/STSGCN).
