 # Deterministic Human Motion Forecasting on Human3.6M
 
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

Put the all downloaded datasets in ../datasets directory.

### Train
The arguments for running the code are defined in [parser.py](utils/parser.py). We have used the following commands for training the network on Human3.6M with skeleton representation:
 
```bash
 python main_h36_3d.py --input_n ${number of historical sequence frames} --output_n ${maximum number of predicted frames} --skip_rate 1 
 ```

It's able to predict the future motion considering the global translation

```bash
 python main_h36_3d.py --input_n ${number of historical sequence frames} --output_n ${maximum number of predicted frames} --skip_rate 1 --global_translation
 ```

We provide the pretrained model with 10 historical sequence frames and 10 or 25 future predicted frames following the literature.
 ### Test
 To test on the pretrained model, we have used the following commands:
 ```bash
 python main_h36_3d.py --input_n ${number of historical sequence frames} --output_n ${maximum number of predicted frames} --test_output_n ${sequence number of the test frame} --skip_rate 1 --mode test --model_path ./checkpoints/CKPT_3D_H36M
  ```

 With global translation, we have the following commands:
  ```bash
 python main_h36_3d.py --input_n ${number of historical sequence frames} --output_n ${maximum number of predicted frames} --test_output_n ${sequence number of the test frame} --skip_rate 1 --mode test --model_path ./checkpoints/CKPT_3D_H36M --global_translation
  ```

### Visualization
 For visualizing from a pretrained model, we have used the following commands:
 ```bash
  python main_h36_3d.py --input_n ${number of historical sequence frames} --output_n ${maximum number of predicted frames} --skip_rate 1 --mode viz --model_path ./checkpoints/CKPT_3D_H36M --n_viz 5
 ```
With global translation, we have the following commands:
 ```bash
  python main_h36_3d.py --input_n ${number of historical sequence frames} --output_n ${maximum number of predicted frames} --skip_rate 1 --mode viz --model_path ./checkpoints/CKPT_3D_H36M --n_viz 5 --global_translation
 ```
 
 ### Acknowledgments
 
 Some of our code here was adapted from [STSGCN](https://github.com/FraLuca/STSGCN).
