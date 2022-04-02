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
The arguments for running the code are defined in [parser.py](utils/parser.py). We have used the following commands for training the network,on different datasets and body pose representations(3D and euler angles):
 
```bash
 python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 
 ```
 
 ### Test
 To test on the pretrained model, we have used the following commands:
 ```bash
 python main_h36_3d.py --input_n 10 --output_n 25 --test_output_n 25 --skip_rate 1 --mode test --model_path ./checkpoints/CKPT_3D_H36M
  ```

### Visualization
 For visualizing from a pretrained model, we have used the following commands:
 ```bash
  python main_h36_3d.py --input_n 10 --output_n 25 --skip_rate 1 --mode viz --model_path ./checkpoints/CKPT_3D_H36M --n_viz 5
 ```

 
 ### Acknowledgments
 
 Some of our code here was adapted from [STSGCN](https://github.com/FraLuca/STSGCN).
