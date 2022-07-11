# NDPX Performance Estimator
To cite NdpxEstimator, cite
``` bibtex
@ARTICLE{9609620,  author={Ham, Hyungkyu and Cho, Hyunuk and Kim, Minjae and Park, Jueon and Hong, Jeongmin and Sung, Hyojin and Park, Eunhyeok and Lim, Euicheol and Kim, Gwangsun},  journal={IEEE Computer Architecture Letters},   title={Near-Data Processing in Memory Expander for DNN Acceleration on GPUs},   year={2021},  volume={20},  number={2},  pages={171-174},  doi={10.1109/LCA.2021.3126450}}
```

This project is generated to estimate the NDPX's performance.

The NdpxEstimator estimates the performance of NdpxKernel based on the:
 - shape size
 - #inputs
 - #outputs
 - and, #ops.

## Datasets
I manually collected all the data from the previous experiments.

## Model
I used a simple Transformer model to estimate
