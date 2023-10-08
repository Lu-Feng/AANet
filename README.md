# AANet
This repository provides the code for ICRA2023 paper "[AANet: Aggregation and Alignment Network with Semi-hard Positive Sample Mining for Hierarchical Place Recognition](https://ieeexplore.ieee.org/abstract/document/10160734)".

The usage of this repo is similar to the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark). You can refer to it to prepare datasets.

![introduction](images/intro.png)
## Training
For MSLS
```
python3 train.py --datasets_folder=/path/to/your/datasets/folder --dataset_name=msls --queries_per_epoch=40000 --trunc_te=8 --freeze_te=1 --negs_num_per_query=2 --candipositive_global=0.3 --candipositive_local=0.3
```
For Pitts30k
```
python3 train.py --datasets_folder=/path/to/your/datasets/folder --dataset_name=pitts30k  --trunc_te=8 --freeze_te=1 --negs_num_per_query=2 --horizontal_flip --random_resized_crop=0.3 --candipositive_global=1 --candipositive_local=2 --resume /path/to/your/trained/msls/model/msls.pth
```

## Evaluation
For MSLS
```
python3 eval.py --datasets_folder=/path/to/your/datasets/folder --dataset_name=msls --trunc_te=8 --freeze_te=1 --resume /path/to/your/trained/msls/model/msls.pth
```
For Pitts30k
```
python3 eval.py --datasets_folder=/path/to/your/datasets/folder --dataset_name=pitts30k --trunc_te=8 --freeze_te=1 --resume /path/to/your/trained/pitts30k/model/pitts30k.pth
```

## Trained models
The [model](https://www.dropbox.com/scl/fi/aff148nlmsogs3wucandh/msls.pth?rlkey=4l78pxxock65f11fujomtw27n&dl=0) for MSLS.

The [model](https://www.dropbox.com/scl/fi/pfetfhhekl1grgh83zhbl/pitts30k.pth?rlkey=nbyij3llw5sy0y2j9cykhp7h0&dl=0) for Pitts30k

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@INPROCEEDINGS{aanet,
  author={Lu, Feng and Zhang, Lijun and Dong, Shuting and Chen, Baifan and Yuan, Chun},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={AANet: Aggregation and Alignment Network with Semi-hard Positive Sample Mining for Hierarchical Place Recognition}, 
  year={2023},
  pages={11771-11778},
  doi={10.1109/ICRA48891.2023.10160734}}
```

## Acknowledgements
The structure of this repo is based on [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark).

