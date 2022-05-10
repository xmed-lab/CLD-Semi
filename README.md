## CLD-Semi

Yiqun Lin, Huifeng Yao, Zezhong Li, Guoyan Zheng, Xiaomeng Li, "Calibrating Label Distribution for Class-Imbalanced Barely-Supervised Knee Segmentation", MICCAI 2022 (Provisionally Accepted). [[paper](https://arxiv.org/abs/2205.03644)]

### 0. Citation

```
@misc{cld2022lin,
  doi = {10.48550/ARXIV.2205.03644},
  url = {https://arxiv.org/abs/2205.03644},
  author = {Lin, Yiqun and Yao, Huifeng and Li, Zezhong and Zheng, Guoyan and Li, Xiaomeng},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Calibrating Label Distribution for Class-Imbalanced Barely-Supervised Knee Segmentation},
  publisher = {arXiv},
  year = {2022}
}
```

### 1. Environment

This code has been tested with Python 3.6, PyTorch 1.8, torchvision 0.9.0, and CUDA 11.1 on Ubuntu 20.04.

### 2. Data Preparation

The MR imaging scans are available at https://oai.nih.gov/. Run the function `process_npy` in `./code/data/preprocess.py` to convert `.nii.gz` files into `.npy` for faster loading. To generate the labeled/unlabeled splits, run the function `process_split_semi` or use our pre-split files in `./knee_data/*.txt`. After preprocessing, the `./knee_data/` folder should be organized as follows:

```shell
./knee_data/
├── imagesTr
│   ├── <id>_0000.nii.gz
├── labelsTr
│   ├── <id>.nii.gz
├── imagesTs
│   ├── <id>_0000.nii.gz
├── labelsTs
│   ├── <id>.nii.gz
├── npy
│   ├── <id>_image.npy
│   ├── <id>_label.npy
├── splits
│   ├── labeled.txt
│   ├── unlabeled.txt
│   ├── train.txt
│   ├── eval.txt
│   ├── test.txt
```

### 3. Training

Run the following commands for training.

```shell
mkdir -p ./logs/__nohup

bash py_run.sh code/train_cld.py --exp cld -g 0
```

### 4. Testing

Run the following commands for testing.

```shell
bash py_run.sh code/test.py --exp cld -ep 280 --cps A -g 0
python code/evaluate.py -p ./logs/cld/predictions/ep_280/
```

| Model    | Avg. | DF   | FC   | Ti   | TC   | Link                                                         |
| -------- | ---- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| CLD-Semi | 87.2 | 93.8 | 83.7 | 92.8 | 78.6 | [ep_280.pth](https://drive.google.com/file/d/1rMd6v91oNs_GoJ0zeHWiC9xL3IIuIPGE/view?usp=sharing) |

## License

This repository is released under MIT License (see LICENSE file for details).

