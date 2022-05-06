# knee-semi

training data: `<base_dir>/imagesTr/<id>_0000.nii.gz`, `<base_dir>/labelsTr/<id>.nii.gz`

testing data: `<base_dir>/imagesTs/<id>_0000.nii.gz`, `<base_dir>/labelsTs/<id>.nii.gz`

processed `.npy` data: `<base_dir>/npy/<id>_image.npy`, `<base_dir>/npy/<id>_label.npy`

split: see `./dataloaders/preprocess.py:process_split_fully, process_split_semi`

Training:

```shell
mkdir -p ./logs/__nohup

bash py_run.sh code/train_cps.py --exp cps
```

Testing:

```shell
bash py_run.sh code/test.py --exp cps -ep 200 --cps A

python code/evaluate.py -p ./logs/cps/predictions/ep_200/
```