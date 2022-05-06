label_base=/home/ylindw/datas/knee_data/labelsTs/

nnUNet_evaluate_folder \
    -ref $label_base \
    -pred $1 \
    -l 1 2 3 4
