

# Automatic Progressive Mixed-Precision Network Quantization

This resposity is the official implementation of our paper. 

### Dependencies
* Python3.6
* PyTorch >== 1.4.0

### Params Settings 

> Please Kindly Refers To `utils/option.py`

### Usage

* > **Train-Float Baseline Models**
  >
  > Please Kindly Refers to `scripts/run.sh`

```shell
## FLOAT BASELINE MODEL
python main/train.py \
        --arch=${ARCH} \
        --num_layers=${LAYERS} \
        --archtype=${ARCHTYPE} \
        --lr=0.001 --weight_decay=1e-4 \
        --train_batch_size=256 --k_bits=8 \
        --pre_k_bits=8 --ratio=1.0 \
        --mission=${MISSION} --gpus=0 \
        --train_epochs=300 --dataset=${DATASET} \
        --data_dir=/gdata/ImageNet2012/ \
        --job_dir=../outputs --clip \
        --resume=${RESUME}
```

* > **Train-Quant Baseline Models**
  >
  > Please Kindly Refers to `scripts/run.sh`

```shell
## Quant BASELINE MODEL
ARCHTYPE = quant
python main/train.py \
        --arch=${ARCH} \
        --num_layers=${LAYERS} \
        --archtype=${ARCHTYPE} \
        --lr=0.001 --weight_decay=1e-4 \
        --train_batch_size=256 --k_bits=8 \
        --pre_k_bits=8 --ratio=1.0 \
        --mission=${MISSION} --gpus=0 \
        --train_epochs=300 --dataset=${DATASET} \
        --data_dir=/gdata/ImageNet2012/ \
        --job_dir=../outputs --clip \
        --resume=${RESUME}
```

* > **Mix-Precision Model Searching**
  >
  > Please Kindly Refers to `scripts/search.sh`

```shell
# Mix-Precision Model Searching
python3 main/search.py \
    --step=${STEP} --lam=${LAMBDA} --interval=${INTERVAL}   \
    --ratio=${RATIO} --search_epochs=${SEARCH_EPOCHS}       \
    --dataset=${DATASET} --arch=${ARCH}  --lr=${LR}         \
    --num_layers=${LAYERS} --k_bits=8 --gpus=0              \
    --data_dir /userhome/memory_data/imagenet --clip        \
    --mission=${MISSION} --resume=${RESUME}
```

* > **Computation Ratio**

```shell
bash measure/compute.sh
```

### Experiment Results

| Quantization | W-bits | A-bits |  Acc.-1   |    Cost    |
| :----------: | :----: | :----: | :-------: | :--------: |
|   Baseline   |   32   |   32   |   70.20   |   1.0000   |
|   ABC-Net    |   5    |   5    |   65.00   |   0.1563   |
|    Dorefa    |   5    |   5    |   68.40   |   0.1563   |
|     PACT     |   5    |   5    |   69.80   |   0.1563   |
|  **ProQHA**  | **MP** | **MP** | **70.01** | **0.1491** |
|   ABC-Net    |   3    |   3    |   61.00   |   0.0938   |
|    Dorefa    |   3    |   3    |   67.50   |   0.0938   |
|     PACT     |   3    |   3    |   68.10   |   0.0938   |
|   LQ-Nets    |   3    |   3    |   68.20   |   0.0938   |
|  **ProQHA**  | **MP** | **MP** | **68.34** | **0.0918** |