# Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

## 安装命令

uv 安装

```bash
安装本地cuda12.8

uv sync

uv pip install -e . --no-build-isolation
```

conda安装

```bash
conda env create -f environment.yml

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 这条命令使用了上海交通大学镜像速度快。
# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu128

pip install -e . --no-build-isolation
```

运行

```bash
CUDA_VISIBLE_DEVICES=1 python tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /mnt/DATA/wanglingfeng/glove MODEL.PRETRAINED_DETECTOR_CKPT /mnt/DATA/wanglingfeng/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /mnt/DATA/wanglingfeng/checkpoints/motif-precls-exmp
```
