# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from maskrcnn_benchmark import _C

# [修改] 移除 apex
# from apex import amp

# Only valid with fp32 inputs - give AMP the hint
# nms = amp.float_function(_C.nms)

def nms(dets, scores, threshold):
    # 显式转为 float，替代 amp.float_function 的功能
    return _C.nms(dets.float(), scores.float(), threshold)