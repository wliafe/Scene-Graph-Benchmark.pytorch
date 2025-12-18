// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


std::tuple<at::Tensor, at::Tensor> ROIPool_forward(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width) {
  // [修改 1] input.type().is_cuda() -> input.is_cuda()
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    return ROIPool_forward_cuda(input, rois, spatial_scale, pooled_height, pooled_width);
#else
    // [修改 2] AT_ERROR -> TORCH_CHECK
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }
  TORCH_CHECK(false, "Not implemented on the CPU");
}

at::Tensor ROIPool_backward(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  // [修改 3] grad.type().is_cuda() -> grad.is_cuda()
  if (grad.is_cuda()) {
#ifdef WITH_CUDA
    return ROIPool_backward_cuda(grad, input, rois, argmax, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width);
#else
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }
  TORCH_CHECK(false, "Not implemented on the CPU");
}
