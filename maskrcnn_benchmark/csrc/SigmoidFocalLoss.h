// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor SigmoidFocalLoss_forward(
    const at::Tensor& logits,
    const at::Tensor& targets,
    const int num_classes, 
    const float gamma, 
    const float alpha) {
  // [修改 1] logits.type().is_cuda() -> logits.is_cuda()
  if (logits.is_cuda()) {
#ifdef WITH_CUDA
    return SigmoidFocalLoss_forward_cuda(logits, targets, num_classes, gamma, alpha);
#else
    // [修改 2] AT_ERROR -> TORCH_CHECK
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }
  TORCH_CHECK(false, "Not implemented on the CPU");
}

at::Tensor SigmoidFocalLoss_backward(
           const at::Tensor& logits,
                             const at::Tensor& targets,
           const at::Tensor& d_losses,
           const int num_classes,
           const float gamma,
           const float alpha) {
  // [修改 3] logits.type().is_cuda() -> logits.is_cuda()
  if (logits.is_cuda()) {
#ifdef WITH_CUDA
    return SigmoidFocalLoss_backward_cuda(logits, targets, d_losses, num_classes, gamma, alpha);
#else
    TORCH_CHECK(false, "Not compiled with GPU support");
#endif
  }
  TORCH_CHECK(false, "Not implemented on the CPU");
}
