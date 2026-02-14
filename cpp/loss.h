#ifndef MYDL_LOSS_H
#define MYDL_LOSS_H

#include "tensor.h"
#include <memory>

namespace mydl {

// Softmax along last dimension (per sample). Numerically stable (subtract max).
std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> x);

// Cross entropy: input logits (N, C), labels (N,) as class indices. Returns scalar.
// Numerically stable: log_softmax then NLL.
std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> logits,
    const std::vector<size_t>& labels);

}  // namespace mydl

#endif
