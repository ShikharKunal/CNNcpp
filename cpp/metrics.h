#ifndef MYDL_METRICS_H
#define MYDL_METRICS_H

#include "tensor.h"
#include <memory>
#include <vector>

namespace mydl {

// Accuracy: logits (N, C), labels (N). Returns correct / N.
double accuracy(std::shared_ptr<Tensor> logits, const std::vector<size_t>& labels);

// Loss value from scalar loss tensor.
double loss_value(std::shared_ptr<Tensor> loss);

}  // namespace mydl

#endif
