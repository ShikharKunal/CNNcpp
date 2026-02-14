#ifndef MYDL_OPS_H
#define MYDL_OPS_H

#include "tensor.h"
#include <memory>

namespace mydl {

// Element-wise and reduction ops with autograd

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> reshape(std::shared_ptr<Tensor> a, const std::vector<size_t>& new_shape);
std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a);
std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a);

}  // namespace mydl

#endif
