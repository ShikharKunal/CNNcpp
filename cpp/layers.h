#ifndef MYDL_LAYERS_H
#define MYDL_LAYERS_H

#include "tensor.h"
#include "ops.h"
#include <memory>
#include <vector>

namespace mydl {

// Conv2D: input (N, C_in, H, W), kernel (C_out, C_in, kH, kW), output (N, C_out, H', W')
// stride=1, padding=0
std::shared_ptr<Tensor> conv2d_forward(std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias,
    size_t stride, size_t pad);

// ReLU: element-wise max(0, x)
std::shared_ptr<Tensor> relu_forward(std::shared_ptr<Tensor> x);

// MaxPool2D: (N, C, H, W) -> (N, C, H', W'), kernel_size, stride
std::shared_ptr<Tensor> max_pool2d_forward(std::shared_ptr<Tensor> x, size_t kernel_size, size_t stride);

// Linear: y = x @ W^T + b. x (N, in_features), W (out_features, in_features), b (out_features)
std::shared_ptr<Tensor> linear_forward(std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias);

// Layer wrappers with stored parameters (for CNN building)
struct Conv2DLayer {
    size_t in_channels, out_channels, kernel_size, stride, pad;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    Conv2DLayer(size_t in_c, size_t out_c, size_t k, size_t s = 1, size_t p = 0);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

struct ReLULayer {
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

struct MaxPool2DLayer {
    size_t kernel_size, stride;
    MaxPool2DLayer(size_t k = 2, size_t s = 2);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

struct LinearLayer {
    size_t in_features, out_features;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    LinearLayer(size_t in_f, size_t out_f);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
};

}  // namespace mydl

#endif
