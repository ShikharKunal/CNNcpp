#ifndef MYDL_MODEL_H
#define MYDL_MODEL_H

#include "tensor.h"
#include "layers.h"
#include "ops.h"
#include <memory>
#include <string>
#include <vector>

namespace mydl {

// Simple CNN: Conv2D -> ReLU -> MaxPool -> Flatten -> Linear
// Input: (N, 3, 32, 32). Output: (N, num_classes).
class SimpleCNN {
public:
    SimpleCNN(size_t num_classes);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x);
    std::vector<std::shared_ptr<Tensor>> parameters() const;
    size_t count_parameters() const;
    size_t count_macs() const;   // MACs per forward pass (one batch)
    size_t count_flops() const;  // FLOPs per forward pass
    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

private:
    size_t num_classes_;
    std::shared_ptr<Conv2DLayer> conv_;
    std::shared_ptr<ReLULayer> relu_;
    std::shared_ptr<MaxPool2DLayer> pool_;
    std::shared_ptr<LinearLayer> linear_;
    size_t flat_size_;  // after conv+relu+pool: C * H' * W'
};

}  // namespace mydl

#endif
