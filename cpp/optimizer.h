#ifndef MYDL_OPTIMIZER_H
#define MYDL_OPTIMIZER_H

#include "tensor.h"
#include <memory>
#include <vector>

namespace mydl {

class SGD {
public:
    SGD(const std::vector<std::shared_ptr<Tensor>>& parameters, float learning_rate);
    void step();
    void zero_grad();

private:
    std::vector<std::shared_ptr<Tensor>> parameters_;
    float lr_;  // ✅ Changed from double to float
};

}  // namespace mydl

#endif
