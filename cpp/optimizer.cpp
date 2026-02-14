#include "optimizer.h"

namespace mydl {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>>& parameters, float learning_rate)
    : parameters_(parameters), lr_(learning_rate) {}

void SGD::zero_grad() {
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (parameters_[i]) parameters_[i]->zero_grad();
    }
}

void SGD::step() {
    for (size_t pidx = 0; pidx < parameters_.size(); ++pidx) {
        auto& p = parameters_[pidx];
        if (!p || !p->requires_grad) continue;
        
        for (size_t i = 0; i < p->data.size(); ++i)
            p->data[i] -= lr_ * p->grad[i];
    }
}

}  // namespace mydl
