#include "tensor.h"
#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <sstream>

namespace mydl {

Tensor::Tensor() : requires_grad(false) {}

Tensor::Tensor(const std::vector<float>& data_, const std::vector<size_t>& shape_, bool requires_grad_)
    : data(data_), shape(shape_), requires_grad(requires_grad_) {
    grad.reserve(data.size());  // Pre-allocate to avoid reallocation
    grad.resize(data.size(), 0.0f);
}

Tensor::Tensor(const std::vector<size_t>& shape_, bool requires_grad_)
    : shape(shape_), requires_grad(requires_grad_) {
    size_t n = 1;
    for (size_t s : shape) n *= s;
    data.reserve(n);  // Pre-allocate capacity
    data.resize(n, 0.0f);
    grad.reserve(n);  // Pre-allocate capacity
    grad.resize(n, 0.0f);
}

size_t Tensor::numel() const {
    size_t n = 1;
    for (size_t s : shape) n *= s;
    return n;
}

size_t Tensor::index(const std::vector<size_t>& idx) const {
    size_t pos = 0;
    size_t stride = 1;
    for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
        pos += idx[static_cast<size_t>(d)] * stride;
        stride *= shape[static_cast<size_t>(d)];
    }
    return pos;
}

float Tensor::at(const std::vector<size_t>& idx) const {
    return data[index(idx)];
}

void Tensor::set(const std::vector<size_t>& idx, float v) {
    data[index(idx)] = v;
}

void Tensor::zero_grad() {
    // ✅ OPTIMIZED: memset is faster than std::fill for zeroing
    if (!grad.empty()) {
        std::memset(grad.data(), 0, grad.size() * sizeof(float));
    }
}

void build_topo(Tensor* root, std::vector<Tensor*>& order, std::unordered_set<Tensor*>& visited) {
    if (!root || visited.count(root)) return;
    visited.insert(root);
    for (const auto& p : root->parents) {
        if (p) build_topo(p.get(), order, visited);
    }
    order.push_back(root);
}

void Tensor::backward() {
    if (grad.empty() || grad.size() != data.size())
        grad.resize(data.size(), 0.0f);
    // Caller must set grad (e.g. loss sets grad to 1.0 for scalar). If scalar and unset, seed 1.0.
    if (numel() == 1 && grad[0] == 0.0f)
        grad[0] = 1.0f;

    std::vector<Tensor*> order;
    std::unordered_set<Tensor*> visited;
    build_topo(this, order, visited);
    // order is [leaves ... root]; run backward_fn from root to leaves
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Tensor* node = *it;
        if (node->backward_fn)
            node->backward_fn();
    }
    
    // ✅ Clear computation graph to free memory immediately
    for (auto* node : order) {
        node->parents.clear();
        node->backward_fn = nullptr;
    }
}

std::string Tensor::repr() const {
    std::ostringstream os;
    os << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) os << ",";
        os << shape[i];
    }
    os << "], numel=" << numel() << ")";
    return os.str();
}

}  // namespace mydl
