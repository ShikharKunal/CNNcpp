#ifndef MYDL_TENSOR_H
#define MYDL_TENSOR_H

#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>

namespace mydl {

// Tensor: node in computation graph with dynamic autograd
struct Tensor {
    std::vector<float> data;            // flat storage (using float for performance)
    std::vector<size_t> shape;          // e.g. {N, C, H, W}
    std::vector<float> grad;            // gradients (same size as data)
    bool requires_grad;

    std::vector<std::shared_ptr<Tensor>> parents;
    std::function<void()> backward_fn;

    Tensor();
    Tensor(const std::vector<float>& data, const std::vector<size_t>& shape, bool requires_grad = false);
    Tensor(const std::vector<size_t>& shape, bool requires_grad = false);  // zeros

    size_t numel() const;
    void zero_grad();
    void backward();  // reverse-mode AD

    // Indexing: flat index from multi-index
    size_t index(const std::vector<size_t>& idx) const;
    float at(const std::vector<size_t>& idx) const;
    void set(const std::vector<size_t>& idx, float v);

    std::string repr() const;
};

// Build topological order: post-order DFS so order = [leaves ... root].
// Backward will iterate in reverse (root first).
void build_topo(Tensor* root, std::vector<Tensor*>& order, std::unordered_set<Tensor*>& visited);

}  // namespace mydl

#endif
