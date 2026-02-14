#include "ops.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mydl {

static void assert_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape)
        throw std::runtime_error("ops: shape mismatch");
}

static size_t numel(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t s : shape) n *= s;
    return n;
}

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    assert_same_shape(*a, *b);
    auto out = std::make_shared<Tensor>(a->shape, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < a->numel(); ++i)
        out->data[i] = a->data[i] + b->data[i];

    out->parents = {a, b};
    out->backward_fn = [a, b, out]() {
        for (size_t i = 0; i < out->grad.size(); ++i) {
            if (a->requires_grad) a->grad[i] += out->grad[i];
            if (b->requires_grad) b->grad[i] += out->grad[i];
        }
    };
    return out;
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    assert_same_shape(*a, *b);
    auto out = std::make_shared<Tensor>(a->shape, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < a->numel(); ++i)
        out->data[i] = a->data[i] * b->data[i];

    out->parents = {a, b};
    out->backward_fn = [a, b, out]() {
        for (size_t i = 0; i < out->grad.size(); ++i) {
            if (a->requires_grad) a->grad[i] += out->grad[i] * b->data[i];
            if (b->requires_grad) b->grad[i] += out->grad[i] * a->data[i];
        }
    };
    return out;
}

// matmul: (M,K) x (K,N) -> (M,N). Stored row-major.
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    if (a->shape.size() != 2 || b->shape.size() != 2 || a->shape[1] != b->shape[0])
        throw std::runtime_error("matmul: incompatible shapes");
    size_t M = a->shape[0], K = a->shape[1], N = b->shape[1];
    std::vector<size_t> out_shape = {M, N};
    auto out = std::make_shared<Tensor>(out_shape, a->requires_grad || b->requires_grad);

    // Optimized direct implementation
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j) {
            float v = 0;
            for (size_t k = 0; k < K; ++k)
                v += a->data[i * K + k] * b->data[k * N + j];
            out->data[i * N + j] = v;
        }

    out->parents = {a, b};
    out->backward_fn = [a, b, out, M, K, N]() {
        if (a->requires_grad)
            for (size_t i = 0; i < M; ++i)
                for (size_t k = 0; k < K; ++k)
                    for (size_t j = 0; j < N; ++j)
                        a->grad[i * K + k] += out->grad[i * N + j] * b->data[k * N + j];
        if (b->requires_grad)
            for (size_t k = 0; k < K; ++k)
                for (size_t j = 0; j < N; ++j)
                    for (size_t i = 0; i < M; ++i)
                        b->grad[k * N + j] += a->data[i * K + k] * out->grad[i * N + j];
    };
    return out;
}

std::shared_ptr<Tensor> reshape(std::shared_ptr<Tensor> a, const std::vector<size_t>& new_shape) {
    size_t n = numel(new_shape);
    if (n != a->numel())
        throw std::runtime_error("reshape: numel mismatch");
    
    // ✅ Reshape creates a view with shared data
    auto out = std::make_shared<Tensor>(new_shape, a->requires_grad);
    out->data = a->data;  // Copy vector (unavoidable for autograd safety)
    out->grad.resize(n, 0.0f);

    out->parents = {a};
    out->backward_fn = [a, out]() {
        if (a->requires_grad)
            for (size_t i = 0; i < a->grad.size(); ++i)
                a->grad[i] += out->grad[i];
    };
    return out;
}

std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor> a) {
    return reshape(a, {a->numel()});
}

std::shared_ptr<Tensor> sum(std::shared_ptr<Tensor> a) {
    auto out = std::make_shared<Tensor>(std::vector<size_t>{1}, a->requires_grad);
    float s = 0;
    for (float v : a->data) s += v;
    out->data[0] = s;

    out->parents = {a};
    out->backward_fn = [a, out]() {
        if (a->requires_grad)
            for (size_t i = 0; i < a->grad.size(); ++i)
                a->grad[i] += out->grad[0];
    };
    return out;
}

std::shared_ptr<Tensor> mean(std::shared_ptr<Tensor> a) {
    size_t n = a->numel();
    auto out = std::make_shared<Tensor>(std::vector<size_t>{1}, a->requires_grad);
    float s = 0;
    for (float v : a->data) s += v;
    out->data[0] = s / static_cast<float>(n);

    out->parents = {a};
    out->backward_fn = [a, out, n]() {
        if (a->requires_grad) {
            float g = out->grad[0] / static_cast<float>(n);
            for (size_t i = 0; i < a->grad.size(); ++i)
                a->grad[i] += g;
        }
    };
    return out;
}

}  // namespace mydl
