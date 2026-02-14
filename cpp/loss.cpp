#include "loss.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mydl {

// Softmax along last dim (per row for 2D (N, C)). Numerically stable.
std::shared_ptr<Tensor> softmax(std::shared_ptr<Tensor> x) {
    if (x->shape.empty()) throw std::runtime_error("softmax: empty shape");
    size_t last_dim = x->shape.back();
    size_t n_rows = x->numel() / last_dim;

    auto out = std::make_shared<Tensor>(x->shape, x->requires_grad);
    
    for (size_t i = 0; i < n_rows; ++i) {
        const float* row_in = x->data.data() + i * last_dim;
        float* row_out = out->data.data() + i * last_dim;
        
        // Find max for numerical stability
        float m = -1e20f;
        for (size_t j = 0; j < last_dim; ++j)
            if (row_in[j] > m) m = row_in[j];
        
        // Compute exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < last_dim; ++j) {
            row_out[j] = expf(row_in[j] - m);
            sum += row_out[j];
        }
        
        // Normalize
        const float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < last_dim; ++j)
            row_out[j] *= inv_sum;
    }

    out->parents = {x};
    
    // ✅ Optimized backward using cached output (no Jacobian recomputation)
    out->backward_fn = [x, out, n_rows, last_dim]() {
        if (!x->requires_grad) return;
        
        for (size_t i = 0; i < n_rows; ++i) {
            const float* s = out->data.data() + i * last_dim;
            const float* g_out = out->grad.data() + i * last_dim;
            float* g_in = x->grad.data() + i * last_dim;
            
            // Compute dot product: sum(g_out * softmax)
            float sum_g = 0.0f;
            for (size_t j = 0; j < last_dim; ++j)
                sum_g += g_out[j] * s[j];
            
            // Apply Jacobian: g_in = softmax * (g_out - sum_g)
            for (size_t j = 0; j < last_dim; ++j)
                g_in[j] += s[j] * (g_out[j] - sum_g);
        }
    };
    return out;
}

// Cross entropy: logits (N, C), labels (N) class indices. Returns scalar. Numerically stable.
std::shared_ptr<Tensor> cross_entropy_loss(std::shared_ptr<Tensor> logits,
    const std::vector<size_t>& labels) {
    if (logits->shape.size() != 2) throw std::runtime_error("cross_entropy: logits must be 2D");
    size_t N = logits->shape[0], C = logits->shape[1];
    if (labels.size() != N) throw std::runtime_error("cross_entropy: labels size mismatch");

    // Pre-compute and cache softmax for backward pass (avoid recomputing exp)
    std::vector<float> softmax_cache(N * C);
    
    // Log-softmax: log(s_j) = x_j - log(sum(exp(x_i))) = x_j - (max + log(sum(exp(x_i - max))))
    auto out = std::make_shared<Tensor>(std::vector<size_t>{1}, true);
    float loss_val = 0.0f;
    for (size_t n = 0; n < N; ++n) {
        const float* row = logits->data.data() + n * C;
        float* softmax_row = &softmax_cache[n * C];
        
        // Find max for numerical stability
        float m = -1e20f;
        for (size_t j = 0; j < C; ++j)
            if (row[j] > m) m = row[j];
        
        // Compute softmax AND cache it for backward
        float sum_exp = 0.0f;
        for (size_t j = 0; j < C; ++j) {
            softmax_row[j] = expf(row[j] - m);
            sum_exp += softmax_row[j];
        }
        
        // Normalize softmax
        const float inv_sum = 1.0f / sum_exp;
        for (size_t j = 0; j < C; ++j)
            softmax_row[j] *= inv_sum;
        
        // Compute loss: -log(p[correct_class])
        size_t y = labels[n];
        if (y >= C) throw std::runtime_error("cross_entropy: label out of range");
        loss_val += -logf(softmax_row[y]);
    }
    out->data[0] = loss_val / static_cast<float>(N);

    out->parents = {logits};
    
    // ✅ Optimized backward: Uses cached softmax, simplified gradient
    out->backward_fn = [logits, labels, out, N, C, softmax_cache]() {
        if (!logits->requires_grad) return;
        const float scale = out->grad[0] / static_cast<float>(N);
        
        for (size_t n = 0; n < N; ++n) {
            float* g = logits->grad.data() + n * C;
            const float* softmax_row = &softmax_cache[n * C];
            const size_t y = labels[n];
            
            // ✅ Simplified gradient: grad = scale * (softmax - one_hot)
            for (size_t j = 0; j < C; ++j) {
                const float one_hot = (j == y) ? 1.0f : 0.0f;
                g[j] += scale * (softmax_row[j] - one_hot);
            }
        }
    };
    return out;
}

}  // namespace mydl
