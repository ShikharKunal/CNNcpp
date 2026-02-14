#include "layers.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace mydl {

// ---- Conv2D (optimized direct convolution) ----
// x: (N, C_in, H, W), weight: (C_out, C_in, kH, kW), bias: (C_out,)
// out: (N, C_out, H', W'), H' = (H + 2*pad - kH) / stride + 1
std::shared_ptr<Tensor> conv2d_forward(std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias,
    size_t stride, size_t pad) {
    if (x->shape.size() != 4 || weight->shape.size() != 4)
        throw std::runtime_error("conv2d: expected 4D input and weight");
    size_t N = x->shape[0], C_in = x->shape[1], H = x->shape[2], W = x->shape[3];
    size_t C_out = weight->shape[0];
    size_t kH = weight->shape[2], kW = weight->shape[3];
    if (weight->shape[1] != C_in)
        throw std::runtime_error("conv2d: channel mismatch");

    size_t H_out = (H + 2 * pad - kH) / stride + 1;
    size_t W_out = (W + 2 * pad - kW) / stride + 1;
    std::vector<size_t> out_shape = {N, C_out, H_out, W_out};
    auto out = std::make_shared<Tensor>(out_shape, x->requires_grad || weight->requires_grad || (bias && bias->requires_grad));

    auto get_x = [&](size_t n, size_t c, size_t hi, size_t wi) -> float {
        long h = static_cast<long>(hi) - static_cast<long>(pad);
        long w = static_cast<long>(wi) - static_cast<long>(pad);
        if (h < 0 || h >= static_cast<long>(H) || w < 0 || w >= static_cast<long>(W))
            return 0.0f;
        return x->data[n * (C_in * H * W) + c * (H * W) + static_cast<size_t>(h) * W + static_cast<size_t>(w)];
    };

    // Cache-optimized forward pass
    const float* x_ptr = x->data.data();
    const float* w_ptr = weight->data.data();
    const float* b_ptr = bias ? bias->data.data() : nullptr;
    float* out_ptr = out->data.data();

    for (size_t n = 0; n < N; ++n) {
        for (size_t co = 0; co < C_out; ++co) {
            const float b_val = b_ptr ? b_ptr[co] : 0.0f;
            
            for (size_t ho = 0; ho < H_out; ++ho) {
                for (size_t wo = 0; wo < W_out; ++wo) {
                    float sum = b_val;
                    
                    // Pre-compute valid kernel ranges (eliminate branches from inner loops)
                    const long h_start_signed = static_cast<long>(ho * stride) - static_cast<long>(pad);
                    const long w_start_signed = static_cast<long>(wo * stride) - static_cast<long>(pad);
                    
                    // Clamp kernel ranges to avoid out-of-bounds
                    const size_t h_in_start = (h_start_signed < 0) ? 0 : static_cast<size_t>(h_start_signed);
                    const size_t h_in_end = std::min(H, static_cast<size_t>(h_start_signed + kH));
                    const size_t w_in_start = (w_start_signed < 0) ? 0 : static_cast<size_t>(w_start_signed);
                    const size_t w_in_end = std::min(W, static_cast<size_t>(w_start_signed + kW));
                    
                    const size_t kh_start = (h_start_signed < 0) ? static_cast<size_t>(-h_start_signed) : 0;
                    const size_t kw_start = (w_start_signed < 0) ? static_cast<size_t>(-w_start_signed) : 0;
                    
                    for (size_t ci = 0; ci < C_in; ++ci) {
                        const float* x_channel = &x_ptr[n * (C_in * H * W) + ci * (H * W)];
                        const float* w_filter = &w_ptr[co * (C_in * kH * kW) + ci * (kH * kW)];
                        
                        // Inner loops with NO branches - all bounds checked outside
                        size_t kh = kh_start;
                        for (size_t h_in = h_in_start; h_in < h_in_end; ++h_in, ++kh) {
                            const float* x_row = &x_channel[h_in * W];
                            const float* w_row = &w_filter[kh * kW];
                            
                            size_t kw = kw_start;
                            // ✅ Innermost loop: sequential memory access, no branches!
                            for (size_t w_in = w_in_start; w_in < w_in_end; ++w_in, ++kw) {
                                sum += x_row[w_in] * w_row[kw];
                            }
                        }
                    }
                    out_ptr[n * (C_out * H_out * W_out) + co * (H_out * W_out) + ho * W_out + wo] = sum;
                }
            }
        }
    }

    out->parents = {x, weight};
    if (bias) out->parents.push_back(bias);
    
    // ✅ Optimized backward pass with cache locality
    out->backward_fn = [x, weight, bias, out, N, C_in, C_out, H, W, H_out, W_out, kH, kW, stride, pad]() {
        const float* out_grad_ptr = out->grad.data();
        const float* x_data_ptr = x->data.data();
        const float* w_data_ptr = weight->data.data();
        
        // Gradient w.r.t. input
        if (x->requires_grad) {
            float* x_grad_ptr = x->grad.data();
            
            // ✅ OPTIMIZED: Eliminate integer division/modulo from inner loops
            for (size_t n = 0; n < N; ++n) {
                for (size_t ci = 0; ci < C_in; ++ci) {
                    for (size_t hi = 0; hi < H; ++hi) {
                        for (size_t wi = 0; wi < W; ++wi) {
                            float grad_sum = 0.0f;
                            const size_t x_idx = n * (C_in * H * W) + ci * (H * W) + hi * W + wi;
                            
                            // Which output positions use this input pixel?
                            // Instead of division/modulo, compute valid output range directly
                            for (size_t co = 0; co < C_out; ++co) {
                                const float* w_filter = &w_data_ptr[co * (C_in * kH * kW) + ci * (kH * kW)];
                                
                                for (size_t kh = 0; kh < kH; ++kh) {
                                    // ✅ Rewritten to avoid modulo/division
                                    // Original: ho_long = hi + pad - kh, check if divisible by stride
                                    // New: Directly compute ho range that uses this input
                                    const long input_h = static_cast<long>(hi) + static_cast<long>(pad) - static_cast<long>(kh);
                                    
                                    if (input_h < 0) continue;
                                    
                                    // ✅ For stride=2: ho = input_h / 2 (if divisible)
                                    // Check divisibility without modulo for stride=2
                                    if (stride == 2) {
                                        if ((input_h & 1) != 0) continue;  // ✅ Bit test instead of modulo!
                                        const size_t ho = static_cast<size_t>(input_h) >> 1;  // ✅ Bit shift instead of division!
                                        if (ho >= H_out) continue;
                                        
                                        for (size_t kw = 0; kw < kW; ++kw) {
                                            const long input_w = static_cast<long>(wi) + static_cast<long>(pad) - static_cast<long>(kw);
                                            if (input_w < 0) continue;
                                            if ((input_w & 1) != 0) continue;  // ✅ Bit test
                                            const size_t wo = static_cast<size_t>(input_w) >> 1;  // ✅ Bit shift
                                            if (wo >= W_out) continue;
                                            
                                            const size_t out_idx = n * (C_out * H_out * W_out) + co * (H_out * W_out) + ho * W_out + wo;
                                            grad_sum += out_grad_ptr[out_idx] * w_filter[kh * kW + kw];
                                        }
                                    } else {
                                        // Fallback for stride != 2 (currently unused, but safe)
                                        if (input_h % static_cast<long>(stride) != 0) continue;
                                        const size_t ho = static_cast<size_t>(input_h) / stride;
                                        if (ho >= H_out) continue;
                                        
                                        for (size_t kw = 0; kw < kW; ++kw) {
                                            const long input_w = static_cast<long>(wi) + static_cast<long>(pad) - static_cast<long>(kw);
                                            if (input_w < 0 || input_w % static_cast<long>(stride) != 0) continue;
                                            const size_t wo = static_cast<size_t>(input_w) / stride;
                                            if (wo >= W_out) continue;
                                            
                                            const size_t out_idx = n * (C_out * H_out * W_out) + co * (H_out * W_out) + ho * W_out + wo;
                                            grad_sum += out_grad_ptr[out_idx] * w_filter[kh * kW + kw];
                                        }
                                    }
                                }
                            }
                            x_grad_ptr[x_idx] += grad_sum;
                        }
                    }
                }
            }
        }
        
        // Gradient w.r.t. weight
        if (weight->requires_grad) {
            float* w_grad_ptr = weight->grad.data();
            
            for (size_t co = 0; co < C_out; ++co) {
                for (size_t ci = 0; ci < C_in; ++ci) {
                    for (size_t kh = 0; kh < kH; ++kh) {
                        for (size_t kw = 0; kw < kW; ++kw) {
                            float g = 0.0f;
                            const size_t w_idx = co * (C_in * kH * kW) + ci * (kH * kW) + kh * kW + kw;
                            
                            for (size_t n = 0; n < N; ++n) {
                                const float* x_channel = &x_data_ptr[n * (C_in * H * W) + ci * (H * W)];
                                const float* out_grad_channel = &out_grad_ptr[n * (C_out * H_out * W_out) + co * (H_out * W_out)];
                                
                                for (size_t ho = 0; ho < H_out; ++ho) {
                                    const long h_pos = static_cast<long>(ho * stride + kh) - static_cast<long>(pad);
                                    if (h_pos < 0 || h_pos >= static_cast<long>(H)) continue;
                                    
                                    for (size_t wo = 0; wo < W_out; ++wo) {
                                        const long w_pos = static_cast<long>(wo * stride + kw) - static_cast<long>(pad);
                                        if (w_pos >= 0 && w_pos < static_cast<long>(W)) {
                                            g += out_grad_channel[ho * W_out + wo] * 
                                                 x_channel[static_cast<size_t>(h_pos) * W + static_cast<size_t>(w_pos)];
                                        }
                                    }
                                }
                            }
                            w_grad_ptr[w_idx] += g;
                        }
                    }
                }
            }
        }
        
        // Gradient w.r.t. bias
        if (bias && bias->requires_grad) {
            for (size_t co = 0; co < C_out; ++co) {
                float g = 0.0f;
                for (size_t n = 0; n < N; ++n) {
                    const float* grad_channel = &out_grad_ptr[n * (C_out * H_out * W_out) + co * (H_out * W_out)];
                    for (size_t i = 0; i < H_out * W_out; ++i)
                        g += grad_channel[i];
                }
                bias->grad[co] += g;
            }
        }
    };
    return out;
}

// ---- ReLU ----
std::shared_ptr<Tensor> relu_forward(std::shared_ptr<Tensor> x) {
    auto out = std::make_shared<Tensor>(x->shape, x->requires_grad);
    
    // ✅ Slightly optimized: avoid std::max overhead
    const size_t n = x->numel();
    for (size_t i = 0; i < n; ++i) {
        const float val = x->data[i];
        out->data[i] = (val > 0.0f) ? val : 0.0f;
    }

    out->parents = {x};
    out->backward_fn = [x, out]() {
        if (x->requires_grad) {
            for (size_t i = 0; i < x->grad.size(); ++i)
                if (x->data[i] > 0.0f) x->grad[i] += out->grad[i];
        }
    };
    return out;
}

// ---- MaxPool2D (optimized) ----
std::shared_ptr<Tensor> max_pool2d_forward(std::shared_ptr<Tensor> x, size_t kernel_size, size_t stride) {
    if (x->shape.size() != 4) throw std::runtime_error("max_pool2d: expected 4D input");
    size_t N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
    size_t H_out = (H - kernel_size) / stride + 1;
    size_t W_out = (W - kernel_size) / stride + 1;
    std::vector<size_t> out_shape = {N, C, H_out, W_out};
    auto out = std::make_shared<Tensor>(out_shape, x->requires_grad);
    std::vector<size_t> max_idx(out->numel(), 0);

    const float* x_data = x->data.data();
    float* out_data = out->data.data();
    
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            const size_t channel_offset = n * (C * H * W) + c * (H * W);
            const size_t out_channel_offset = n * (C * H_out * W_out) + c * (H_out * W_out);
            
            for (size_t ho = 0; ho < H_out; ++ho) {
                for (size_t wo = 0; wo < W_out; ++wo) {
                    float best = -1e30f;
                    size_t best_idx = 0;
                    const size_t h_start = ho * stride;
                    const size_t w_start = wo * stride;
                    
                    for (size_t kh = 0; kh < kernel_size; ++kh) {
                        const size_t row_offset = channel_offset + (h_start + kh) * W;
                        for (size_t kw = 0; kw < kernel_size; ++kw) {
                            const size_t idx = row_offset + w_start + kw;
                            if (x_data[idx] > best) {
                                best = x_data[idx];
                                best_idx = idx;
                            }
                        }
                    }
                    const size_t out_idx = out_channel_offset + ho * W_out + wo;
                    out_data[out_idx] = best;
                    max_idx[out_idx] = best_idx;
                }
            }
        }
    }

    out->parents = {x};
    out->backward_fn = [x, out, max_idx]() {
        if (x->requires_grad) {
            for (size_t i = 0; i < out->grad.size(); ++i) {
                x->grad[max_idx[i]] += out->grad[i];
            }
        }
    };
    return out;
}

// ---- Linear: y = x @ W^T + b ----  x (N, in_f), W (out_f, in_f), b (out_f)
std::shared_ptr<Tensor> linear_forward(std::shared_ptr<Tensor> x,
    std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias) {
    if (x->shape.size() != 2 || weight->shape.size() != 2)
        throw std::runtime_error("linear: expected 2D x and weight");
    size_t N = x->shape[0], in_f = x->shape[1], out_f = weight->shape[0];
    if (weight->shape[1] != in_f) throw std::runtime_error("linear: feature mismatch");

    std::vector<size_t> out_shape = {N, out_f};
    auto out = std::make_shared<Tensor>(out_shape, x->requires_grad || weight->requires_grad || (bias && bias->requires_grad));

    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < out_f; ++j) {
            float v = bias ? bias->data[j] : 0.0;
            for (size_t k = 0; k < in_f; ++k)
                v += x->data[i * in_f + k] * weight->data[j * in_f + k];
            out->data[i * out_f + j] = v;
        }

    out->parents = {x, weight};
    if (bias) out->parents.push_back(bias);
    out->backward_fn = [x, weight, bias, out, N, in_f, out_f]() {
        if (x->requires_grad)
            for (size_t i = 0; i < N; ++i)
                for (size_t k = 0; k < in_f; ++k)
                    for (size_t j = 0; j < out_f; ++j)
                        x->grad[i * in_f + k] += out->grad[i * out_f + j] * weight->data[j * in_f + k];
        if (weight->requires_grad)
            for (size_t j = 0; j < out_f; ++j)
                for (size_t k = 0; k < in_f; ++k)
                    for (size_t i = 0; i < N; ++i)
                        weight->grad[j * in_f + k] += out->grad[i * out_f + j] * x->data[i * in_f + k];
        if (bias && bias->requires_grad)
            for (size_t j = 0; j < out_f; ++j)
                for (size_t i = 0; i < N; ++i)
                    bias->grad[j] += out->grad[i * out_f + j];
    };
    return out;
}

// ---- Layer wrappers ----
static float kaiming_uniform(size_t fan_in, size_t fan_out) {
    float std = sqrtf(2.0f / static_cast<float>(fan_in));
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-std * sqrtf(3.0f), std * sqrtf(3.0f));
    return dist(rng);
}

Conv2DLayer::Conv2DLayer(size_t in_c, size_t out_c, size_t k, size_t s, size_t p)
    : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s), pad(p) {
    size_t n = out_c * in_c * k * k;
    std::vector<float> w(n);
    float scale = sqrtf(2.0f / (in_c * k * k));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-scale * sqrtf(3.0f), scale * sqrtf(3.0f));
    for (size_t i = 0; i < n; ++i) w[i] = dist(rng);
    std::vector<size_t> w_shape = {out_c, in_c, k, k};
    weight = std::make_shared<Tensor>(w, w_shape, true);
    std::vector<float> b(out_c, 0.0f);
    std::vector<size_t> b_shape = {out_c};
    bias = std::make_shared<Tensor>(b, b_shape, true);
}

std::shared_ptr<Tensor> Conv2DLayer::forward(std::shared_ptr<Tensor> x) {
    return conv2d_forward(x, weight, bias, stride, pad);
}

std::shared_ptr<Tensor> ReLULayer::forward(std::shared_ptr<Tensor> x) {
    return relu_forward(x);
}

MaxPool2DLayer::MaxPool2DLayer(size_t k, size_t s) : kernel_size(k), stride(s) {}

std::shared_ptr<Tensor> MaxPool2DLayer::forward(std::shared_ptr<Tensor> x) {
    return max_pool2d_forward(x, kernel_size, stride);
}

LinearLayer::LinearLayer(size_t in_f, size_t out_f) : in_features(in_f), out_features(out_f) {
    size_t n = in_f * out_f;
    std::vector<float> w(n);
    float scale = sqrtf(2.0f / in_f);
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dist(-scale * sqrtf(3.0f), scale * sqrtf(3.0f));
    for (size_t i = 0; i < n; ++i) w[i] = dist(rng);
    std::vector<size_t> w_shape = {out_f, in_f};
    weight = std::make_shared<Tensor>(w, w_shape, true);
    std::vector<float> b(out_f, 0.0f);
    std::vector<size_t> b_shape = {out_f};
    bias = std::make_shared<Tensor>(b, b_shape, true);
}

std::shared_ptr<Tensor> LinearLayer::forward(std::shared_ptr<Tensor> x) {
    return linear_forward(x, weight, bias);
}

}  // namespace mydl
