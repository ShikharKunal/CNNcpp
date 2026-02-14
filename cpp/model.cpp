#include "model.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mydl {

SimpleCNN::SimpleCNN(size_t num_classes) : num_classes_(num_classes), flat_size_(0) {
    // Input (N, 3, 32, 32) -> Conv(3, 8, 3, stride=2) -> (N, 8, 15, 15) -> ReLU -> MaxPool(2,2) -> (N, 8, 7, 7)
    // Reduced channels (16→8) and stride=2 for faster training
    conv_ = std::make_shared<Conv2DLayer>(3, 8, 3, 2, 0);  // stride=2, out: 15x15
    relu_ = std::make_shared<ReLULayer>();
    pool_ = std::make_shared<MaxPool2DLayer>(2, 2);        // out: 7x7
    flat_size_ = 8 * 7 * 7;  // 392 (was 3600)
    linear_ = std::make_shared<LinearLayer>(flat_size_, num_classes);
}

std::shared_ptr<Tensor> SimpleCNN::forward(std::shared_ptr<Tensor> x) {
    auto h = conv_->forward(x);
    h = relu_->forward(h);
    h = pool_->forward(h);
    // After pool: (N, 16, 15, 15), need to reshape to (N, 3600) for linear
    size_t batch_size = h->shape[0];
    h = reshape(h, {batch_size, flat_size_});
    return linear_->forward(h);
}

std::vector<std::shared_ptr<Tensor>> SimpleCNN::parameters() const {
    std::vector<std::shared_ptr<Tensor>> p;
    p.push_back(conv_->weight);
    p.push_back(conv_->bias);
    p.push_back(linear_->weight);
    p.push_back(linear_->bias);
    return p;
}

size_t SimpleCNN::count_parameters() const {
    size_t n = 0;
    for (const auto& t : parameters())
        n += t->numel();
    return n;
}

size_t SimpleCNN::count_macs() const {
    // Conv: N * C_out * H_out * W_out * C_in * kH * kW (multiply-add = 1 MAC per output)
    size_t N = 1;  // per sample
    size_t conv_macs = N * 8 * 15 * 15 * 3 * 3 * 3;  // stride=2 gives 15x15 output
    // Linear: N * out_f * in_f
    size_t linear_macs = N * num_classes_ * flat_size_;
    return conv_macs + linear_macs;
}

size_t SimpleCNN::count_flops() const {
    // Approx: MACs * 2 (multiply + add) for conv/linear; ReLU 1 FLOP per element; Pool compare+select
    size_t conv_flops = 2 * (1 * 8 * 15 * 15 * 3 * 3 * 3);
    size_t relu_flops = 1 * 8 * 15 * 15;
    size_t pool_flops = 1 * 8 * 7 * 7 * 2 * 2;  // 2*2 kernel comparisons
    size_t linear_flops = 2 * (1 * num_classes_ * flat_size_);
    return conv_flops + relu_flops + pool_flops + linear_flops;
}

void SimpleCNN::save_weights(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_weights: cannot open " + path);
    auto params = parameters();
    for (const auto& p : params) {
        size_t n = p->numel();
        f.write(reinterpret_cast<const char*>(&n), sizeof(n));
        f.write(reinterpret_cast<const char*>(p->data.data()), n * sizeof(float));
    }
}

void SimpleCNN::load_weights(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_weights: cannot open " + path);
    auto params = parameters();
    for (const auto& p : params) {
        size_t n;
        f.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (n != p->numel()) throw std::runtime_error("load_weights: size mismatch");
        f.read(reinterpret_cast<char*>(p->data.data()), n * sizeof(float));
    }
}

}  // namespace mydl
