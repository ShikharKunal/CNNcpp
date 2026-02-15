#include "model.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mydl {

SimpleCNN::SimpleCNN(size_t num_classes) : num_classes_(num_classes), flat_size_(0) {
    // Input (N, 3, 32, 32)
    // -> Conv(3, 8, 3, s=2, p=0) -> (N, 8, 15, 15) -> ReLU
    // -> Conv(8, 16, 3, s=1, p=1) -> (N, 16, 15, 15) -> ReLU
    // -> MaxPool(2,2) -> (N, 16, 7, 7)
    // -> Conv(16, 32, 3, s=1, p=1) -> (N, 32, 7, 7) -> ReLU
    // -> MaxPool(2,2) -> (N, 32, 3, 3)
    // -> Flatten (288) -> Linear(288, 128) -> ReLU -> Linear(128, num_classes)
    conv_ = std::make_shared<Conv2DLayer>(3, 8, 3, 2, 0);  // stride=2, out: 15x15
    relu_ = std::make_shared<ReLULayer>();
    conv2_ = std::make_shared<Conv2DLayer>(8, 16, 3, 1, 1);  // out: 15x15
    relu2_ = std::make_shared<ReLULayer>();
    pool_ = std::make_shared<MaxPool2DLayer>(2, 2);           // out: 7x7
    conv3_ = std::make_shared<Conv2DLayer>(16, 32, 3, 1, 1);  // out: 7x7
    relu3_ = std::make_shared<ReLULayer>();
    pool2_ = std::make_shared<MaxPool2DLayer>(2, 2);          // out: 3x3
    flat_size_ = 32 * 3 * 3;                                  // 288
    linear1_ = std::make_shared<LinearLayer>(flat_size_, 128);
    relu_fc_ = std::make_shared<ReLULayer>();
    linear2_ = std::make_shared<LinearLayer>(128, num_classes);
}

std::shared_ptr<Tensor> SimpleCNN::forward(std::shared_ptr<Tensor> x) {
    auto h = conv_->forward(x);
    h = relu_->forward(h);
    h = conv2_->forward(h);
    h = relu2_->forward(h);
    h = pool_->forward(h);
    h = conv3_->forward(h);
    h = relu3_->forward(h);
    h = pool2_->forward(h);
    // After second pool: (N, 32, 3, 3), reshape to (N, 288)
    size_t batch_size = h->shape[0];
    h = reshape(h, {batch_size, flat_size_});
    h = linear1_->forward(h);
    h = relu_fc_->forward(h);
    return linear2_->forward(h);
}

std::vector<std::shared_ptr<Tensor>> SimpleCNN::parameters() const {
    std::vector<std::shared_ptr<Tensor>> p;
    p.push_back(conv_->weight);
    p.push_back(conv_->bias);
    p.push_back(conv2_->weight);
    p.push_back(conv2_->bias);
    p.push_back(conv3_->weight);
    p.push_back(conv3_->bias);
    p.push_back(linear1_->weight);
    p.push_back(linear1_->bias);
    p.push_back(linear2_->weight);
    p.push_back(linear2_->bias);
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
    size_t conv1_macs = N * 8 * 15 * 15 * 3 * 3 * 3;
    size_t conv2_macs = N * 16 * 15 * 15 * 8 * 3 * 3;
    size_t conv3_macs = N * 32 * 7 * 7 * 16 * 3 * 3;
    // Linear: N * out_f * in_f
    size_t fc1_macs = N * 128 * flat_size_;
    size_t fc2_macs = N * num_classes_ * 128;
    return conv1_macs + conv2_macs + conv3_macs + fc1_macs + fc2_macs;
}

size_t SimpleCNN::count_flops() const {
    // Approx: MACs * 2 (multiply + add) for conv/linear; ReLU 1 FLOP per element; Pool compare+select
    size_t conv_flops =
        2 * (1 * 8 * 15 * 15 * 3 * 3 * 3) +
        2 * (1 * 16 * 15 * 15 * 8 * 3 * 3) +
        2 * (1 * 32 * 7 * 7 * 16 * 3 * 3);
    size_t relu_flops =
        1 * 8 * 15 * 15 +
        1 * 16 * 15 * 15 +
        1 * 32 * 7 * 7 +
        1 * 128;
    size_t pool_flops =
        1 * 16 * 7 * 7 * 2 * 2 +
        1 * 32 * 3 * 3 * 2 * 2;  // 2*2 kernel comparisons
    size_t linear_flops =
        2 * (1 * 128 * flat_size_) +
        2 * (1 * num_classes_ * 128);
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
