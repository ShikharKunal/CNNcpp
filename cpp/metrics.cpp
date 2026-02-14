#include "metrics.h"
#include <stdexcept>

namespace mydl {

double accuracy(std::shared_ptr<Tensor> logits, const std::vector<size_t>& labels) {
    if (!logits || logits->shape.size() != 2) throw std::runtime_error("accuracy: logits must be 2D");
    size_t N = logits->shape[0], C = logits->shape[1];
    if (labels.size() != N) throw std::runtime_error("accuracy: labels size mismatch");

    size_t correct = 0;
    for (size_t i = 0; i < N; ++i) {
        size_t pred = 0;
        float best = logits->data[i * C + 0];
        for (size_t j = 1; j < C; ++j) {
            if (logits->data[i * C + j] > best) {
                best = logits->data[i * C + j];
                pred = j;
            }
        }
        if (pred == labels[i]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(N);
}

double loss_value(std::shared_ptr<Tensor> loss) {
    if (!loss || loss->numel() != 1) throw std::runtime_error("loss_value: scalar tensor expected");
    return loss->data[0];
}

}  // namespace mydl
