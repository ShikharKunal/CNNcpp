#include "dataloader.h"
#include <cstring>
#include <stdexcept>

namespace mydl {

// Python-loaded dataset wrapper - no OpenCV needed!
// Images are loaded in Python using opencv-python (pip package)
// and passed to C++ as numpy arrays via pybind11

DatasetResult create_dataset_from_numpy(const std::vector<float>& data,
                                        const std::vector<size_t>& shape,
                                        const std::vector<size_t>& labels_,
                                        size_t num_classes_,
                                        double load_time_) {
    auto images = std::make_shared<Tensor>(data, shape, false);
    
    DatasetResult r;
    r.images = images;
    r.labels = labels_;
    r.num_classes = num_classes_;
    r.load_time_seconds = load_time_;
    return r;
}

DataLoader::DataLoader(std::shared_ptr<Tensor> images_, const std::vector<size_t>& labels_,
                       size_t num_classes_, size_t batch_size_)
    : images(images_), labels(labels_), batch_size(batch_size_), num_classes(num_classes_),
      num_samples(labels_.size()), current(0) {
    // ✅ Pre-allocate batch cache to avoid repeated allocations
    std::vector<size_t> batch_shape = images->shape;
    batch_shape[0] = batch_size;
    batch_cache = std::make_shared<Tensor>(batch_shape, false);
    batch_labels_cache.resize(batch_size);
}

bool DataLoader::has_next() const {
    return current < num_samples;
}

std::pair<std::shared_ptr<Tensor>, std::vector<size_t>> DataLoader::next() {
    size_t start = current;
    size_t end = std::min(current + batch_size, num_samples);
    current = end;
    size_t B = end - start;
    
    size_t per_sample = images->numel() / images->shape[0];
    
    // ✅ OPTIMIZED: Reuse pre-allocated tensor, just copy data
    // Avoids 1,875 tensor allocations per epoch!
    if (B != batch_cache->shape[0]) {
        // Last batch might be smaller - recreate tensor
        std::vector<size_t> actual_shape = images->shape;
        actual_shape[0] = B;
        batch_cache = std::make_shared<Tensor>(actual_shape, false);
    }
    
    // Fast memcpy into pre-allocated space
    const float* src = images->data.data() + start * per_sample;
    std::memcpy(batch_cache->data.data(), src, B * per_sample * sizeof(float));
    
    // Copy labels (tiny - only 256 bytes for batch_size=32)
    batch_labels_cache.resize(B);
    std::memcpy(batch_labels_cache.data(), labels.data() + start, B * sizeof(size_t));
    
    return {batch_cache, batch_labels_cache};
}

void DataLoader::reset() {
    current = 0;
}

}  // namespace mydl
