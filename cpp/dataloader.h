#ifndef MYDL_DATALOADER_H
#define MYDL_DATALOADER_H

#include "tensor.h"
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace mydl {

// Dataset result structure
// Images are loaded in Python and passed as numpy arrays
struct DatasetResult {
    std::shared_ptr<Tensor> images;   // (N, C, H, W) C=3, H=W=32, values in [0,1]
    std::vector<size_t> labels;       // (N,) class indices
    size_t num_classes;
    double load_time_seconds;
};

// Create dataset from Python-loaded numpy data
DatasetResult create_dataset_from_numpy(const std::vector<float>& data,
                                        const std::vector<size_t>& shape,
                                        const std::vector<size_t>& labels,
                                        size_t num_classes,
                                        double load_time);

// Batch iterator: yields (batch_images, batch_labels) for training.
// batch_images: (B, C, H, W), batch_labels: (B,)
struct DataLoader {
    std::shared_ptr<Tensor> images;
    std::vector<size_t> labels;
    size_t batch_size;
    size_t num_classes;
    size_t num_samples;
    size_t current;
    
    // ✅ Pre-allocated batch tensors to avoid repeated allocation
    std::shared_ptr<Tensor> batch_cache;
    std::vector<size_t> batch_labels_cache;

    DataLoader(std::shared_ptr<Tensor> images_, const std::vector<size_t>& labels_,
               size_t num_classes_, size_t batch_size_);
    bool has_next() const;
    std::pair<std::shared_ptr<Tensor>, std::vector<size_t>> next();
    void reset();
};

}  // namespace mydl

#endif
