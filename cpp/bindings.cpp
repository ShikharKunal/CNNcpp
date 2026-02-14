#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "ops.h"
#include "layers.h"
#include "loss.h"
#include "optimizer.h"
#include "dataloader.h"
#include "metrics.h"
#include "model.h"

namespace py = pybind11;
using namespace mydl;

PYBIND11_MODULE(mydl_cpp, m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<float>&, const std::vector<size_t>&, bool>(),
             py::arg("data"), py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<const std::vector<size_t>&, bool>(),
             py::arg("shape"), py::arg("requires_grad") = false)
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("grad", &Tensor::grad)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("requires_grad", &Tensor::requires_grad)
        .def("numel", &Tensor::numel)
        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", &Tensor::backward)
        .def("repr", &Tensor::repr)
        .def("__repr__", &Tensor::repr);

    m.def("add", &add);
    m.def("mul", &mul);
    m.def("matmul", &matmul);
    m.def("reshape", &reshape, py::arg("a"), py::arg("new_shape"));
    m.def("flatten", &flatten);
    m.def("sum", &sum);
    m.def("mean", &mean);

    py::class_<Conv2DLayer, std::shared_ptr<Conv2DLayer>>(m, "Conv2DLayer")
        .def(py::init<size_t, size_t, size_t, size_t, size_t>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
             py::arg("stride") = 1, py::arg("pad") = 0)
        .def("forward", &Conv2DLayer::forward)
        .def_readonly("weight", &Conv2DLayer::weight)
        .def_readonly("bias", &Conv2DLayer::bias);

    py::class_<ReLULayer, std::shared_ptr<ReLULayer>>(m, "ReLULayer")
        .def(py::init<>())
        .def("forward", &ReLULayer::forward);

    py::class_<MaxPool2DLayer, std::shared_ptr<MaxPool2DLayer>>(m, "MaxPool2DLayer")
        .def(py::init<size_t, size_t>(), py::arg("kernel_size") = 2, py::arg("stride") = 2)
        .def("forward", &MaxPool2DLayer::forward);

    py::class_<LinearLayer, std::shared_ptr<LinearLayer>>(m, "LinearLayer")
        .def(py::init<size_t, size_t>(), py::arg("in_features"), py::arg("out_features"))
        .def("forward", &LinearLayer::forward)
        .def_readonly("weight", &LinearLayer::weight)
        .def_readonly("bias", &LinearLayer::bias);

    m.def("softmax", &softmax);
    m.def("cross_entropy_loss", &cross_entropy_loss, py::arg("logits"), py::arg("labels"));

    py::class_<SGD>(m, "SGD")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>>&, float>(),
             py::arg("parameters"), py::arg("learning_rate"))
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);

    // Dataset loading (Python-side with opencv-python)
    py::class_<DatasetResult>(m, "DatasetResult")
        .def_readonly("images", &DatasetResult::images)
        .def_readonly("labels", &DatasetResult::labels)
        .def_readonly("num_classes", &DatasetResult::num_classes)
        .def_readonly("load_time_seconds", &DatasetResult::load_time_seconds);

    m.def("create_dataset_from_numpy", &create_dataset_from_numpy,
          py::arg("data"), py::arg("shape"), py::arg("labels"), 
          py::arg("num_classes"), py::arg("load_time"),
          "Create dataset from Python-loaded numpy arrays");

    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<std::shared_ptr<Tensor>, const std::vector<size_t>&, size_t, size_t>(),
             py::arg("images"), py::arg("labels"), py::arg("num_classes"), py::arg("batch_size"))
        .def("has_next", &DataLoader::has_next)
        .def("next", &DataLoader::next)
        .def("reset", &DataLoader::reset)
        .def_readonly("num_samples", &DataLoader::num_samples)
        .def_readonly("num_classes", &DataLoader::num_classes)
        .def_readonly("batch_size", &DataLoader::batch_size);

    m.def("accuracy", &accuracy);
    m.def("loss_value", &loss_value);

    py::class_<SimpleCNN, std::shared_ptr<SimpleCNN>>(m, "SimpleCNN")
        .def(py::init<size_t>(), py::arg("num_classes"))
        .def("forward", &SimpleCNN::forward)
        .def("parameters", &SimpleCNN::parameters)
        .def("count_parameters", &SimpleCNN::count_parameters)
        .def("count_macs", &SimpleCNN::count_macs)
        .def("count_flops", &SimpleCNN::count_flops)
        .def("save_weights", &SimpleCNN::save_weights, py::arg("path"))
        .def("load_weights", &SimpleCNN::load_weights, py::arg("path"));
}
