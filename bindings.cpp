#include <algorithm>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "matrix_ops.h"

namespace py = pybind11;

namespace {
py::array_t<double> as_matrix(const py::handle &value)
{
    auto array = py::array_t<double,
        py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array || array.ndim() != 2) {
        throw std::invalid_argument("Expected a rectangular numeric 2-D matrix.");
    }
    return array;
}

py::array_t<double> matmul(const py::object &left, const py::object &right)
{
    const auto a = as_matrix(left);
    const auto b = as_matrix(right);
    const auto rows_a = static_cast<std::size_t>(a.shape(0));
    const auto cols_a = static_cast<std::size_t>(a.shape(1));
    const auto rows_b = static_cast<std::size_t>(b.shape(0));
    const auto cols_b = static_cast<std::size_t>(b.shape(1));

    if (cols_a != rows_b) {
        throw std::invalid_argument("Matrix dimensions are incompatible for multiplication.");
    }
    double *raw = ast_matmul(a.data(), b.data(), rows_a, cols_a, cols_b);
    if (raw == nullptr) {
        throw std::bad_alloc();
    }

    py::array_t<double> output({static_cast<py::ssize_t>(rows_a),
                                static_cast<py::ssize_t>(cols_b)});
    std::copy(raw, raw + (rows_a * cols_b), output.mutable_data());
    ast_free(raw);
    return output;
}
} // namespace

PYBIND11_MODULE(backend, module)
{
    module.doc() = "AstraTensor native row-major matrix operations";
    module.def("matmul", &matmul, py::arg("a"), py::arg("b"),
               "Multiply two 2-D numeric matrices using the native C kernel.");
}
