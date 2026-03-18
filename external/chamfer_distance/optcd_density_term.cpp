#include <torch/extension.h>

void CalculateDensityLauncher(
    const at::Tensor& coordinates,
    const at::Tensor& point_cloud,
    at::Tensor& density,
    float radius);

void calculate_density_forward(
    const at::Tensor coordinates,
    const at::Tensor point_cloud,
    at::Tensor density,
    float radius) {
    CalculateDensityLauncher(coordinates, point_cloud, density, radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_density", &calculate_density_forward, "Density calculation forward");
}
