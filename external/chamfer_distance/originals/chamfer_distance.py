import torch
import os
script_path = os.path.dirname(os.path.abspath(__file__))

from torch.utils.cpp_extension import load

# Load the original chamfer distance module
cd = load(
    name="cd",
    sources=[
        os.path.join(script_path, "chamfer_distance.cpp"),
        os.path.join(script_path, "chamfer_distance.cu"),
    ],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=[
        "--expt-relaxed-constexpr",
        "-I/usr/local/cuda-12.6/include",
    ],
    extra_ldflags=["-L/usr/local/cuda-12.6/lib64"],
    verbose=True,
)

# Load the modified chamfer distance module
cd_new = load(
    name="cd_new",
    sources=[
        os.path.join(script_path, "chamfer_distance_pow1.cpp"),
        os.path.join(script_path, "chamfer_distance_pow1.cu"),
    ],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=[
        "--expt-relaxed-constexpr",
        "-I/usr/local/cuda-12.6/include",
    ],
    extra_ldflags=["-L/usr/local/cuda-12.6/lib64"],
    verbose=True,
)

density_cuda = load(
    name="density",
    sources=[os.path.join(script_path, "optcd_density_term.cpp"), os.path.join(script_path, "optcd_density_term.cu")],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=[
        "--expt-relaxed-constexpr",
        "-I/usr/local/cuda-12.6/include",
    ],
    extra_ldflags=["-L/usr/local/cuda-12.6/lib64"],
    verbose=True,
)

# Define the original ChamferDistanceFunction
class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2

# Define the modified ChamferDistanceFunction
class ChamferDistanceNewFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)
        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd_new.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            cd_new.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)

        if not graddist1.is_cuda:
            cd_new.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.cuda()
            gradxyz2 = gradxyz2.cuda()
            cd_new.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2

# Define the original ChamferDistance module
class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceFunction.apply(xyz1, xyz2)

# Define the new ChamferDistance module
class ChamferDistanceNew(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        return ChamferDistanceNewFunction.apply(xyz1, xyz2)

def calculate_density(coordinates, point_cloud, radius=0.1):
    """
    PyTorch interface for density calculation.
    """
    B, N, _ = coordinates.size()
    _, M, _ = point_cloud.size()

    density = torch.zeros(B, N, device=coordinates.device, dtype=torch.float32)
    density_cuda.calculate_density(coordinates, point_cloud, density, radius)
    return density
