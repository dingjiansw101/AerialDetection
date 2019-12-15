#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor poly_nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);

at::Tensor poly_nms(const at::Tensor& dets, const float threshold) {
    CHECK_CUDA(dets);
    if (dets.numel() == 0)
        return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
    return poly_nms_cuda(dets, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("poly_nms", &poly_nms, "polygon non-maximum suppression");
}