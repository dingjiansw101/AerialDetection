// adapted from "psroi_rotated_align_ave" in RoITransformer "https://github.com/dingjiansw101/RoITransformer_DOTA/blob/master/fpn/operator_cxx/psroi_rotated_align_ave.cu"
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
    int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int max_block_num = 65000;
    return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(
    const scalar_t* bottom_data,
    const int height,
    const int width,
    scalar_t y,
    scalar_t x,
    const int index /* index for debug only*/) {
// deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
        // empty
        return 0;
    }

    if (y <= 0) {
        y = 0;
    }
    if (x <= 0) {
        x = 0;
    }

    int y_low = static_cast<int>(y);
    int x_low = static_cast<int>(x);
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
        y_high = y_low = height - 1;
        y = (scalar_t)y_low;
    } else {
        y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
        x_high = x_low = width - 1;
        x = (scalar_t)x_low;
    } else {
        x_high = x_low + 1;
    }

    scalar_t ly = y - y_low;
    scalar_t lx = x - x_low;
    scalar_t hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    scalar_t v1 = bottom_data[y_low * width + x_low];
    scalar_t v2 = bottom_data[y_low * width + x_high];
    scalar_t v3 = bottom_data[y_high * width + x_low];
    scalar_t v4 = bottom_data[y_high * width + x_high];
    scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    return val;
}
  

template <typename scalar_t>
__global__ void PSROIALIGNRotatedForward(
  const int count,
  const scalar_t* bottom_data,
  const scalar_t* bottom_rois,
  const scalar_t spatial_scale,
  const int sampling_ratio, const int channels,
  const int height, const int width,
  const int pooled_height, const int pooled_width,
  const int output_dim,
  const int group_size,
  scalar_t* top_data) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const scalar_t* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];
  
    // scalar_t roi_start_w = (offset_bottom_rois[1]) * spatial_scale;
    // scalar_t roi_start_h = (offset_bottom_rois[2]) * spatial_scale;
    // scalar_t roi_end_w = (offset_bottom_rois[3]) * spatial_scale;
    // scalar_t roi_end_h = (offset_bottom_rois[4]) * spatial_scale;
    
    // Do not using rounding; this implementation detail is critical
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // T theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];

    // // Force too small ROIs to be 1x1
    // scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)1.);  // avoid 0
    // scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)1.);

    // Force malformed ROIs to be 1x1
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);

    // Compute w and h at bottom
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    int gw = floor(static_cast<scalar_t>(pw)* group_size / pooled_width);
    int gh = floor(static_cast<scalar_t>(ph)* group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    int c = (ctop*group_size + gh)*group_size + gw;

    const scalar_t* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
   
       // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosTheta = cos(theta);
    scalar_t sinTheta = sin(theta);

    const scalar_t sample_count = roi_bin_grid_h * roi_bin_grid_w; // e.g., iy = 0, 1
    scalar_t output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const scalar_t yy = roi_start_h + ph * bin_size_h +
          static_cast<scalar_t>(iy + .5f) * bin_size_h /
              static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        // T x = xx * cosTheta + yy * sinTheta + roi_center_w;
        // T y = yy * cosTheta - xx * sinTheta + roi_center_h;
        scalar_t x = xx * cosTheta - yy * sinTheta + roi_center_w;
        scalar_t y = xx * sinTheta + yy * cosTheta + roi_center_h;

        scalar_t val = bilinear_interpolate<scalar_t>(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= sample_count;

    top_data[index] = output_val;
    // scalar_t out_sum = 0;
    // for (int h = hstart; h < hend; ++h) {
    //   for (int w = wstart; w < wend; ++w) {
    //     int bottom_index = h*width + w;
    //     out_sum += offset_bottom_data[bottom_index];
    //   }
    // }

    // scalar_t bin_area = (hend - hstart)*(wend - wstart);
    // top_data[index] = is_empty? (scalar_t)0. : out_sum/bin_area;
  }
}

int PSROIAlignRotatedForwardLaucher(const at::Tensor features, const at::Tensor rois,
    const float spatial_scale, const int sample_num,
    const int channels, const int height,
    const int width, const int num_rois,
    const int pooled_height, const int pooled_width,   
    const int output_dim, const int group_size,
    at::Tensor output) {
// const int output_size = num_rois * pooled_height * pooled_width * channels;
   const int output_size = num_rois * pooled_height * pooled_width * output_dim;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    features.type(), "PSROIAlignRotatedLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();

        PSROIALIGNRotatedForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>> (
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sample_num, channels, height, width, pooled_height,
                pooled_width, output_dim, group_size, top_data);
    }));
    THCudaCheck(cudaGetLastError());
    return 1;
}

template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    scalar_t y,
    scalar_t x,
    scalar_t* w1,
    scalar_t* w2,
    scalar_t* w3,
    scalar_t* w4,
    int* x_low,
    int* x_high,
    int* y_low,
    int* y_high,
    const int /*index*/ /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    *w1 = *w2 = *w3 = *w4 = 0.;
    *x_low = *x_high = *y_low = *y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  *y_low = static_cast<int>(y);
  *x_low = static_cast<int>(x);

  if (*y_low >= height - 1) {
    *y_high = *y_low = height - 1;
    y = (scalar_t)*y_low;
  } else {
    *y_high = *y_low + 1;
  }

  if (*x_low >= width - 1) {
    *x_high = *x_low = width - 1;
    x = (scalar_t)*x_low;
  } else {
    *x_high = *x_low + 1;
  }

  scalar_t ly = y - *y_low;
  scalar_t lx = x - *x_low;
  scalar_t hy = 1. - ly, hx = 1. - lx;

  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;

  return;
}

template <typename scalar_t>
__global__ void PSROIALIGNRotatedBackward(
  const int count,
  const scalar_t* top_diff,
  const scalar_t* bottom_rois,
//   const int num_rois,
  const scalar_t spatial_scale,
  const int sampling_ratio,
  const int channels,
  const int height, const int width, const int pooled_height, 
  const int pooled_width,
  const int output_dim, const int group_size,
  scalar_t* bottom_diff
  ) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const scalar_t* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];
     // Do not round
    scalar_t roi_center_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_center_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_width = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_height = offset_bottom_rois[4] * spatial_scale;
    // T theta = offset_bottom_rois[5] * M_PI / 180.0;
    scalar_t theta = offset_bottom_rois[5];

    // scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    // scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    // scalar_t roi_end_w = offset_bottom_rois[3] * spatial_scale;
    // scalar_t roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force too small ROIs to be 1x1
    // scalar_t roi_width = max(roi_end_w - roi_start_w, (scalar_t)1.);  // avoid 0
    // scalar_t roi_height = max(roi_end_h - roi_start_h, (scalar_t)1.);
    roi_width = max(roi_width, (scalar_t)1.);
    roi_height = max(roi_height, (scalar_t)1.);
    // Compute w and h at bottom
    scalar_t bin_size_h = static_cast<scalar_t>(roi_height) / static_cast<scalar_t>(pooled_height);
    scalar_t bin_size_w = static_cast<scalar_t>(roi_width) / static_cast<scalar_t>(pooled_width);

    // int hstart = floor(static_cast<scalar_t>(ph)* bin_size_h
    //   + roi_start_h);
    // int wstart = floor(static_cast<scalar_t>(pw)* bin_size_w
    //   + roi_start_w);
    // int hend = ceil(static_cast<scalar_t>(ph + 1) * bin_size_h
    //   + roi_start_h);
    // int wend = ceil(static_cast<scalar_t>(pw + 1) * bin_size_w
    //   + roi_start_w);
    // // Add roi offsets and clip to input boundaries
    // hstart = min(max(hstart, 0), height);
    // hend = min(max(hend, 0), height);
    // wstart = min(max(wstart, 0), width);
    // wend = min(max(wend, 0), width);
    // bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Compute c at bottom
    int gw = floor(static_cast<scalar_t>(pw)* group_size / pooled_width);
    int gh = floor(static_cast<scalar_t>(ph)* group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    int c = (ctop*group_size + gh)*group_size + gw;
    scalar_t* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    // scalar_t bin_area = (hend - hstart)*(wend - wstart);
    // scalar_t diff_val = is_empty ? (scalar_t)0. : top_diff[index] / bin_area;
    // for (int h = hstart; h < hend; ++h) {
    //   for (int w = wstart; w < wend; ++w) {
    //     int bottom_index = h*width + w;
    //     atomicAdd(offset_bottom_diff + bottom_index, diff_val);
    //   }
    // }
    // int top_offset = (n * channels + ctop) * pooled_height * pooled_width;
    // const scalar_t* offset_top_diff = top_diff + top_offset;
    // const scalar_t top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    const scalar_t top_diff_this_bin = top_diff[index];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
   
    // roi_start_h and roi_start_w are computed wrt the center of RoI (x, y).
    // Appropriate translation needs to be applied after.
    scalar_t roi_start_h = -roi_height / 2.0;
    scalar_t roi_start_w = -roi_width / 2.0;
    scalar_t cosTheta = cos(theta);
    scalar_t sinTheta = sin(theta);

    // We do average (integral) pooling inside a bin
    const scalar_t sample_count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const scalar_t yy = roi_start_h + ph * bin_size_h +
          static_cast<scalar_t>(iy + .5f) * bin_size_h /
              static_cast<scalar_t>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const scalar_t xx = roi_start_w + pw * bin_size_w +
            static_cast<scalar_t>(ix + .5f) * bin_size_w /
                static_cast<scalar_t>(roi_bin_grid_w);

        // Rotate by theta around the center and translate
        // T x = xx * cosTheta + yy * sinTheta + roi_center_w;
        // T y = yy * cosTheta - xx * sinTheta + roi_center_h;
        scalar_t x = xx * cosTheta - yy * sinTheta + roi_center_w;
        scalar_t y = xx * sinTheta + yy * cosTheta + roi_center_h;

        scalar_t w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<scalar_t>(
            height,
            width,
            y,
            x,
            &w1,
            &w2,
            &w3,
            &w4,
            &x_low,
            &x_high,
            &y_low,
            &y_high,
            index); // TODO: choose the index

        scalar_t g1 = top_diff_this_bin * w1 / sample_count;
        scalar_t g2 = top_diff_this_bin * w2 / sample_count;
        scalar_t g3 = top_diff_this_bin * w3 / sample_count;
        scalar_t g4 = top_diff_this_bin * w4 / sample_count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(
              offset_bottom_diff + y_low * width + x_low, static_cast<scalar_t>(g1));
          atomicAdd(
              offset_bottom_diff + y_low * width + x_high, static_cast<scalar_t>(g2));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_low, static_cast<scalar_t>(g3));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_high, static_cast<scalar_t>(g4));
        }  // if
      }  // ix
    }  // iy          
  }
}

int PSROIAlignRotatedBackwardLaucher(const at::Tensor top_grad, const at::Tensor rois,
    const float spatial_scale, const int sample_num,
    const int channels, const int height,
    const int width, const int num_rois,
    const int pooled_height, const int pooled_width,
    const int output_dim, const int group_size, 
    at::Tensor bottom_grad) {
        const int output_size = num_rois * pooled_height * pooled_width * output_dim;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            top_grad.type(), "PSROIAlignLaucherBackward", ([&] {
                const scalar_t *top_diff = top_grad.data<scalar_t>();
                const scalar_t *rois_data = rois.data<scalar_t>();
                scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
                if (sizeof(scalar_t) == sizeof(double)) {
                    fprintf(stderr, "double is not supported");
                    exit(-1);
                }
                PSROIALIGNRotatedBackward<scalar_t>
                    <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                        output_size, top_diff, rois_data, spatial_scale, sample_num,
                        channels, height, width, pooled_height, pooled_width,
                        output_dim, group_size,
                        bottom_diff);
            }));
        THCudaCheck(cudaGetLastError());
        return 1;
    }