#include "image_sampler.h"

ImageSampler::ImageSampler() { use_opencl = false; }

ImageSampler::ImageSampler(OpenCLManager *cl_manager) {
  use_opencl = true;
  this->cl_manager = cl_manager;
  cl_int ret = 0;

  std::ifstream sample_rect_kernel_file(
      "src/image_sampler_sample_rect_kernel.cl");
  std::string sample_rect_kernel_contents(
      (std::istreambuf_iterator<char>(sample_rect_kernel_file)),
      std::istreambuf_iterator<char>());

  sample_rect_program = cl::Program(cl_manager->context,
                                    sample_rect_kernel_contents, false, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Creating sample rect program failed:" << ret << std::endl;
    exit(EXIT_FAILURE);
  }
  ret = sample_rect_program.build(std::vector<cl::Device>{cl_manager->device},
                                  NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::string build_log;
    sample_rect_program.getBuildInfo<std::string>(
        cl_manager->device, CL_PROGRAM_BUILD_LOG, &build_log);
    std::cerr << "Building sample rect program failed:" << ret << " "
              << OpenCLManager::GetCLErrorString(ret) << std::endl
              << build_log << std::endl;
    exit(EXIT_FAILURE);
  }
  sample_rect_kernel =
      cl::Kernel(sample_rect_program, "sample_rect_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Creating sample rect kernel failed:" << ret << std::endl;
    std::cerr << cl_manager->GetCLErrorString(ret) << std::endl;
    exit(EXIT_FAILURE);
  }
  create_grid_kernel =
      cl::Kernel(sample_rect_program, "create_grid_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Creating grid kernel failed: " << ret << std::endl;
    exit(EXIT_FAILURE);
  }

  std::ifstream sample_logpolar_kernel_file(
      "src/image_sampler_sample_logpolar_kernel.cl");
  std::string sample_logpolar_kernel_contents(
      (std::istreambuf_iterator<char>(sample_logpolar_kernel_file)),
      std::istreambuf_iterator<char>());
  const size_t sample_logpolar_kernel_length =
      sample_logpolar_kernel_contents.length();
  const char *sample_logpolar_kernel_chars =
      sample_logpolar_kernel_contents.c_str();
  sample_logpolar_program = cl::Program(
      cl_manager->context, sample_logpolar_kernel_contents, false, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Creating sample logpolar program failed:" << ret << std::endl;
  }
  ret = sample_logpolar_program.build(
      std::vector<cl::Device>{cl_manager->device});
  if (ret != CL_SUCCESS) {
    std::string build_log;
    sample_logpolar_program.getBuildInfo<std::string>(
        cl_manager->device, CL_PROGRAM_BUILD_LOG, &build_log);
    std::cerr << "Building sample logpolar program failed:" << ret << std::endl;
    std::cerr << cl_manager->GetCLErrorString(ret) << std::endl;
    std::cerr << build_log << std::endl;
    exit(EXIT_FAILURE);
  }
  sample_logpolar_kernel =
      cl::Kernel(sample_logpolar_program, "sample_logpolar_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Create sample logpolar kernel failed:" << ret << std::endl;
    exit(EXIT_FAILURE);
  }
  create_logpolar_grid_kernel =
      cl::Kernel(sample_logpolar_program, "create_logpolar_grid_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Create logpolar grid kernel failed: " << ret << std::endl;
    exit(EXIT_FAILURE);
  }

  logpolar_gaussian_blur_kernel = cl::Kernel(
      sample_logpolar_program, "logpolar_gaussian_blur_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__
              << " Create logpolar_gaussian_blur_kernel failed:" << ret
              << std::endl;
  }

  std::ifstream interpolate_kernel_file(
      "src/image_sampler_interpolate_kernel.cl");
  std::string interpolate_kernel_contents(
      (std::istreambuf_iterator<char>(interpolate_kernel_file)),
      std::istreambuf_iterator<char>());
  interpolate_program = cl::Program(cl_manager->context,
                                    interpolate_kernel_contents, false, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << " Create interpolate program failed:" << ret
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ret = interpolate_program.build();
  if (ret != CL_SUCCESS) {
    std::string build_log;
    interpolate_program.getBuildInfo<std::string>(
        cl_manager->device, CL_PROGRAM_BUILD_LOG, &build_log);
    std::cerr << __FUNCTION__
              << " Build interpolate logpolar program failed: " << ret
              << std::endl
              << build_log << std::endl;
    std::exit(EXIT_FAILURE);
  }

  interpolate_logpolar_kernel =
      cl::Kernel(interpolate_program, "interpolate_logpolar_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__
              << " Create interpolate logpolar kernel failed:" << ret
              << std::endl;
  }
}

ImageSampler::~ImageSampler() {}

void ImageSampler::InitializeGrid(int target_width, int target_height,
                                  int source_width, int source_height) {
  size_t new_grid_size =
      (target_width + 1) * (target_height + 1) * sizeof(int16_t) * 2;
  if (grid_size == new_grid_size) {
    return;
  }
  std::cout << "Initializing Grid" << std::endl;
  grid_buffer =
      cl::Buffer(cl_manager->context, CL_MEM_READ_WRITE, new_grid_size);
  grid_size = new_grid_size;

  cl_int ret = 0;

  // Set all the parameters and call the kernel
  ret =
      clSetKernelArg(create_grid_kernel(), 0, sizeof(uint16_t *), &grid_buffer);
  ret = clSetKernelArg(create_grid_kernel(), 1, sizeof(int), &target_width);
  ret = clSetKernelArg(create_grid_kernel(), 2, sizeof(int), &target_height);
  ret = clSetKernelArg(create_grid_kernel(), 3, sizeof(int), &source_width);
  ret = clSetKernelArg(create_grid_kernel(), 4, sizeof(int), &source_height);

  cl::NDRange global_item_size(8 * ((target_width + 7) / 8),
                               8 * ((target_height + 7) / 8));
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      create_grid_kernel, 0, global_item_size, local_item_size, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[ImageSampler::InitializeGrid] Launch Kernel Failed; " << ret
              << std::endl;
    return;
  }
}

void ImageSampler::InitializeLogpolarGrid(int target_width, int target_height,
                                          int source_width, int source_height) {
  size_t new_grid_size =
      (target_width + 1) * (target_height + 1) * sizeof(int16_t) * 2;
  if (logpolar_grid_size == new_grid_size) {
    return;
  }
  std::cout << "Initializing Grid" << std::endl;
  logpolar_grid_buffer =
      cl::Buffer(cl_manager->context, CL_MEM_READ_WRITE, new_grid_size);
  logpolar_grid_size = new_grid_size;

  cl_int ret = 0;

  // Set all the parameters and call the kernel
  ret = clSetKernelArg(create_logpolar_grid_kernel(), 0, sizeof(uint16_t *),
                       &logpolar_grid_buffer);
  ret = clSetKernelArg(create_logpolar_grid_kernel(), 1, sizeof(int),
                       &target_width);
  ret = clSetKernelArg(create_logpolar_grid_kernel(), 2, sizeof(int),
                       &target_height);
  ret = clSetKernelArg(create_logpolar_grid_kernel(), 3, sizeof(int),
                       &source_width);
  ret = clSetKernelArg(create_logpolar_grid_kernel(), 4, sizeof(int),
                       &source_height);

  cl::NDRange global_item_size((8 * ((target_width + 7) / 8)),
                               (8 * ((target_height + 7) / 8)));
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      create_logpolar_grid_kernel, 0, global_item_size, local_item_size, NULL,
      NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[ImageSampler::InitializeLogpolarGrid] Launch Kernel Failed; "
              << ret << ", " << OpenCLManager::GetCLErrorString(ret)
              << std::endl;
    return;
  }
  return;

Error:
  std::cerr << "[ImageSampler::InitializeLogpolarGrid] Some error occurred"
            << std::endl;
}

void ImageSampler::SampleFrameRectGPU(cl_mem cl_target_buffer, int target_width,
                                      int target_height, int target_linesize,
                                      cl_mem cl_source_buffer, int source_width,
                                      int source_height, int source_linesize,
                                      float center_x, float center_y) {
  if (!use_opencl) {
    std::cerr
        << "[ImageSampler::SampleFrameRectGPU] Not initialized with OpenCL"
        << std::endl;
    return;
  }

  if (grid_size == -1) {
    // std::cerr << "[ImageSampler::SampleFrameRectGPU] Grid Not Initialized" <<
    // std::endl;
    InitializeGrid(target_width, target_height, source_width, source_height);
  }

  cl_int ret = 0;

  // Set all the parameters and call the kernel
  ret = clSetKernelArg(sample_rect_kernel(), 0, sizeof(uint8_t *),
                       &cl_target_buffer);
  ret = clSetKernelArg(sample_rect_kernel(), 1, sizeof(int), &target_width);
  ret = clSetKernelArg(sample_rect_kernel(), 2, sizeof(int), &target_height);
  ret = clSetKernelArg(sample_rect_kernel(), 3, sizeof(int), &target_linesize);
  ret = clSetKernelArg(sample_rect_kernel(), 4, sizeof(uint8_t *),
                       &cl_source_buffer);
  ret = clSetKernelArg(sample_rect_kernel(), 5, sizeof(int), &source_width);
  ret = clSetKernelArg(sample_rect_kernel(), 6, sizeof(int), &source_height);
  ret = clSetKernelArg(sample_rect_kernel(), 7, sizeof(int), &source_linesize);
  ret = clSetKernelArg(sample_rect_kernel(), 8, sizeof(cl_mem), &grid_buffer);
  ret = clSetKernelArg(sample_rect_kernel(), 9, sizeof(float), &center_x);
  ret = clSetKernelArg(sample_rect_kernel(), 10, sizeof(float), &center_y);

  cl::NDRange global_item_size(target_width, target_height);
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      sample_rect_kernel, 0, global_item_size, local_item_size, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[ImageSampler::SampleFrameRectGPU] Sample rect kernel launch "
                 "failed:"
              << ret << std::endl;
    return;
  }
  return;
Error:
  std::cerr << "[ImageSampler::SampleFrameRectGPU] Some error occurred in "
               "SampleFrameRect"
            << std::endl;
}

// Untested
void ImageSampler::SampleFrameRectCPU(AVFrame *target_frame, uint32_t *buffer,
                                      AVCodecContext *codec_ctx, float center_x,
                                      float center_y) {
  using namespace std;
  cl_int ret = 0;
  int source_width = codec_ctx->width;
  int source_height = codec_ctx->height;
  int input_bytes_per_pixel = 4;
  int input_linesize = 4 * source_width;

  int rect_buffer_width = target_frame->width;
  int rect_buffer_height = target_frame->height;

  uint32_t *source_buffer = buffer;
  uint8_t *output_buffer = target_frame->data[0];

  int output_linesize = target_frame->linesize[0];
  int output_bytes_per_pixel = output_linesize / rect_buffer_width;

  float lambdaX = source_width / (exp(1.0f) - 1);
  float lambdaY = source_height / (exp(1.0f) - 1);

  for (int i = 0; i < rect_buffer_width; i++) {
    for (int j = 0; j < rect_buffer_height; j++) {
      // Assume the height is 0 to 1
      // float aspectRatio = ((float)width / height);

      int u = i - rect_buffer_width / 2;
      int v = j - rect_buffer_height / 2;
      int delta_x =
          max((int)abs(u),
              (int)(lambdaX *
                    (exp(pow(2.0 * abs(u) / rect_buffer_width, 4.0)) - 1))) *
          ((u > 0) - (u < 0));
      int delta_y =
          max((int)abs(v),
              (int)(lambdaY *
                    (exp(pow(2.0 * abs(v) / rect_buffer_height, 4.0)) - 1))) *
          ((v > 0) - (v < 0));
      int x_pos = center_x * source_width + delta_x;
      int y_pos = center_y * source_height + delta_y;

      int target_coord = j * output_linesize + i * output_bytes_per_pixel;
      int bottom_right_coord =
          y_pos * input_linesize + x_pos * input_bytes_per_pixel;
      output_buffer[target_coord] = source_buffer[bottom_right_coord];
      output_buffer[target_coord + 1] = source_buffer[bottom_right_coord + 1];
      output_buffer[target_coord + 2] = source_buffer[bottom_right_coord + 2];
    }
  }
  return;
Error:
  std::cerr << "[ImageSampler::SampleFrameRectCPU] Some error occurred"
            << std::endl;
}

void ImageSampler::ExpandSampledFrameRectCPU(AVFrame *target_frame,
                                             AVFrame *source_frame,
                                             float center_x, float center_y) {
  using namespace std;
  int source_width = source_frame->width;
  int source_height = source_frame->height;
  int source_linesize = source_frame->linesize[0];
  int source_bytes_per_pixel = source_linesize / source_width;
  uint8_t *source_buffer = source_frame->data[0];

  int target_width = target_frame->width;
  int target_height = target_frame->height;
  int target_linesize = target_frame->linesize[0];
  int target_bytes_per_pixel = target_linesize / target_width;
  uint8_t *target_buffer = target_frame->data[0];

  int rect_buffer_width = source_width;
  int rect_buffer_height = source_height;

  float lambdaX = target_width / (exp(1.0f) - 1);
  float lambdaY = target_height / (exp(1.0f) - 1);

  for (int i = 0; i < source_width; i++) {
    for (int j = 0; j < source_height; j++) {
      int u = i - source_width / 2;
      int v = j - source_height / 2;
      int delta_x =
          max((int)abs(u),
              (int)(lambdaX *
                    (exp(pow(2.0 * abs(u) / rect_buffer_width, 4.0)) - 1))) *
          ((u > 0) - (u < 0));
      int delta_x_minus =
          max((int)abs(u - 1),
              (int)(lambdaX *
                    (exp(pow(2.0 * abs(u - 1) / rect_buffer_width, 4.0)) -
                     1))) *
          ((u - 1 > 0) - (u - 1 < 0));
      int delta_y =
          max((int)abs(v),
              (int)(lambdaY *
                    (exp(pow(2.0 * abs(v) / rect_buffer_height, 4.0)) - 1))) *
          ((v > 0) - (v < 0));
      int delta_y_minus =
          max((int)abs(v - 1),
              (int)(lambdaY *
                    (exp(pow(2.0 * (v - 1) / rect_buffer_height, 4.0)) - 1))) *
          ((v - 1 > 0) - (v - 1 < 0));
      int x_pos = center_x * target_width + delta_x;
      int y_pos = center_y * target_height + delta_y;

      if (x_pos >= 0 && x_pos < target_width && y_pos >= 0 &&
          y_pos < target_height) {
        int target_coord =
            y_pos * target_linesize + x_pos * target_bytes_per_pixel;
        int source_coord = j * source_linesize + i * source_bytes_per_pixel;
        target_buffer[target_coord] = source_buffer[source_coord];
        target_buffer[target_coord + 1] = source_buffer[source_coord + 1];
        target_buffer[target_coord + 2] = source_buffer[source_coord + 2];
      }
    }
  }
}

void ImageSampler::InterpolateFrameRectCPU(AVFrame *target_frame,
                                           AVFrame *source_frame,
                                           float center_x, float center_y) {
  using namespace std;
  int source_width = source_frame->width;
  int source_height = source_frame->height;
  int source_linesize = source_frame->linesize[0];
  int source_bytes_per_pixel = source_linesize / source_width;
  uint8_t *source_buffer = source_frame->data[0];

  int target_width = target_frame->width;
  int target_height = target_frame->height;
  int target_linesize = target_frame->linesize[0];
  int target_bytes_per_pixel = target_linesize / target_width;
  uint8_t *target_buffer = target_frame->data[0];

  int rect_buffer_width = source_width;
  int rect_buffer_height = source_height;

  float lambdaX = target_width / (exp(1.0f) - 1);
  float lambdaY = target_height / (exp(1.0f) - 1);

  for (int x_pos = 0; x_pos < target_width; x_pos++) {
    for (int y_pos = 0; y_pos < target_height; y_pos++) {
      // Step 1: Get the UV coordinates corresponding to this XY position
      int center_x_pos = center_x * target_width;
      int center_y_pos = center_y * target_height;
      int delta_x = x_pos - center_x_pos;
      int delta_y = y_pos - center_y_pos;
      int u = ceil(0.5 * rect_buffer_width *
                   pow(log(abs(delta_x) / lambdaX + 1), 0.25)) *
              ((delta_x > 0) - (delta_x < 0));
      int v = ceil(0.5 * rect_buffer_height *
                   pow(log(abs(delta_y) / lambdaY + 1), 0.25)) *
              ((delta_y > 0) - (delta_y < 0));
      int target_coord =
          y_pos * target_linesize + x_pos * target_bytes_per_pixel;

      if (abs(u) > abs(delta_x) || u == 0) {
        u = delta_x;
      }
      if (abs(v) > abs(delta_y) || v == 0) {
        v = delta_y;
      }
      int delta_x_calculated =
          max((int)abs(u),
              (int)(lambdaX *
                    (exp(pow(2.0 * abs(u) / rect_buffer_width, 4.0)) - 1))) *
          ((u > 0) - (u < 0));
      int delta_y_calculated =
          max((int)abs(v),
              (int)(lambdaY *
                    (exp(pow(2.0 * abs(v) / rect_buffer_height, 4.0)) - 1))) *
          ((v > 0) - (v < 0));
      if (delta_x_calculated == delta_x && delta_y_calculated == delta_y) {
        int source_coord = (v + rect_buffer_height / 2) * source_linesize +
                           (u + rect_buffer_width / 2) * source_bytes_per_pixel;

        target_buffer[target_coord] = source_buffer[source_coord];
        target_buffer[target_coord + 1] = source_buffer[source_coord + 1];
        target_buffer[target_coord + 2] = source_buffer[source_coord + 2];
      } else {
        // Bottom Right
        int delta_u = (x_pos < center_x_pos) - (x_pos > center_x_pos);
        int delta_v = (y_pos < center_y_pos) - (y_pos > center_y_pos);
        int delta_x_min =
            max((int)abs(u + delta_u),
                (int)(lambdaX *
                      (exp(pow(2.0 * abs(u + delta_u) / rect_buffer_width,
                               4.0)) -
                       1))) *
            ((u > 0) - (u < 0));
        int delta_y_min =
            max((int)abs(v + delta_v),
                (int)(lambdaY *
                      (exp(pow(2.0 * abs(v + delta_v) / rect_buffer_height,
                               4.0)) -
                       1))) *
            ((v > 0) - (v < 0));

        int min_x =
            min(center_x_pos + delta_x_min, center_x_pos + delta_x_calculated);
        int min_y =
            min(center_y_pos + delta_y_min, center_y_pos + delta_y_calculated);
        int max_x =
            max(center_x_pos + delta_x_min, center_x_pos + delta_x_calculated);
        int max_y =
            max(center_y_pos + delta_y_min, center_y_pos + delta_y_calculated);

        int min_u = min(u, u + delta_u);
        int min_v = min(v, v + delta_v);
        int max_u = max(u, u + delta_u);
        int max_v = max(v, v + delta_v);

        if (min_x < 0) {
          min_u = max_u;
        }
        if (max_x >= target_width) {
          max_u = min_u;
        }
        if (min_y < 0) {
          min_v = max_v;
        }
        if (max_y >= target_height) {
          max_v = min_v;
        }
        int top_left_coord =
            (min_v + rect_buffer_height / 2) * source_linesize +
            (min_u + rect_buffer_width / 2) * source_bytes_per_pixel;
        int top_right_coord =
            (min_v + rect_buffer_height / 2) * source_linesize +
            (max_u + rect_buffer_width / 2) * source_bytes_per_pixel;
        int bottom_left_coord =
            (max_v + rect_buffer_height / 2) * source_linesize +
            (min_u + rect_buffer_width / 2) * source_bytes_per_pixel;
        int bottom_right_coord =
            (max_v + rect_buffer_height / 2) * source_linesize +
            (max_u + rect_buffer_width / 2) * source_bytes_per_pixel;

        float y_ratio = max_y == min_y
                            ? 0
                            : clamp((float)(y_pos - min_y) / (max_y - min_y),
                                    (float)0, (float)1);
        float x_ratio = max_x == min_x
                            ? 0
                            : clamp((float)(x_pos - min_x) / (max_x - min_x),
                                    (float)0, (float)1);
        float left_color_r =
            lerp((float)source_buffer[top_left_coord],
                 (float)source_buffer[bottom_left_coord], y_ratio);
        float left_color_g =
            lerp((float)source_buffer[top_left_coord + 1],
                 (float)source_buffer[bottom_left_coord + 1], y_ratio);
        float left_color_b =
            lerp((float)source_buffer[top_left_coord + 2],
                 (float)source_buffer[bottom_left_coord + 2], y_ratio);
        float right_color_r =
            lerp((float)source_buffer[top_right_coord],
                 (float)source_buffer[bottom_right_coord], y_ratio);
        float right_color_g =
            lerp((float)source_buffer[top_right_coord + 1],
                 (float)source_buffer[bottom_right_coord + 1], y_ratio);
        float right_color_b =
            lerp((float)source_buffer[top_right_coord + 2],
                 (float)source_buffer[bottom_right_coord + 2], y_ratio);
        target_buffer[target_coord] =
            lerp(left_color_r, right_color_r, x_ratio);
        target_buffer[target_coord + 1] =
            lerp(left_color_g, right_color_g, x_ratio);
        target_buffer[target_coord + 2] =
            lerp(left_color_b, right_color_b, x_ratio);
      }
    }
  }
}

void ImageSampler::SampleFrameLogPolarGPU(
    cl_mem cl_target_buffer, int target_width, int target_height,
    int target_linesize, cl_mem cl_source_buffer, int source_width,
    int source_height, int source_linesize, float center_x, float center_y) {
  if (!use_opencl) {
    std::cerr
        << "[ImageSampler::SampleFrameLogPolarGPU] Not initialized with OpenCL"
        << std::endl;
    return;
  }

  if (logpolar_grid_size <= 0) {
    InitializeLogpolarGrid(target_width, target_height, source_width,
                           source_height);
  }

  cl_int ret = 0;

  // Set all the parameters and call the kernel
  ret = sample_logpolar_kernel.setArg(0, sizeof(uint8_t *), &cl_target_buffer);
  ret = sample_logpolar_kernel.setArg(1, sizeof(int), &target_width);
  ret = sample_logpolar_kernel.setArg(2, sizeof(int), &target_height);
  ret = sample_logpolar_kernel.setArg(3, sizeof(int), &target_linesize);
  ret = sample_logpolar_kernel.setArg(4, sizeof(uint8_t *), &cl_source_buffer);
  ret = sample_logpolar_kernel.setArg(5, sizeof(int), &source_width);
  ret = sample_logpolar_kernel.setArg(6, sizeof(int), &source_height);
  ret = sample_logpolar_kernel.setArg(7, sizeof(int), &source_linesize);
  ret = sample_logpolar_kernel.setArg(8, sizeof(cl_mem), &logpolar_grid_buffer);
  ret = sample_logpolar_kernel.setArg(9, sizeof(float), &center_x);
  ret = sample_logpolar_kernel.setArg(10, sizeof(float), &center_y);

  cl::NDRange global_item_size(8 * ((target_width + 7) / 8),
                               8 * ((target_height + 7) / 8));
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      sample_logpolar_kernel, 0, global_item_size, local_item_size, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "Sample logpolar kernel launch failed:" << ret << " "
              << OpenCLManager::GetCLErrorString(ret) << std::endl;
    return;
  }
  return;
Error:
  std::cerr << "Some error occurred in SampleFrameLogPolar" << std::endl;
}

void ImageSampler::ExpandSampledFrameLogPolarCPU(AVFrame *target_frame,
                                                 AVFrame *source_frame,
                                                 float center_x,
                                                 float center_y) {
  using namespace std;
  float alpha = 1.0f;
  int source_width = source_frame->width;
  int source_height = source_frame->height;
  int source_linesize = source_frame->linesize[0];
  int source_bytes_per_pixel = source_linesize / source_width;
  uint8_t *source_buffer = source_frame->data[0];

  int target_width = target_frame->width;
  int target_height = target_frame->height;
  int target_linesize = target_frame->linesize[0];
  int target_bytes_per_pixel = target_linesize / target_width;
  uint8_t *target_buffer = target_frame->data[0];

  int rect_buffer_width = source_width;
  int rect_buffer_height = source_height;

  for (int i = 0; i < source_width; i++) {
    for (int j = 0; j < source_height; j++) {
      int u = i - source_width / 2;
      int v = j - source_height / 2;
      float delta_x = exp(10.0f * pow((float)i / source_width, alpha)) *
                      cos((float)j / source_height * 2 * M_PI);
      float delta_y = exp(10.0f * pow((float)i / source_width, alpha)) *
                      sin((float)j / source_height * 2 * M_PI);
      int x_pos = center_x * target_width + delta_x;
      int y_pos = center_y * target_height + delta_y;

      if (x_pos >= 0 && x_pos < target_width && y_pos >= 0 &&
          y_pos < target_height) {
        int target_coord =
            y_pos * target_linesize + x_pos * target_bytes_per_pixel;
        int source_coord = j * source_linesize + i * source_bytes_per_pixel;
        target_buffer[target_coord] = source_buffer[source_coord];
        target_buffer[target_coord + 1] = source_buffer[source_coord + 1];
        target_buffer[target_coord + 2] = source_buffer[source_coord + 2];
      }
    }
  }
}

void ImageSampler::InterpolateFrameLogPolarCPU(AVFrame *target_frame,
                                               AVFrame *source_frame,
                                               float center_x, float center_y) {
  using namespace std;
  float alpha = 1.0f;

  int source_width = source_frame->width;
  int source_height = source_frame->height;
  int source_linesize = source_frame->linesize[0];
  int source_bytes_per_pixel = source_linesize / source_width;
  uint8_t *source_buffer = source_frame->data[0];

  int target_width = target_frame->width;
  int target_height = target_frame->height;
  int target_linesize = target_frame->linesize[0];
  int target_bytes_per_pixel = target_linesize / target_width;
  uint8_t *target_buffer = target_frame->data[0];

  int rect_buffer_width = source_width;
  int rect_buffer_height = source_height;

  for (int x_pos = 0; x_pos < target_width; x_pos++) {
    for (int y_pos = 0; y_pos < target_height; y_pos++) {
      // Step 1: Get the UV coordinates corresponding to this XY position
      int center_x_pos = center_x * target_width;
      int center_y_pos = center_y * target_height;
      int delta_x = x_pos - center_x_pos;
      int delta_y = y_pos - center_y_pos;
      float i_float =
          delta_x == 0 && delta_y == 0
              ? 0.0f
              : rect_buffer_width *
                    pow(log(sqrt(pow(delta_x, 2.0f) + pow(delta_y, 2.0f))) /
                            10.0f,
                        (1.0f / alpha));
      int i = clamp(round(i_float), 0, rect_buffer_width - 1);
      float j_float = 0.0f;
      if (delta_x != 0) {
        j_float = (atan((float)delta_y / delta_x) + M_PI * (delta_x < 0)) *
                  ((float)rect_buffer_height / (2.0 * M_PI));
        j_float = fmod(j_float + 2 * rect_buffer_height, source_height);
      } else {
        j_float = (M_PI_2 + M_PI * (delta_y < 0)) *
                  (rect_buffer_height / (2.0 * M_PI));
      }
      int j = clamp(round(j_float), 0, rect_buffer_height - 1);

      int target_coord =
          y_pos * target_linesize + x_pos * target_bytes_per_pixel;

      int calculated_x_pos = center_x * target_width +
                             exp(10.0f * pow((float)i / source_width, alpha)) *
                                 cos((float)j / source_height * 2.0f * M_PI);
      int calculated_y_pos = center_y * target_height +
                             exp(10.0f * pow((float)i / source_width, alpha)) *
                                 sin((float)j / source_height * 2.0f * M_PI);

      if (calculated_x_pos == x_pos && calculated_y_pos == y_pos) {
        int source_coord = (j)*source_linesize + (i)*source_bytes_per_pixel;

        target_buffer[target_coord] = source_buffer[source_coord];
        target_buffer[target_coord + 1] = source_buffer[source_coord + 1];
        target_buffer[target_coord + 2] = source_buffer[source_coord + 2];
      } else {
        // Bottom Right

        int min_i = clamp(floor(i_float), 0, source_width - 1);
        int min_j = (int)floor(j_float + source_height) % source_height;
        int max_i = clamp(ceil(i_float), 0, source_width - 1);
        int max_j = (int)ceil(j_float + source_height) % source_height;

        int top_left_coord =
            (min_j)*source_linesize + (min_i)*source_bytes_per_pixel;
        int top_right_coord =
            (min_j)*source_linesize + (max_i)*source_bytes_per_pixel;
        int bottom_left_coord =
            (max_j)*source_linesize + (min_i)*source_bytes_per_pixel;
        int bottom_right_coord =
            (max_j)*source_linesize + (max_i)*source_bytes_per_pixel;

        float i_ratio = i_float - floor(i_float);
        float j_ratio = j_float - floor(j_float);
        float left_color_r =
            lerp((float)source_buffer[top_left_coord],
                 (float)source_buffer[bottom_left_coord], j_ratio);
        float left_color_g =
            lerp((float)source_buffer[top_left_coord + 1],
                 (float)source_buffer[bottom_left_coord + 1], j_ratio);
        float left_color_b =
            lerp((float)source_buffer[top_left_coord + 2],
                 (float)source_buffer[bottom_left_coord + 2], j_ratio);
        float right_color_r =
            lerp((float)source_buffer[top_right_coord],
                 (float)source_buffer[bottom_right_coord], j_ratio);
        float right_color_g =
            lerp((float)source_buffer[top_right_coord + 1],
                 (float)source_buffer[bottom_right_coord + 1], j_ratio);
        float right_color_b =
            lerp((float)source_buffer[top_right_coord + 2],
                 (float)source_buffer[bottom_right_coord + 2], j_ratio);

        target_buffer[target_coord] =
            lerp(left_color_r, right_color_r, i_ratio);
        target_buffer[target_coord + 1] =
            lerp(left_color_g, right_color_g, i_ratio);
        target_buffer[target_coord + 2] =
            lerp(left_color_b, right_color_b, i_ratio);
      }
    }
  }
}

void ImageSampler::InterpolateFrameLogPolarGPU(
    cl_mem cl_target_buffer, int target_width, int target_height,
    int target_linesize, cl_mem cl_source_buffer, int source_width,
    int source_height, int source_linesize, float center_x, float center_y) {
  if (!use_opencl) {
    std::cerr
        << "[SATDecoder::InterpolateFrameRectGPU] Not initialized with OpenCL"
        << std::endl;
    return;
  }

  cl_int ret = 0;

  cl_float2 center = {center_x, center_y};
  // Set all the parameters and call the kernel
  ret = interpolate_logpolar_kernel.setArg(0, sizeof(uint8_t *),
                                           &cl_target_buffer);
  ret = interpolate_logpolar_kernel.setArg(1, sizeof(int), &target_width);
  ret = interpolate_logpolar_kernel.setArg(2, sizeof(int), &target_height);
  ret = interpolate_logpolar_kernel.setArg(3, sizeof(uint8_t *),
                                           &cl_source_buffer);
  ret = interpolate_logpolar_kernel.setArg(4, sizeof(int), &source_width);
  ret = interpolate_logpolar_kernel.setArg(5, sizeof(int), &source_height);
  ret = interpolate_logpolar_kernel.setArg(6, sizeof(cl_float2), &center);

  cl::NDRange global_item_size(target_width, target_height);
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      interpolate_logpolar_kernel, 0, global_item_size, local_item_size, NULL,
      NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[SATDecoder::InterpolateFrameRectGPU] Sample rect kernel "
                 "launch failed:"
              << ret << " " << OpenCLManager::GetCLErrorString(ret)
              << std::endl;
    exit(EXIT_FAILURE);
  }
  return;
}

void ImageSampler::ApplyLogPolarGaussianBlur(cl_mem cl_target_buffer,
                                             int target_width,
                                             int target_height,
                                             int target_linesize,
                                             cl_mem cl_source_buffer) {
  if (!use_opencl) {
    std::cerr
        << "[SATDecoder::InterpolateFrameRectGPU] Not initialized with OpenCL"
        << std::endl;
    return;
  }

  cl_int ret = 0;

  // Set all the parameters and call the kernel
  ret = logpolar_gaussian_blur_kernel.setArg(0, sizeof(uint8_t *),
                                             &cl_target_buffer);
  ret = logpolar_gaussian_blur_kernel.setArg(1, sizeof(int), &target_width);
  ret = logpolar_gaussian_blur_kernel.setArg(2, sizeof(int), &target_height);
  ret = logpolar_gaussian_blur_kernel.setArg(3, sizeof(int), &target_linesize);
  ret = logpolar_gaussian_blur_kernel.setArg(4, sizeof(uint8_t *),
                                             &cl_source_buffer);

  cl::NDRange global_item_size(target_width, target_height);
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      logpolar_gaussian_blur_kernel, 0, global_item_size, local_item_size, NULL,
      NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[SATDecoder::ApplyLogPolarGaussianBlur] "
                 "ApplyLogPolarGaussianBlur kernel "
                 "launch failed:"
              << ret << " " << OpenCLManager::GetCLErrorString(ret)
              << std::endl;
    exit(EXIT_FAILURE);
  }
  return;
}