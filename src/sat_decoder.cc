
#include "sat_decoder.h"

SATDecoder::SATDecoder() { use_opencl = false; }

SATDecoder::SATDecoder(OpenCLManager *cl_manager) {
  use_opencl = true;
  this->cl_manager = cl_manager;
  cl_int ret = 0;

  std::ifstream decode_kernel_file("src/sat_decoder_decode_kernel.cl");
  std::string decode_kernel_contents(
      (std::istreambuf_iterator<char>(decode_kernel_file)),
      std::istreambuf_iterator<char>());
  decode_program =
      cl::Program(cl_manager->context, decode_kernel_contents, false, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << " Create decode program failed:" << ret
              << std::endl;
  }
  ret = decode_program.build();
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << " Build decode program failed:" << ret
              << std::endl;
    // PrintClProgramBuildFailure(ret, decode_program(), cl_manager->device());
  }

  decode_kernel = cl::Kernel(decode_program, "decode_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << " Create decode kernel failed:" << ret
              << std::endl;
  }

  std::ifstream sample_rect_kernel_file(
      "src/sat_decoder_sample_rect_kernel.cl");
  std::string sample_rect_kernel_contents(
      (std::istreambuf_iterator<char>(sample_rect_kernel_file)),
      std::istreambuf_iterator<char>());
  sample_rect_program = cl::Program(cl_manager->context,
                                    sample_rect_kernel_contents, false, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << " Create sample rect program failed:" << ret
              << std::endl;
    exit(EXIT_FAILURE);
  }
  ret = sample_rect_program.build();
  if (ret != CL_SUCCESS) {
    std::string build_log;
    sample_rect_program.getBuildInfo(cl_manager->device, CL_PROGRAM_BUILD_LOG,
                                     &build_log);
    std::cerr << __FUNCTION__ << " Build sample rect program failed:" << ret
              << std::endl
              << build_log << std::endl;
    exit(EXIT_FAILURE);
  }
  sample_rect_kernel =
      cl::Kernel(sample_rect_program, "sample_rect_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << " Create sample rect kernel failed:" << ret
              << std::endl;
    exit(EXIT_FAILURE);
  }
  sample_rect_360_kernel =
      cl::Kernel(sample_rect_program, "sample_rect_360_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << "Create sample rect 360 kernel failed:" << ret
              << std::endl;
    exit(EXIT_FAILURE);
  }
  create_grid_kernel =
      cl::Kernel(sample_rect_program, "create_grid_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__
              << "[SATDecoder::SatDecoder] Create grid kernel failed"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  create_reduced_sat_kernel =
      cl::Kernel(sample_rect_program, "create_reduced_sat_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr
        << "[SATDecoder::SatDecoder] Create create_reduced_sat_kernel failed"
        << std::endl;
    exit(EXIT_FAILURE);
  }
  sample_rect_from_reduced_sat_kernel = cl::Kernel(
      sample_rect_program, "sample_rect_from_reduced_sat_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "[SATDecoder::SatDecoder] Create "
                 "sample_rect_from_reduced_sat_kernel failed"
              << std::endl;
  }

  std::ifstream interpolate_kernel_file(
      "src/sat_decoder_interpolate_kernel.cl");
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
    std::cerr << __FUNCTION__ << " Build interpolate program failed: " << ret
              << std::endl
              << build_log << std::endl;
    std::exit(EXIT_FAILURE);
  }

  interpolate_kernel =
      cl::Kernel(interpolate_program, "interpolate_rect_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << " Create interpolate kernel failed:" << ret
              << std::endl;
  }

  grid_buffer = NULL;
  grid_size = -1;
}

SATDecoder::~SATDecoder() { FreeClResources(); }

void SATDecoder::FreeClResources() {
  if (use_opencl) {
    if (grid_size > 0) {
      grid_buffer = cl::Buffer();
      grid_size = -1;
    }
  }
}

void SATDecoder::InitializeGrid(int target_width, int target_height,
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
  ret = create_grid_kernel.setArg(0, sizeof(uint16_t *), &grid_buffer);
  ret = create_grid_kernel.setArg(1, sizeof(int), &target_width);
  ret = create_grid_kernel.setArg(2, sizeof(int), &target_height);
  ret = create_grid_kernel.setArg(3, sizeof(int), &source_width);
  ret = create_grid_kernel.setArg(4, sizeof(int), &source_height);

  cl::NDRange global_item_size(8 * ((target_width + 1 + 7) / 8),
                               8 * ((target_height + 1 + 7) / 8));
  cl::NDRange local_item_size(8, 8);
  cl_manager->command_queue.enqueueNDRangeKernel(
      create_grid_kernel, 0, global_item_size, local_item_size, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[SATDecoder::InitializeGrid] Launch Kernel Failed; " << ret
              << std::endl;
    return;
  }
  return;

Error:
  std::cerr << "[SATDecoder::InitializeGrid] Some error occurred" << std::endl;
}

void SATDecoder::DecodeFrameGPU(cl_mem cl_target_buffer, int target_linesize,
                                cl_mem cl_source_buffer, int width,
                                int height) {
  if (!use_opencl) {
    std::cerr << "[SATDecoder::DecodeFrameGPU] Not initialized with OpenCL"
              << std::endl;
    return;
  }

  cl_int ret = 0;

  int source_linesize = 3 * width;
  // Set all the parameters and call the kernel
  ret = decode_kernel.setArg(0, sizeof(uint8_t *), &cl_target_buffer);
  ret = decode_kernel.setArg(1, sizeof(int), &target_linesize);
  ret = decode_kernel.setArg(2, sizeof(uint32_t *), &cl_source_buffer);
  ret = decode_kernel.setArg(3, sizeof(int), &width);
  ret = decode_kernel.setArg(4, sizeof(int), &height);
  ret = decode_kernel.setArg(5, sizeof(int), &source_linesize);

  size_t global_item_size[2] = {(size_t)width, (size_t)height};
  size_t local_item_size[2] = {1, 1};
  ret = clEnqueueNDRangeKernel(cl_manager->command_queue(), decode_kernel(), 0,
                               NULL, global_item_size, local_item_size, 0, NULL,
                               NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[SATDecoder::DecodeFrameGPU] decode kernel launch failed:"
              << ret << std::endl;
    return;
  }
  return;

Error:
  std::cerr << "[SATDecoder::DecodeFrameGPU] Some error occurred" << std::endl;
}

void SATDecoder::DecodeFrameCPU(AVFrame *target_frame, uint32_t *buffer,
                                AVCodecContext *codec_ctx) {
  uint8_t *output_buffer = target_frame->data[0];
  uint32_t *source_buffer = buffer;
  int width = codec_ctx->width;
  int height = codec_ctx->height;
  int input_linesize = 3 * width;
  int output_linesize = target_frame->linesize[0];
  int bytes_per_pixel = output_linesize / width;

  std::cout << "width: " << width << ", height: " << height << std::endl;

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      int x_pos = x;
      int y_pos = y;

      int delta_x = 1;
      int delta_y = 1;

      int target_coord = y_pos * output_linesize + x_pos * bytes_per_pixel;
      int sourceCoordinate0 = y_pos * input_linesize + x_pos * 3;

      if ((x_pos > 0) && (y_pos > 0)) {
        int truedelta_x = std::min(x_pos, delta_x);
        int truedelta_y = std::min(y_pos, delta_y);
        int rectangle_size = truedelta_x * truedelta_y;
        int sourceCoordinate1 =
            (y_pos - truedelta_y) * input_linesize + x_pos * 3;
        int sourceCoordinate2 =
            y_pos * input_linesize + (x_pos - truedelta_x) * 3;
        int sourceCoordinate3 =
            (y_pos - truedelta_y) * input_linesize + (x_pos - truedelta_x) * 3;
        output_buffer[target_coord] = (source_buffer[sourceCoordinate0] -
                                       source_buffer[sourceCoordinate1] +
                                       source_buffer[sourceCoordinate3] -
                                       source_buffer[sourceCoordinate2]) /
                                      rectangle_size;
        output_buffer[target_coord + 1] =
            (source_buffer[sourceCoordinate0 + 1] -
             source_buffer[sourceCoordinate1 + 1] +
             source_buffer[sourceCoordinate3 + 1] -
             source_buffer[sourceCoordinate2 + 1]) /
            rectangle_size;
        output_buffer[target_coord + 2] =
            (source_buffer[sourceCoordinate0 + 2] -
             source_buffer[sourceCoordinate1 + 2] +
             source_buffer[sourceCoordinate3 + 2] -
             source_buffer[sourceCoordinate2 + 2]) /
            rectangle_size;
      } else if (x_pos > 0) {
        int truedelta_x = std::min(x_pos, delta_x);
        int sourceCoordinate2 =
            y_pos * input_linesize + (x_pos - truedelta_x) * 3;
        output_buffer[target_coord] = (source_buffer[sourceCoordinate0] -
                                       source_buffer[sourceCoordinate2]) /
                                      truedelta_x;
        output_buffer[target_coord + 1] =
            (source_buffer[sourceCoordinate0 + 1] -
             source_buffer[sourceCoordinate2 + 1]) /
            truedelta_x;
        output_buffer[target_coord + 2] =
            (source_buffer[sourceCoordinate0 + 2] -
             source_buffer[sourceCoordinate2 + 2]) /
            truedelta_x;
      } else if (y_pos > 0) {
        int truedelta_y = std::min(y_pos, delta_y);
        int sourceCoordinate1 =
            (y_pos - truedelta_y) * input_linesize + x_pos * 3;
        output_buffer[target_coord] = (source_buffer[sourceCoordinate0] -
                                       source_buffer[sourceCoordinate1]) /
                                      truedelta_y;
        output_buffer[target_coord + 1] =
            (source_buffer[sourceCoordinate0 + 1] -
             source_buffer[sourceCoordinate1 + 1]) /
            truedelta_y;
        output_buffer[target_coord + 2] =
            (source_buffer[sourceCoordinate0 + 2] -
             source_buffer[sourceCoordinate1 + 2]) /
            truedelta_y;
      } else {
        output_buffer[target_coord] = source_buffer[sourceCoordinate0];
        output_buffer[target_coord + 1] = source_buffer[sourceCoordinate0 + 1];
        output_buffer[target_coord + 2] = source_buffer[sourceCoordinate0 + 2];
      }
    }
  }
}

void SATDecoder::SampleFrameRectGPU(cl_mem cl_target_buffer, int target_width,
                                    int target_height, int target_linesize,
                                    cl_mem cl_source_buffer,
                                    AVCodecContext *codec_ctx, float center_x,
                                    float center_y) {
  if (!use_opencl) {
    std::cerr << "[SATDecoder::SampleFrameRectGPU] Not initialized with OpenCL"
              << std::endl;
    return;
  }

  if (grid_size <= 0) {
    std::cerr << "[SATDecoder::SampleFrameRectGPU] Grid Not Initialized"
              << std::endl;
    InitializeGrid(target_width, target_height, codec_ctx->width,
                   codec_ctx->height);
  }

  cl_int ret = 0;

  cl_float2 center = {center_x, center_y};
  // Set all the parameters and call the kernel
  ret = sample_rect_kernel.setArg(0, sizeof(uint8_t *), &cl_target_buffer);
  ret = sample_rect_kernel.setArg(1, sizeof(int), &target_width);
  ret = sample_rect_kernel.setArg(2, sizeof(int), &target_height);
  ret = sample_rect_kernel.setArg(3, sizeof(int), &target_linesize);
  ret = sample_rect_kernel.setArg(4, sizeof(uint32_t *), &cl_source_buffer);
  ret = sample_rect_kernel.setArg(5, sizeof(int), &codec_ctx->width);
  ret = sample_rect_kernel.setArg(6, sizeof(int), &codec_ctx->height);
  ret = sample_rect_kernel.setArg(7, sizeof(int16_t *), &grid_buffer);
  ret = sample_rect_kernel.setArg(8, sizeof(cl_float2), &center);

  cl::NDRange global_item_size(8 * ((target_width + 7) / 8),
                               8 * ((target_height + 7) / 8));
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      sample_rect_kernel, 0, global_item_size, local_item_size, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr
        << "[SATDecoder::SampleFrameRectGPU] Sample rect kernel launch failed:"
        << ret << " " << OpenCLManager::GetCLErrorString(ret) << std::endl;
    return;
  }
  return;
Error:
  std::cerr << "[SATDecoder::SampleFrameRectGPU] Some error occurred"
            << std::endl;
}

void SATDecoder::SampleFrameRectGPU360(cl_mem cl_target_buffer,
                                       int target_width, int target_height,
                                       int target_linesize,
                                       cl_mem cl_source_buffer,
                                       AVCodecContext *codec_ctx,
                                       float center_x, float center_y) {
  if (!use_opencl) {
    std::cerr
        << "[SATDecoder::SampleFrameRectGPU360] Not initialized with OpenCL"
        << std::endl;
    return;
  }

  if (grid_size <= 0) {
    std::cerr << "[SATDecoder::SampleFrameRectGPU360] Grid Not Initialized"
              << std::endl;
    InitializeGrid(target_width, target_height, codec_ctx->width,
                   codec_ctx->height);
  }

  cl_int ret = 0;

  cl_float2 center = {center_x, center_y};
  // Set all the parameters and call the kernel
  ret = sample_rect_360_kernel.setArg(0, sizeof(uint8_t *), &cl_target_buffer);
  ret = sample_rect_360_kernel.setArg(1, sizeof(int), &target_width);
  ret = sample_rect_360_kernel.setArg(2, sizeof(int), &target_height);
  ret = sample_rect_360_kernel.setArg(3, sizeof(int), &target_linesize);
  ret = sample_rect_360_kernel.setArg(4, sizeof(uint32_t *), &cl_source_buffer);
  ret = sample_rect_360_kernel.setArg(5, sizeof(int), &codec_ctx->width);
  ret = sample_rect_360_kernel.setArg(6, sizeof(int), &codec_ctx->height);
  ret = sample_rect_360_kernel.setArg(7, sizeof(int16_t *), &grid_buffer);
  ret = sample_rect_360_kernel.setArg(8, sizeof(cl_float2), &center);

  cl::NDRange global_item_size(target_width, target_height);
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      sample_rect_360_kernel, 0, global_item_size, local_item_size, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr
        << "[SATDecoder::SampleFrameRectGPU] Sample rect kernel launch failed:"
        << ret << " " << OpenCLManager::GetCLErrorString(ret) << std::endl;
    return;
  }
  return;
Error:
  std::cerr << "[SATDecoder::SampleFrameRectGPU] Some error occurred"
            << std::endl;
}

void SATDecoder::SampleFrameRectCPU(AVFrame *target_frame, uint32_t *buffer,
                                    AVCodecContext *codec_ctx, float center_x,
                                    float center_y) {
  using namespace std;
  cl_int ret = 0;
  int source_width = codec_ctx->width;
  int source_height = codec_ctx->height;
  int input_bytes_per_pixel = 3;
  int input_linesize = 4 * source_width;

  int rect_buffer_width = target_frame->width;
  int rect_buffer_height = target_frame->height;

  uint32_t *source_buffer = buffer;
  uint8_t *output_buffer = target_frame->data[0];

  int output_linesize = target_frame->linesize[0];
  int output_bytes_per_pixel = output_linesize / rect_buffer_width;

  for (int i = 0; i < rect_buffer_width; i++) {
    for (int j = 0; j < rect_buffer_height; j++) {
      // Assume the height is 0 to 1
      // float aspectRatio = ((float)width / height);

      int u = i - rect_buffer_width / 2;
      int v = j - rect_buffer_height / 2;

      float lambdaX = source_width / (exp(1.0f) - 1);
      float lambdaY = source_height / (exp(1.0f) - 1);
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
                    (exp(pow(2.0 * abs(v - 1) / rect_buffer_height, 4.0)) -
                     1))) *
          ((v - 1 > 0) - (v - 1 < 0));
      int x_pos = center_x * source_width + delta_x;
      int x_pos_minus = center_x * source_width + delta_x_minus;
      int y_pos = center_y * source_height + delta_y;
      int y_pos_minus = center_y * source_height + delta_y_minus;

      if (x_pos >= 0 && x_pos < source_width && y_pos >= 0 &&
          y_pos < source_height) {
        x_pos_minus = min(max(x_pos_minus, 0), x_pos - 1);
        y_pos_minus = min(max(y_pos_minus, 0), y_pos - 1);
        int target_coord = j * output_linesize + i * output_bytes_per_pixel;
        if (x_pos > 0 && y_pos > 0) {
          int top_left_coord = y_pos_minus * input_linesize +
                               x_pos_minus * input_bytes_per_pixel;
          int top_right_coord =
              y_pos_minus * input_linesize + x_pos * input_bytes_per_pixel;
          int bottom_left_coord =
              y_pos * input_linesize + x_pos_minus * input_bytes_per_pixel;
          int bottom_right_coord =
              y_pos * input_linesize + x_pos * input_bytes_per_pixel;
          int rectangle_size = (x_pos - x_pos_minus) * (y_pos - y_pos_minus);
          output_buffer[target_coord] =
              (source_buffer[bottom_right_coord] -
               source_buffer[top_right_coord] + source_buffer[top_left_coord] -
               source_buffer[bottom_left_coord]) /
              rectangle_size;
          output_buffer[target_coord + 1] =
              (source_buffer[bottom_right_coord + 1] -
               source_buffer[top_right_coord + 1] +
               source_buffer[top_left_coord + 1] -
               source_buffer[bottom_left_coord + 1]) /
              rectangle_size;
          output_buffer[target_coord + 2] =
              (source_buffer[bottom_right_coord + 2] -
               source_buffer[top_right_coord + 2] +
               source_buffer[top_left_coord + 2] -
               source_buffer[bottom_left_coord + 2]) /
              rectangle_size;
        } else if (x_pos > 0) {
          // y_pos is 0
          int right_coordinate = x_pos * input_bytes_per_pixel;
          int left_coordinate = x_pos_minus * input_bytes_per_pixel;
          int rectangle_size = (x_pos - x_pos_minus);
          output_buffer[target_coord] = (source_buffer[right_coordinate] -
                                         source_buffer[left_coordinate]) /
                                        rectangle_size;
          output_buffer[target_coord + 1] =
              (source_buffer[right_coordinate + 1] -
               source_buffer[left_coordinate + 1]) /
              rectangle_size;
          output_buffer[target_coord + 2] =
              (source_buffer[right_coordinate + 2] -
               source_buffer[left_coordinate + 2]) /
              rectangle_size;
        } else if (y_pos > 0) {
          // x_pos is 0
          int top_coordinate = y_pos_minus * input_linesize;
          int bottom_coordinate = y_pos * input_linesize;
          int rectangle_size = y_pos - y_pos_minus;
          output_buffer[target_coord] = (source_buffer[bottom_coordinate] -
                                         source_buffer[top_coordinate]) /
                                        rectangle_size;
          output_buffer[target_coord + 1] =
              (source_buffer[bottom_coordinate + 1] -
               source_buffer[top_coordinate + 1]) /
              rectangle_size;
          output_buffer[target_coord + 2] =
              (source_buffer[bottom_coordinate + 2] -
               source_buffer[top_coordinate + 2]) /
              rectangle_size;
        } else {
          output_buffer[target_coord] = source_buffer[0];
          output_buffer[target_coord + 1] = source_buffer[1];
          output_buffer[target_coord + 2] = source_buffer[2];
        }
      }
    }
  }

  return;
Error:
  std::cerr << "Some error occurred in SampleFrameRect" << std::endl;
}

void SATDecoder::PrintClProgramBuildFailure(cl_int ret, cl_program program,
                                            cl_device_id device_id) {
  if (ret == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);

    // Allocate memory for the log
    char *log = (char *)malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size,
                          log, NULL);

    // Print the log
    printf("%s\n", log);
    fflush(stdout);
    free(log);
  }
}

void SATDecoder::ExpandSampledFrameRectCPU(AVFrame *target_frame,
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

void SATDecoder::InterpolateFrameRectCPU(AVFrame *target_frame,
                                         AVFrame *source_frame, float center_x,
                                         float center_y) {
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

void SATDecoder::CreateReducedSAT(cl_mem cl_target_buffer, int target_width,
                                  int target_height, cl_mem cl_source_buffer,
                                  int source_width, int source_height,
                                  int source_linesize, cl_mem u_buffer,
                                  cl_mem v_buffer, cl_mem sv_buffer,
                                  float center_x, float center_y, int sv_count,
                                  float delta_range[3]) {
  if (!use_opencl) {
    std::cerr << "[SATDecoder::CreateReducedSAT] Not initialized with OpenCL"
              << std::endl;
    return;
  }

  if (grid_size == -1) {
    std::cerr << "[SATDecoder::CreateReducedSAT] Grid Not Initialized"
              << std::endl;
    InitializeGrid(target_width, target_height, source_width, source_height);
  }

  cl_int ret = 0;
  cl_float3 cl_delta_range =
      (cl_float3){delta_range[0], delta_range[1], delta_range[2]};
  // Set all the parameters and call the kernel
  ret = create_reduced_sat_kernel.setArg(0, sizeof(cl_mem), &cl_target_buffer);
  ret = create_reduced_sat_kernel.setArg(1, sizeof(int), &target_width);
  ret = create_reduced_sat_kernel.setArg(2, sizeof(int), &target_height);
  ret = create_reduced_sat_kernel.setArg(3, sizeof(cl_mem), &cl_source_buffer);
  ret = create_reduced_sat_kernel.setArg(4, sizeof(int), &source_width);
  ret = create_reduced_sat_kernel.setArg(5, sizeof(int), &source_height);
  ret = create_reduced_sat_kernel.setArg(6, sizeof(int), &source_linesize);
  ret = create_reduced_sat_kernel.setArg(7, sizeof(cl_mem), &grid_buffer);
  ret = create_reduced_sat_kernel.setArg(8, sizeof(float), &center_x);
  ret = create_reduced_sat_kernel.setArg(9, sizeof(float), &center_y);
  ret = create_reduced_sat_kernel.setArg(10, sizeof(cl_mem), &u_buffer);
  ret = create_reduced_sat_kernel.setArg(11, sizeof(cl_mem), &v_buffer);
  ret = create_reduced_sat_kernel.setArg(12, sizeof(cl_mem), &sv_buffer);
  ret = create_reduced_sat_kernel.setArg(13, sizeof(int), &sv_count);
  ret =
      create_reduced_sat_kernel.setArg(14, sizeof(cl_float3), &cl_delta_range);
  if (ret != CL_SUCCESS) {
    std::cerr << "[SATDecoder::CreateReducedSAT] Set arg failed:" << ret << ":"
              << OpenCLManager::GetCLErrorString(ret) << std::endl;
    exit(EXIT_FAILURE);
  }

  cl::NDRange global_item_size(8 * (size_t)((target_width + 1 + 7) / 8),
                               8 * (size_t)((target_height + 1 + 7) / 8));
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      create_reduced_sat_kernel, 0, global_item_size, local_item_size, NULL,
      NULL);
  if (ret != CL_SUCCESS) {
    std::cerr
        << "[SATDecoder::CreateReducedSAT] Sample rect kernel launch failed:"
        << ret << ":" << OpenCLManager::GetCLErrorString(ret) << std::endl;
    exit(EXIT_FAILURE);
    return;
  }
  return;
}

void SATDecoder::SampleFrameFromReducedSAT(cl_mem cl_target_buffer,
                                           int target_width, int target_height,
                                           int target_linesize,
                                           cl_mem cl_reduced_sat) {
  if (!use_opencl) {
    std::cerr
        << "[SATDecoder::SampleFrameFromReducedSAT] Not initialized with OpenCL"
        << std::endl;
    return;
  }

  if (grid_size == -1) {
    std::cerr << "[SATDecoder::SampleFrameFromReducedSAT] Grid Not Initialized"
              << std::endl;
    exit(1);
  }

  cl_int ret = 0;

  // __global uchar *output_buffer, int output_width, int output_height,
  // int output_linesize, __global uint *source_buffer
  // Set all the parameters and call the kernel
  ret = sample_rect_from_reduced_sat_kernel.setArg(0, sizeof(cl_mem),
                                                   &cl_target_buffer);
  ret =
      sample_rect_from_reduced_sat_kernel.setArg(1, sizeof(int), &target_width);
  ret = sample_rect_from_reduced_sat_kernel.setArg(2, sizeof(int),
                                                   &target_height);
  ret = sample_rect_from_reduced_sat_kernel.setArg(3, sizeof(int),
                                                   &target_linesize);
  ret = sample_rect_from_reduced_sat_kernel.setArg(4, sizeof(cl_mem),
                                                   &cl_reduced_sat);
  ret = sample_rect_from_reduced_sat_kernel.setArg(5, sizeof(cl_mem),
                                                   &cl_reduced_sat);

  cl::NDRange global_item_size((size_t)target_width, (size_t)target_height);
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      sample_rect_from_reduced_sat_kernel, 0, global_item_size, local_item_size,
      NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[SATDecoder::SampleFrameFromReducedSAT] Sample rect kernel "
                 "launch failed:"
              << ret << std::endl;
    return;
  }
  return;
Error:
  std::cerr << "[SATDecoder::SampleFrameFromReducedSAT] Some error occurred"
            << std::endl;
}

void SATDecoder::InterpolateFrameRectGPU(
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
  ret = interpolate_kernel.setArg(0, sizeof(cl_mem), &cl_target_buffer);
  if (ret != CL_SUCCESS) {
    std::cerr << "Failed to set arg for target buffer: "
              << OpenCLManager::GetCLErrorString(ret) << std::endl;
  }
  ret = interpolate_kernel.setArg(1, sizeof(int), &target_width);
  ret = interpolate_kernel.setArg(2, sizeof(int), &target_height);
  ret = interpolate_kernel.setArg(3, sizeof(cl_mem), &cl_source_buffer);
  ret = interpolate_kernel.setArg(4, sizeof(int), &source_width);
  ret = interpolate_kernel.setArg(5, sizeof(int), &source_height);
  ret = interpolate_kernel.setArg(6, sizeof(cl_float2), &center);

  cl::NDRange global_item_size(8 * ((target_width + 7) / 8),
                               8 * ((target_height + 7) / 8));
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      interpolate_kernel, 0, global_item_size, local_item_size, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[SATDecoder::InterpolateFrameRectGPU] interpolate kernel "
                 "launch failed:"
              << ret << " " << OpenCLManager::GetCLErrorString(ret)
              << std::endl;
    exit(EXIT_FAILURE);
    return;
  }
  return;
}