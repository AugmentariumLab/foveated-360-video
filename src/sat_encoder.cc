#include "sat_encoder.h"

SATEncoder::SATEncoder() { use_OpenCL = false; }

SATEncoder::SATEncoder(OpenCLManager *cl_manager) {
  if (cl_manager->platform() == NULL) {
    std::cerr << "[SATEncoder::SATEncoder] cl_manager is not initialized"
              << std::endl;
    use_OpenCL = false;
    return;
  }
  use_OpenCL = true;
  cl_int ret = 0;
  this->cl_manager = cl_manager;

  std::ifstream encode_kernels_file("src/sat_encoder_encode_kernels.cl");
  std::string encode_kernel_contents(
      (std::istreambuf_iterator<char>(encode_kernels_file)),
      std::istreambuf_iterator<char>());
  const size_t encode_kernel_length = encode_kernel_contents.length();
  const char *encode_kernel_chars = encode_kernel_contents.c_str();
  encode_program =
      clCreateProgramWithSource(cl_manager->context(), 1, &encode_kernel_chars,
                                &encode_kernel_length, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "create encode program failed:" << ret << std::endl;
  }
  ret = clBuildProgram(encode_program, 1, &cl_manager->device(), NULL, NULL,
                       NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "build encode program failed:" << ret << std::endl;
    PrintClProgramBuildFailure(ret, encode_program, cl_manager->device());
  }
  copy_image_kernel = clCreateKernel(encode_program, "copy_image_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "create copy image kernel failed:" << ret << std::endl;
  }
  copy_image_back_kernel =
      clCreateKernel(encode_program, "copy_image_back_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "create copy image back kernel failed:" << ret << std::endl;
  }
  scan_rows_kernel = clCreateKernel(encode_program, "scan_rows_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "create copy image kernel failed:" << ret << std::endl;
  }
  scan_columns_kernel =
      clCreateKernel(encode_program, "scan_columns_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "create copy image kernel failed:" << ret << std::endl;
  }
}

SATEncoder::~SATEncoder() { FreeClResources(); }

void SATEncoder::FreeClResources() {
  cl_int ret = 0;
  if (use_OpenCL) {
    ret = clReleaseKernel(copy_image_kernel);
    ret = clReleaseKernel(copy_image_back_kernel);
    ret = clReleaseKernel(scan_rows_kernel);
    ret = clReleaseKernel(scan_columns_kernel);
    ret = clReleaseProgram(encode_program);
  }
}

void SATEncoder::EncodeFrameGPU(cl_mem cl_target_buffer,
                                cl_mem cl_source_buffer, int source_width,
                                int source_height, int source_linesize) {
  if (!use_OpenCL) {
    std::cerr << "[SATEncoder::EncodeFrameGPU] Not initialized with OpenCL"
              << std::endl;
    return;
  }
  cl_int ret = 0;

  int target_linesize = 3 * source_width;
  // Set all the parameters and call the kernel
  ret = clSetKernelArg(copy_image_kernel, 0, sizeof(uint32_t *),
                       &cl_target_buffer);
  ret = clSetKernelArg(copy_image_kernel, 1, sizeof(int), &target_linesize);
  ret = clSetKernelArg(copy_image_kernel, 2, sizeof(uint8_t *),
                       &cl_source_buffer);
  ret = clSetKernelArg(copy_image_kernel, 3, sizeof(int), &source_width);
  ret = clSetKernelArg(copy_image_kernel, 4, sizeof(int), &source_height);
  ret = clSetKernelArg(copy_image_kernel, 5, sizeof(int), &source_linesize);

  size_t global_item_size[2] = {(size_t)source_width, (size_t)source_height};
  size_t local_item_size[2] = {8, 8};
  ret = clEnqueueNDRangeKernel(cl_manager->command_queue(), copy_image_kernel,
                               2, NULL, global_item_size, local_item_size, 0,
                               NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "Sample rect kernel launch failed:" << ret << std::endl;
    return;
  }

  // Set all the parameters and call the kernel
  ret = clSetKernelArg(scan_rows_kernel, 0, sizeof(uint32_t *),
                       &cl_target_buffer);
  ret = clSetKernelArg(scan_rows_kernel, 1, sizeof(int), &source_width);
  ret = clSetKernelArg(scan_rows_kernel, 2, sizeof(int), &source_height);
  ret = clSetKernelArg(scan_rows_kernel, 3, sizeof(int), &target_linesize);

  size_t global_item_size2 = source_height;
  size_t local_item_size2 = 8;
  ret = clEnqueueNDRangeKernel(cl_manager->command_queue(), scan_rows_kernel, 1,
                               NULL, &global_item_size2, &local_item_size2, 0,
                               NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "Sample rect kernel launch failed:" << ret << std::endl;
    return;
  }

  // Set all the parameters and call the kernel
  ret = clSetKernelArg(scan_columns_kernel, 0, sizeof(uint32_t *),
                       &cl_target_buffer);
  ret = clSetKernelArg(scan_columns_kernel, 1, sizeof(int), &source_width);
  ret = clSetKernelArg(scan_columns_kernel, 2, sizeof(int), &source_height);
  ret = clSetKernelArg(scan_columns_kernel, 3, sizeof(int), &target_linesize);

  size_t global_item_size3 = source_width;
  size_t local_item_size3 = 8;
  ret = clEnqueueNDRangeKernel(cl_manager->command_queue(), scan_columns_kernel,
                               1, NULL, &global_item_size3, &local_item_size3,
                               0, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "Sample rect kernel launch failed:" << ret << std::endl;
    return;
  }

  return;
Error:
  std::cerr << "Some error occurred in SampleFrameRect" << std::endl;
}

void SATEncoder::EncodeFrameCPU(uint32_t *target_frame,
                                AVCodecContext *codec_ctx, AVFrame *frame) {
  int width = codec_ctx->width;
  int height = codec_ctx->height;
  int input_linesize = frame->linesize[0];
  int bytes_per_pixel = input_linesize / width;
  int output_per_pixel = 3;
  int output_linesize = output_per_pixel * width;

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      int source_index = y * input_linesize + x * bytes_per_pixel;
      int target_index = y * output_linesize + x * output_per_pixel;
      target_frame[target_index] = (uint32_t)(frame->data[0][source_index]);
      target_frame[target_index + 1] =
          (uint32_t)(frame->data[0][source_index + 1]);
      target_frame[target_index + 2] =
          (uint32_t)(frame->data[0][source_index + 2]);
    }
  }

  for (int x = 1; x < width; x++) {
    for (int y = 0; y < height; y++) {
      int prev_index = y * output_linesize + (x - 1) * output_per_pixel;
      int target_index = y * output_linesize + x * output_per_pixel;

      target_frame[target_index] =
          target_frame[target_index] + target_frame[prev_index];
      target_frame[target_index + 1] =
          target_frame[target_index + 1] + target_frame[prev_index + 1];
      target_frame[target_index + 2] =
          target_frame[target_index + 2] + target_frame[prev_index + 2];
    }
  }

  for (int y = 1; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int prev_index = (y - 1) * output_linesize + x * output_per_pixel;
      int target_index = y * output_linesize + x * output_per_pixel;

      target_frame[target_index] =
          target_frame[target_index] + target_frame[prev_index];
      target_frame[target_index + 1] =
          target_frame[target_index + 1] + target_frame[prev_index + 1];
      target_frame[target_index + 2] =
          target_frame[target_index + 2] + target_frame[prev_index + 2];
    }
  }
}

void SATEncoder::PrintClProgramBuildFailure(cl_int ret, cl_program program,
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
