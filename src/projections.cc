#include "projections.h"

Projections::Projections(OpenCLManager *cl_manager) {
  this->cl_manager = cl_manager;
  cl_int ret = 0;

  std::string projection_program_contents;
  {
    std::ifstream projection_program_file("src/projections_program.cl");
    if (projection_program_file) {
      projection_program_file.seekg(0, std::ios::end);
      projection_program_contents.resize(projection_program_file.tellg());
      projection_program_file.seekg(0, std::ios::beg);
      projection_program_file.read(&projection_program_contents[0],
                                   projection_program_contents.size());
    }
  }
  if (projection_program_contents.empty()) {
    std::cerr << "Projection Program not found" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  my_program = cl::Program(cl_manager->context, projection_program_contents,
                           false, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << " Create projections program failed:" << ret
              << std::endl;
    exit(EXIT_FAILURE);
  }
  ret = my_program.build();
  if (ret != CL_SUCCESS) {
    std::string build_log;
    my_program.getBuildInfo<std::string>(cl_manager->device,
                                         CL_PROGRAM_BUILD_LOG, &build_log);
    std::cerr << __FUNCTION__ << " Build projections program failed:" << ret
              << " " << OpenCLManager::GetCLErrorString(ret) << std::endl
              << build_log << std::endl;
    exit(EXIT_FAILURE);
  }

  gnomonic_kernel = cl::Kernel(my_program, "gnomonic_kernel", &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << __FUNCTION__ << " Create gnomonic kernel failed:" << ret
              << std::endl;
    exit(EXIT_FAILURE);
  }
  use_OpenCL = true;
}

Projections::~Projections() {}

void Projections::GnomonicProjection(cl_mem cl_target_buffer, int target_width,
                                     int target_height, int target_linesize,
                                     cl_mem cl_source_buffer, int source_width,
                                     int source_height, int source_linesize,
                                     float center_x, float center_y) {
  if (!use_OpenCL) {
    std::cerr << __FUNCTION__ << " Not initialized with OpenCL" << std::endl;
    return;
  }

  cl_int ret = 0;

  cl_float2 center = {center_x, center_y};
  // Set all the parameters and call the kernel
  ret = gnomonic_kernel.setArg(0, sizeof(cl_mem), &cl_target_buffer);
  ret = gnomonic_kernel.setArg(1, sizeof(int), &target_width);
  ret = gnomonic_kernel.setArg(2, sizeof(int), &target_height);
  ret = gnomonic_kernel.setArg(3, sizeof(cl_mem), &cl_source_buffer);
  ret = gnomonic_kernel.setArg(4, sizeof(int), &source_width);
  ret = gnomonic_kernel.setArg(5, sizeof(int), &source_height);
  ret = gnomonic_kernel.setArg(6, sizeof(cl_float2), &center);

  cl::NDRange global_item_size(8 * ((target_width + 7) / 8),
                               8 * ((target_height + 7) / 8));
  cl::NDRange local_item_size(8, 8);
  ret = cl_manager->command_queue.enqueueNDRangeKernel(
      gnomonic_kernel, 0, global_item_size, local_item_size, NULL, NULL);
  if (ret != CL_SUCCESS) {
    std::cerr << "[GnomonicProjection] Gnomonic kernel launch "
                 "failed:"
              << ret << " " << OpenCLManager::GetCLErrorString(ret)
              << std::endl;
    return;
  }
  return;
}
