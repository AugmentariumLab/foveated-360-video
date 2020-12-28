#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.hpp>
#include <CL/cl_gl.h>
#include <iostream>

class OpenCLManager {
 private:
 public:
  cl::Platform platform;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue command_queue;
  std::vector<cl_context_properties> context_properties{0};
  cl_context_properties gl_context = -1;
  cl_context_properties gl_display = -1;
  OpenCLManager();
  ~OpenCLManager();
  int InitializeContext();
  static std::string GetCLErrorString(cl_int error);
};