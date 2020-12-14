#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <CL/cl.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "opencl_manager.h"

class Projections {
 private:
  bool use_OpenCL;
  OpenCLManager *cl_manager;
  cl::Program my_program;
  cl::Kernel gnomonic_kernel;

 public:
  Projections(OpenCLManager *cl_manager);
  ~Projections();
  void GnomonicProjection(cl_mem cl_target_buffer, int target_height,
                          int target_width, int target_linesize,
                          cl_mem cl_source_buffer, int source_width,
                          int source_height, int source_linesize,
                          float center_x, float center_y);
};
