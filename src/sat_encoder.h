#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "opencl_manager.h"

class SATEncoder {
 private:
  bool use_OpenCL;
  OpenCLManager *cl_manager;
  cl_program encode_program;
  cl_kernel copy_image_kernel;
  cl_kernel copy_image_back_kernel;
  cl_kernel scan_rows_kernel;
  cl_kernel scan_columns_kernel;

  void FreeClResources();
  void PrintClProgramBuildFailure(cl_int ret, cl_program program,
                                  cl_device_id device_id);

 public:
  SATEncoder();
  SATEncoder(OpenCLManager *cl_manager);
  ~SATEncoder();
  void EncodeFrameGPU(cl_mem cl_target_buffer, cl_mem cl_source_buffer,
                      int source_width, int source_height, int source_linesize);
  void EncodeFrameCPU(uint32_t *target_frame, AVCodecContext *codec_ctx,
                      AVFrame *frame);
};
