#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "opencl_manager.h"

class SATDecoder {
 private:
  OpenCLManager *cl_manager;
  cl::Program decode_program;
  cl::Kernel decode_kernel;
  cl::Program sample_rect_program;
  cl::Kernel sample_rect_kernel;
  cl::Kernel sample_rect_360_kernel;
  cl::Kernel create_grid_kernel;
  cl::Kernel sample_rect_from_reduced_sat_kernel;
  cl::Kernel create_reduced_sat_kernel;
  cl::Program interpolate_program;
  cl::Kernel interpolate_kernel;
  cl::Buffer grid_buffer;
  int64_t grid_size = -1;

  bool use_opencl = false;

  void FreeClResources();
  void PrintClProgramBuildFailure(cl_int ret, cl_program program,
                                  cl_device_id device_id);
  float clamp(float a, float b, float c) { return std::min(std::max(a, b), c); }
  float lerp(float a, float b, float c) { return a * (1.0 - c) + b * c; }

 public:
  SATDecoder();
  SATDecoder(OpenCLManager *cl_manager);
  ~SATDecoder();
  void InitializeGrid(int target_width, int target_height, int source_width,
                      int source_height);
  void DecodeFrameGPU(cl_mem cl_target_buffer, int target_linesize,
                      cl_mem cl_source_buffer, int width, int height);
  void DecodeFrameCPU(AVFrame *target_frame, uint32_t *buffer,
                      AVCodecContext *codec_ctx);
  void CreateReducedSAT(cl_mem cl_target_buffer, int target_width,
                        int target_height, cl_mem cl_source_buffer,
                        int source_width, int source_height,
                        int source_linesize, cl_mem u_buffer, cl_mem v_buffer,
                        cl_mem sv_buffer, float center_x, float center_y,
                        int sv_count, float delta_range[3]);
  void SampleFrameFromReducedSAT(cl_mem cl_target_buffer, int target_width,
                                 int target_height, int target_linesize,
                                 cl_mem cl_reduced_sat);
  void SampleFrameRectGPU(cl_mem cl_target_buffer, int target_width,
                          int target_height, int target_linesize,
                          cl_mem cl_source_buffer, AVCodecContext *codec_ctx,
                          float center_x, float center_y);
  void SampleFrameRectGPU360(cl_mem cl_target_buffer, int target_width,
                             int target_height, int target_linesize,
                             cl_mem cl_source_buffer, AVCodecContext *codec_ctx,
                             float center_x, float center_y);
  void SampleFrameRectCPU(AVFrame *target_frame, uint32_t *buffer,
                          AVCodecContext *codec_ctx, float center_x,
                          float center_y);
  void ExpandSampledFrameRectCPU(AVFrame *target_frame, AVFrame *source_frame,
                                 float center_x, float center_y);
  void InterpolateFrameRectCPU(AVFrame *target_frame, AVFrame *source_frame,
                               float center_x, float center_y);
  void InterpolateFrameRectGPU(cl_mem cl_target_buffer, int target_width,
                               int target_height, int target_linesize,
                               cl_mem cl_source_buffer, int source_width,
                               int source_height, int source_linesize,
                               float center_x, float center_y);
};
