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

// Rounds x up to the nearest y. E.g. ROUND_UP_TO(5, 8) == 8
#define ROUND_UP_TO(x, y) (y * ((x + y - 1) / y))

/**
 * The ImageSampler class samples directly from the RGB image.
 * It's used for testing what the foveated image would appear like
 * if the SAT was not used.
 * Most of the code is extracted and simplified from the SATDecoder class.
 */
class ImageSampler {
 private:
  OpenCLManager *cl_manager;
  cl::Program sample_rect_program;
  cl::Kernel sample_rect_kernel;
  cl::Kernel create_grid_kernel;
  cl::Program sample_logpolar_program;
  cl::Kernel sample_logpolar_kernel;
  cl::Kernel create_logpolar_grid_kernel;
  cl::Program interpolate_program;
  cl::Kernel interpolate_logpolar_kernel;
  cl::Kernel logpolar_gaussian_blur_kernel;
  cl::Program sample_mipmap_logpolar_program;
  cl::Kernel generate_image_pyramid_kernel;
  cl::Kernel sample_logpolar_from_image_pyramid_kernel;
  cl::Buffer grid_buffer;
  size_t grid_size = 0;
  cl::Buffer logpolar_grid_buffer;
  size_t logpolar_grid_size = 0;

  bool use_opencl = false;
  float clamp(float a, float b, float c) { return std::min(std::max(a, b), c); }
  float lerp(float a, float b, float c) { return a * (1.0 - c) + b * c; }

 public:
  ImageSampler();
  ImageSampler(OpenCLManager *cl_manager);
  ~ImageSampler();
  void InitializeGrid(int target_width, int target_height, int source_width,
                      int source_height);
  void InitializeLogpolarGrid(int target_width, int target_height,
                              int source_width, int source_height);
  void SampleFrameRectGPU(cl_mem cl_target_buffer, int target_width,
                          int target_height, int target_linesize,
                          cl_mem cl_source_buffer, int source_width,
                          int source_height, int source_linesize,
                          float center_x, float center_y);
  void SampleFrameRectCPU(AVFrame *target_frame, uint32_t *buffer,
                          AVCodecContext *codec_ctx, float center_x,
                          float center_y);
  void ExpandSampledFrameRectCPU(AVFrame *target_frame, AVFrame *source_frame,
                                 float center_x, float center_y);
  void InterpolateFrameRectCPU(AVFrame *target_frame, AVFrame *source_frame,
                               float center_x, float center_y);

  void SampleFrameLogPolarGPU(cl_mem cl_target_buffer, int target_width,
                              int target_height, int target_linesize,
                              cl_mem cl_source_buffer, int source_width,
                              int source_height, int source_linesize,
                              float center_x, float center_y);
  void ExpandSampledFrameLogPolarCPU(AVFrame *target_frame,
                                     AVFrame *source_frame, float center_x,
                                     float center_y);
  void InterpolateFrameLogPolarCPU(AVFrame *target_frame, AVFrame *source_frame,
                                   float center_x, float center_y);
  void InterpolateFrameLogPolarGPU(cl_mem cl_target_buffer, int target_width,
                                   int target_height, int target_linesize,
                                   cl_mem cl_source_buffer, int source_width,
                                   int source_height, int source_linesize,
                                   float center_x, float center_y);
  void ApplyLogPolarGaussianBlur(cl_mem cl_target_buffer, int target_width,
                                 int target_height, int target_linesize,
                                 cl_mem cl_source_buffer);

  void GenerateImagePyramid(cl_mem cl_image_pyramid_sizes,
                            cl_mem cl_target_buffer, int target_width,
                            int target_height, cl_mem cl_source_buffer,
                                        int image_pyramid_layers);
  void SampleFrameLogPolarGPUFromImagePyramid(
      cl_mem cl_target_buffer, int target_width, int target_height,
      int target_linesize, cl_mem cl_image_pyramid, int source_width,
      int source_height, cl_mem cl_image_pyramid_sizes, float center_x,
      float center_y, int pyramid_levels);
};
