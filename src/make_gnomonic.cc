#include <cpp-base64/base64.h>
#include <zlib.h>

#include <CL/cl.hpp>
#include <Eigen/Core>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "gaze_view_points.h"
#include "opencl_manager.h"
#include "parameters.h"
#include "projections.h"
#include "save_frame.h"
#include "video_decoder.h"
#include "video_encoder.h"

struct AVFrameDeleter {
  void operator()(AVFrame *p) { av_frame_free(&p); }
};

int MakeGnomonicImage(std::vector<std::string> args);
int MakeGnomonicVideo(std::vector<std::string> args);

/**
 * @brief Reproject an Equirectangular video using the gnomonic projection.
 *
 * @return int
 */
int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) return EXIT_FAILURE;
  std::string input_file = args[1];
  std::string extension = input_file.substr(input_file.find_last_of(".") + 1);
  if (extension == "mp4") {
    return MakeGnomonicVideo(args);
  } else if (extension == "png") {
    return MakeGnomonicImage(args);
  }
  std::cerr << "Unknown extension: " << extension << std::endl;
  return EXIT_FAILURE;
}

/**
 * @brief Reproject ERP image to gnomonic projection.
 * 
 * @param args 
 * @return int 
 */
int MakeGnomonicImage(std::vector<std::string> args) {
  using std::chrono::high_resolution_clock;
  using GazeViewPoint = GazeViewPoints::GazeViewPoint;

  std::string source_image;
  std::string output_image;

  if (args.size() >= 3) {
    source_image = args[1];
    output_image = args[2];
  } else {
    return EXIT_FAILURE;
  }

  float center_x = 0.5;
  float center_y = 0.5;

  if (args.size() >= 5) {
    center_x = std::stof(args[3]);
    center_y = std::stof(args[4]);
  }

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  Projections projection(&cl_manager);

  int ret = -1;

  std::unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  std::unique_ptr<AVFrame, AVFrameDeleter> rgba_frame(av_frame_alloc());

  LoadFramePNG(rgb_frame.get(), source_image);

  int width = rgb_frame->width;
  int height = rgb_frame->height;
  rgba_frame->width = rgb_frame->width;
  rgba_frame->height = rgb_frame->height;
  rgba_frame->format = AV_PIX_FMT_RGB0;
  av_frame_get_buffer(rgba_frame.get(), 1);
  auto sws_ctx = sws_getContext(
      width, height, (AVPixelFormat)rgb_frame->format, width, height,
      (AVPixelFormat)rgba_frame->format, SWS_BILINEAR, NULL, NULL, NULL);
  ret = sws_scale(sws_ctx, rgb_frame->data, rgb_frame->linesize, 0, height,
                  rgba_frame->data, rgba_frame->linesize);
  sws_freeContext(sws_ctx);

  int cl_source_frame_size = cl_source_frame_size =
      4 * rgb_frame->width * rgb_frame->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_WRITE,
                             cl_source_frame_size);
  int cl_output_buffer_size = cl_source_frame_size;
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  // Copy RGB frame to GPU
  ret = cl::copy(cl_manager.command_queue, rgba_frame->data[0],
                 rgba_frame->data[0] + cl_source_frame_size, cl_source_frame);

  projection.GnomonicProjection(cl_output_buffer(), width, height,
                                rgba_frame->linesize[0], cl_source_frame(),
                                width, height, rgb_frame->linesize[0], center_x,
                                center_y);

  ret =
      cl::copy(cl_manager.command_queue, cl_output_buffer, rgba_frame->data[0],
               rgba_frame->data[0] + cl_source_frame_size);
  if (ret != CL_SUCCESS) {
    std::cerr << "Failed to copy off of GPU " << std::endl;
    exit(EXIT_FAILURE);
  }
  SaveFramePNG(rgba_frame.get(), output_image);

  return EXIT_SUCCESS;
}

/**
 * @brief Reproject ERP video to gnomonic video using viewport as gaze position.
 * 
 * @param args 
 * @return int 
 */
int MakeGnomonicVideo(std::vector<std::string> args) {
  using std::chrono::high_resolution_clock;
  using GazeViewPoint = GazeViewPoints::GazeViewPoint;

  std::string source_video;
  std::string gaze_file;
  std::string output_video;

  bool use_static_view = false;
  float center_x = 0.5f;
  float center_y = 0.5f;

  if (args.size() > 3) {
    source_video = args[1];
    gaze_file = args[2];
    output_video = args[3];
  } else {
    return EXIT_FAILURE;
  }

  std::cout << "Source video: " << source_video << std::endl;
  std::cout << "Output video: " << output_video << std::endl;
  std::cout << "Gaze file: " << gaze_file << std::endl;

  auto length = std::chrono::hours(10);

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;
  VideoEncoder video_encoder(source_codec_ctx, NULL, output_video);
  GazeViewPoints gaze_view_points(gaze_file);
  std::cerr << "Number of view points found: " << gaze_view_points.points.size()
            << std::endl;
  Projections projection(&cl_manager);

  int ret = -1;
  bool continue_loop = true;
  int frame = 0;

  std::unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());

  int width = source_codec_ctx->width;
  int height = source_codec_ctx->height;

  int cl_source_frame_size = cl_source_frame_size =
      4 * source_codec_ctx->width * source_codec_ctx->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_WRITE,
                             cl_source_frame_size);
  int cl_output_buffer_size = cl_source_frame_size;
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  // Get frame from decoder
  int64_t maxFrames = 30 * std::chrono::seconds(length).count();
  while (continue_loop && frame < maxFrames) {
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
    if (ret == 0) {
      if (frame % 30 == 0) {
        std::cout << "Processing frame " << frame << std::endl;
      }

      GazeViewPoint gv = gaze_view_points.points[frame];
      if (!use_static_view) {
        center_x = gv.view_point[0];
        center_y = gv.view_point[1];
      }

      // Copy RGB frame to GPU
      ret =
          cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                   rgb_frame->data[0] + cl_source_frame_size, cl_source_frame);

      projection.GnomonicProjection(cl_output_buffer(), width, height,
                                    rgb_frame->linesize[0], cl_source_frame(),
                                    width, height, rgb_frame->linesize[0],
                                    center_x, center_y);

      ret = cl::copy(cl_manager.command_queue, cl_output_buffer,
                     rgb_frame->data[0],
                     rgb_frame->data[0] + cl_source_frame_size);
      if (ret != CL_SUCCESS) {
        std::cerr << "Failed to copy off of GPU " << std::endl;
        exit(EXIT_FAILURE);
      }
      ret = video_encoder.EncodeFrameToFile(rgb_frame.get());
      // SaveFramePNG(rgb_frame.get(), "projection_test.png");
      // std::exit(EXIT_FAILURE);
      frame++;
    } else {
      continue_loop = false;
    }
  }

  video_encoder.EncodeFrameToFile(NULL);
  video_encoder.WriteTrailerAndCloseFile();
  return EXIT_SUCCESS;
}