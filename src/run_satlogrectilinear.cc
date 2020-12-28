#include <cpp-base64/base64.h>
#include <zlib.h>

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <ratio>
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
#include "sat_decoder.h"
#include "sat_encoder.h"
#include "save_frame.h"
#include "video_decoder.h"
#include "video_encoder.h"

namespace fs = std::filesystem;

int TestFunction(const std::vector<std::string> &args);
int SingleFrameExtraction(const std::vector<std::string> &args);
int ExtractSampledFrameRect(const std::vector<std::string> &args);
int InterpolateSampledFrameRect(const std::vector<std::string> &args);
int EncodeVideo(const std::vector<std::string> &args);
int EncodeLogCartesianVideo(const std::vector<std::string> &args);
int EncodeLogCartesianVideoBitrate(const std::vector<std::string> &args);
int DecodeLogCartesianVideo(const std::vector<std::string> &args);
int FoveateLogCartesianVideo(const std::vector<std::string> &args);

struct AVFrameDeleter {
  void operator()(AVFrame *p) { av_frame_free(&p); }
};

/**
 * @brief Code for testing sampling from the summed area table
 *
 * @return int
 */
int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv, argv + argc);
  if (args[1] == "single_frame") {
    return SingleFrameExtraction(args);
  } else if (args[1] == "interpolate_sampled") {
    return InterpolateSampledFrameRect(args);
  } else if (args[1] == "encode_bitrate") {
    return EncodeLogCartesianVideoBitrate(args);
  } else if (args[1] == "decode") {
    return DecodeLogCartesianVideo(args);
  } else if (args[1] == "foveate_no_encoding") {
    return FoveateLogCartesianVideo(args);
  }
  return EXIT_SUCCESS;
}

int TestFunction(const std::vector<std::string> args) {
  // int64_t kilo = std::kilo::num / std::kilo::den;
  // int64_t mega = std::mega::num / std::mega::den;
  // for (int64_t bitrate = 100 * kilo; bitrate <= 10 * mega; bitrate *= 2) {
  //   std::cerr << "Foveating" << std::endl;
  //   EncodeLogCartesianVideoBitrate(args, bitrate);
  //   std::cerr << "Unfoveating" << std::endl;
  //   UnwarpLogCartesianVideoBitrate(args, bitrate);
  // }
  // return EXIT_SUCCESS;
  using std::unique_ptr;
  using std::filesystem::path;

  path source_video = "360_em_dataset/1080p_videos/03_drone_d5d4gnuAJLo.mp4";
  path output_file = "output_videos/temp/tt.png";
  uint64_t frame_to_extract = 21 * 30;

  float center_x = 0.0;
  float center_y = 1.0;

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  SATEncoder sat_encoder(&cl_manager);
  SATDecoder sat_decoder(&cl_manager);

  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;

  int ret = -1;

  // Get frame from decoder
  unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  for (uint64_t i = 0; i < frame_to_extract; i++) {
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
  }
  if (ret != 0) {
    std::cerr << "Failed to get rgb frame" << std::endl;
  }

  int width = source_codec_ctx->width;
  int height = source_codec_ctx->height;
  int reduced_width = 16 * std::ceil(width / 1.8 / 16);
  int reduced_height = 16 * std::ceil(height / 1.8 / 16);
  sat_decoder.InitializeGrid(reduced_width, reduced_height,
                             source_codec_ctx->width, source_codec_ctx->height);

  unique_ptr<AVFrame, AVFrameDeleter> rect_frame(av_frame_alloc());
  rect_frame->width = reduced_width;
  rect_frame->height = reduced_height;
  rect_frame->format = AV_PIX_FMT_RGB0;
  av_frame_get_buffer(rect_frame.get(), 0);

  // Copy the frame to the OpenCL GPU
  int source_frame_size = rgb_frame->linesize[0] * rgb_frame->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_ONLY,
                             source_frame_size);
  // Copy RGB frame to GPU
  ret = cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                 rgb_frame->data[0] + source_frame_size, cl_source_frame);

  int sat_buffer_size = width * height * 3 * sizeof(uint32_t);
  cl::Buffer cl_sat_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                           sat_buffer_size);

  // Encode SAT
  sat_encoder.EncodeFrameGPU(cl_sat_buffer(), cl_source_frame(),
                             rgb_frame->width, rgb_frame->height,
                             rgb_frame->linesize[0]);

  int cl_output_buffer_size = rect_frame->height * rect_frame->linesize[0];
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);
  sat_decoder.SampleFrameRectGPU(cl_output_buffer(), rect_frame->width,
                                 rect_frame->height, rect_frame->linesize[0],
                                 cl_sat_buffer(), source_codec_ctx, center_x,
                                 center_y);

  sat_decoder.InterpolateFrameRectGPU(
      cl_source_frame(), rgb_frame->width, rgb_frame->height,
      rgb_frame->linesize[0], cl_output_buffer(), rect_frame->width,
      rect_frame->height, rect_frame->linesize[0], center_x, center_y);
  ret = cl::copy(
      cl_manager.command_queue, cl_source_frame, rgb_frame->data[0],
      rgb_frame->data[0] + (rgb_frame->height * rgb_frame->linesize[0]));

  // // Copy off of GPU
  // ret =
  //     cl::copy(cl_manager.command_queue, cl_output_buffer,
  //     rect_frame->data[0],
  //              rect_frame->data[0] + cl_output_buffer_size);
  // if (ret != CL_SUCCESS) {
  //   std::cout << "Failed to copy frame off of GPU" << std::endl;
  //   return EXIT_FAILURE;
  // }

  // sat_decoder.InterpolateFrameRectCPU(rgb_frame.get(), rect_frame.get(),
  //                                     center_x, center_y);
  SaveFramePNG(rgb_frame.get(), output_file);
  return EXIT_SUCCESS;
}

int SingleFrameExtraction(const std::vector<std::string> &args) {
  std::string source_video =
      "360_em_dataset/1080p_videos/03_drone_d5d4gnuAJLo.mp4";
  std::string output_file = "single_frame_satlogrectilinear";
  uint64_t frame_to_extract = 100;

  float center_x = 0.65f;
  float center_y = 0.75f;

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  SATEncoder sat_encoder(&cl_manager);
  SATDecoder sat_decoder(&cl_manager);

  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;

  int ret = -1;

  std::unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  for (uint64_t i = 0; i < frame_to_extract; i++) {
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
  }
  if (ret != 0) {
    std::cerr << "Failed to get rgb frame" << std::endl;
    exit(EXIT_FAILURE);
  }

  int width = source_codec_ctx->width;
  int height = source_codec_ctx->height;

  std::unique_ptr<AVFrame, AVFrameDeleter> rect_frame(av_frame_alloc());
  rect_frame->width = REDUCED_BUFFER_WIDTH;
  rect_frame->height = REDUCED_BUFFER_HEIGHT;
  rect_frame->format = AV_PIX_FMT_RGB0;
  av_frame_get_buffer(rect_frame.get(), 0);

  // Copy the frame to the OpenCL GPU
  int source_frame_size = rgb_frame->linesize[0] * rgb_frame->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_ONLY,
                             source_frame_size);
  ret = cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                 rgb_frame->data[0] + source_frame_size, cl_source_frame);

  int sat_buffer_size = width * height * 3 * sizeof(uint32_t);
  cl::Buffer cl_sat_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                           sat_buffer_size);

  sat_encoder.EncodeFrameGPU(cl_sat_buffer(), cl_source_frame(),
                             rgb_frame->width, rgb_frame->height,
                             rgb_frame->linesize[0]);

  int cl_output_buffer_size = rect_frame->height * rect_frame->linesize[0];
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  sat_decoder.SampleFrameRectGPU(cl_output_buffer(), rect_frame->width,
                                 rect_frame->height, rect_frame->linesize[0],
                                 cl_sat_buffer(), source_codec_ctx, center_x,
                                 center_y);

  // Copy off of GPU
  ret =
      cl::copy(cl_manager.command_queue, cl_output_buffer, rect_frame->data[0],
               rect_frame->data[0] + cl_output_buffer_size);

  SaveFramePNG(rect_frame.get(), output_file);
  return 0;
}

int ExtractSampledFrameRect(const std::vector<std::string> &args) {
  using namespace std::chrono;

  std::string source_video = "bulgaria_1080_30.mp4";
  source_video = "360_em_dataset/1080p_videos/03_drone_d5d4gnuAJLo.mp4";
  std::string output_file = "large_rect_buffer_expanded";
  uint64_t frame_to_extract = 100;

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  SATEncoder sat_encoder(&cl_manager);
  SATDecoder sat_decoder(&cl_manager);

  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;

  int ret = -1;
  std::array<char, 256> err_buf;

  const float center_x = 0.5;
  const float center_y = 0.5;

  // Get frame from decoder
  std::unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  for (uint64_t i = 0; i < frame_to_extract; i++) {
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
  }
  if (ret != 0) {
    std::cout << "Failed to get rgb frame" << std::endl;
    exit(EXIT_FAILURE);
  }

  int width = source_codec_ctx->width;
  int height = source_codec_ctx->height;

  std::unique_ptr<AVFrame, AVFrameDeleter> rect_frame(av_frame_alloc());
  rect_frame->width = REDUCED_BUFFER_WIDTH;
  rect_frame->height = REDUCED_BUFFER_HEIGHT;
  rect_frame->format = AV_PIX_FMT_RGB0;
  av_frame_get_buffer(rect_frame.get(), 1);

  // Copy the frame to the OpenCL GPU
  int source_frame_size = rgb_frame->linesize[0] * rgb_frame->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_ONLY,
                             source_frame_size);
  // Copy RGB frame to GPU
  ret = cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                 rgb_frame->data[0] + source_frame_size, cl_source_frame);

  std::memset(rgb_frame->data[0], 0,
              rgb_frame->height * rgb_frame->linesize[0]);

  int sat_buffer_size = width * height * 3 * sizeof(uint32_t);
  cl::Buffer cl_sat_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                           sat_buffer_size);

  // Encode SAT
  sat_encoder.EncodeFrameGPU(cl_sat_buffer(), cl_source_frame(),
                             rgb_frame->width, rgb_frame->height,
                             rgb_frame->linesize[0]);

  int cl_output_buffer_size = rect_frame->height * rect_frame->linesize[0];
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  sat_decoder.SampleFrameRectGPU(cl_output_buffer(), rect_frame->width,
                                 rect_frame->height, rect_frame->linesize[0],
                                 cl_sat_buffer(), source_codec_ctx, center_x,
                                 center_y);

  // Copy off of GPU
  ret =
      cl::copy(cl_manager.command_queue, cl_output_buffer, rect_frame->data[0],
               rect_frame->data[0] + cl_output_buffer_size);
  if (ret != CL_SUCCESS) {
    std::cerr << "Failed to copy rect frame off gpu" << std::endl;
    exit(EXIT_FAILURE);
  }

  sat_decoder.ExpandSampledFrameRectCPU(rgb_frame.get(), rect_frame.get(),
                                        center_x, center_y);
  SaveFramePNG(rgb_frame.get(), output_file);
  return 0;
}

int InterpolateSampledFrameRect(const std::vector<std::string> &args) {
  using std::unique_ptr;
  using std::filesystem::path;

  path source_video = "360_em_dataset/1080p_videos/03_drone_d5d4gnuAJLo.mp4";
  path output_file = "interpolated_satlogrectilinear";
  uint64_t frame_to_extract = 100;

  float center_x = 0.65f;
  float center_y = 0.75f;

  if (args.size() >= 4) {
    source_video = args[2];
    output_file = args[3];
  }

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  SATEncoder sat_encoder(&cl_manager);
  SATDecoder sat_decoder(&cl_manager);

  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;

  int ret = -1;

  // Get frame from decoder
  unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  for (uint64_t i = 0; i < frame_to_extract; i++) {
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
  }
  if (ret != 0) {
    std::cerr << "Failed to get rgb frame" << std::endl;
  }

  int width = source_codec_ctx->width;
  int height = source_codec_ctx->height;
  int reduced_width = 16 * std::ceil(width / 1.8 / 16);
  int reduced_height = 16 * std::ceil(height / 1.8 / 16);
  sat_decoder.InitializeGrid(reduced_width, reduced_height,
                             source_codec_ctx->width, source_codec_ctx->height);

  unique_ptr<AVFrame, AVFrameDeleter> rect_frame(av_frame_alloc());
  rect_frame->width = reduced_width;
  rect_frame->height = reduced_height;
  rect_frame->format = AV_PIX_FMT_RGB0;
  av_frame_get_buffer(rect_frame.get(), 0);

  // Copy the frame to the OpenCL GPU
  int source_frame_size = rgb_frame->linesize[0] * rgb_frame->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_ONLY,
                             source_frame_size);
  // Copy RGB frame to GPU
  ret = cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                 rgb_frame->data[0] + source_frame_size, cl_source_frame);

  int sat_buffer_size = width * height * 3 * sizeof(uint32_t);
  cl::Buffer cl_sat_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                           sat_buffer_size);

  // Encode SAT
  sat_encoder.EncodeFrameGPU(cl_sat_buffer(), cl_source_frame(),
                             rgb_frame->width, rgb_frame->height,
                             rgb_frame->linesize[0]);

  int cl_output_buffer_size = rect_frame->height * rect_frame->linesize[0];
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);
  sat_decoder.SampleFrameRectGPU(cl_output_buffer(), rect_frame->width,
                                 rect_frame->height, rect_frame->linesize[0],
                                 cl_sat_buffer(), source_codec_ctx, center_x,
                                 center_y);

  // Copy off of GPU
  ret =
      cl::copy(cl_manager.command_queue, cl_output_buffer, rect_frame->data[0],
               rect_frame->data[0] + cl_output_buffer_size);
  if (ret != CL_SUCCESS) {
    std::cout << "Failed to copy frame off of GPU" << std::endl;
    return EXIT_FAILURE;
  }

  sat_decoder.InterpolateFrameRectCPU(rgb_frame.get(), rect_frame.get(),
                                      center_x, center_y);
  SaveFramePNG(rgb_frame.get(), output_file);
  return EXIT_SUCCESS;
}

int EncodeVideo(const std::vector<std::string> &args) {
  using std::filesystem::path;
  path source_video = path("360_em_dataset") / "1080p_videos" /
                      "04_turtle_rescue_ncSdc12VzUg.mp4";
  path output_video = path("output_videos") / "turtle_sat_foveated_view.mp4";
  path gaze_file =
      path("360_em_dataset") / "reformatted_data" / "003" / "04.txt";
  auto length = std::chrono::hours(5);
  bool static_gaze_position = false;
  bool use_view_position = false;

  source_video = "input_videos/static_park.mp4";
  output_video = "output_videos/staticpark_sat_foveated_view.mp4";
  gaze_file = path("360_em_dataset") / "reformatted_data" / "003" / "01.txt";

  source_video = "360_em_dataset/1080p_videos/03_drone_d5d4gnuAJLo.mp4";
  output_video = "output_videos/drone_sat_view_foveated.mp4";
  gaze_file = path("360_em_dataset") / "reformatted_data" / "003" / "03.txt";
  use_view_position = true;

  source_video = "360_em_dataset/1080p_videos/13_drone_low_4m7NouQFaxc.mp4";
  output_video = "output_videos/13_drone_low_sat_view_foveated.mp4";
  gaze_file = path("360_em_dataset") / "reformatted_data" / "003" / "13.txt";
  use_view_position = true;

  if (args.size() == 4) {
    source_video = args[1];
    gaze_file = args[2];
    output_video = args[3];
  }

  // static_gaze_position = true;

  float center_x = 0.5;
  float center_y = 0.5;

  std::cout << "Foveating " << std::chrono::seconds(length).count()
            << " seconds of video: " << source_video.filename().string()
            << std::endl;
  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  SATEncoder sat_encoder(&cl_manager);
  SATDecoder sat_decoder(&cl_manager);
  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;
  VideoEncoder video_encoder(source_codec_ctx, NULL, output_video);
  sat_decoder.InitializeGrid(REDUCED_BUFFER_WIDTH, REDUCED_BUFFER_HEIGHT,
                             source_codec_ctx->width, source_codec_ctx->height);
  GazeViewPoints gv_points(gaze_file);

  int ret = -1;

  bool continue_loop = true;
  double elapsed_time;
  int frame = 0;

  std::unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  std::unique_ptr<AVFrame, AVFrameDeleter> output_frame(av_frame_alloc());
  output_frame->format = AV_PIX_FMT_RGB0;
  output_frame->width = REDUCED_BUFFER_WIDTH;
  output_frame->height = REDUCED_BUFFER_HEIGHT;
  av_frame_get_buffer(output_frame.get(), 1);

  int cl_source_frame_size = cl_source_frame_size =
      4 * source_codec_ctx->width * source_codec_ctx->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_WRITE,
                             cl_source_frame_size);
  int cl_sat_buffer_size =
      3 * source_codec_ctx->width * source_codec_ctx->height * sizeof(uint32_t);
  cl::Buffer cl_sat_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                           cl_sat_buffer_size);
  int cl_output_buffer_size = output_frame->linesize[0] * output_frame->height;
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  // Get frame from decoder
  while (continue_loop && frame < 30 * std::chrono::seconds(length).count()) {
    // if (frame > 3) exit(EXIT_SUCCESS);
    // for (int i = 0; i < 675; i++) {
    //   ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
    //   frame++;
    // }
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
    if (ret == 0) {
      if (frame % 30 == 0) {
        std::cout << "Processing frame " << frame << std::endl;
      }

      // Copy RGB frame to GPU
      ret =
          cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                   rgb_frame->data[0] + cl_source_frame_size, cl_source_frame);

      sat_encoder.EncodeFrameGPU(cl_sat_buffer(), cl_source_frame(),
                                 rgb_frame->width, rgb_frame->height,
                                 rgb_frame->linesize[0]);

      memset(rgb_frame->data[0], 0, rgb_frame->height * rgb_frame->linesize[0]);

      if (!static_gaze_position) {
        center_x = gv_points.points[frame].gaze_point[0];
        center_y = gv_points.points[frame].gaze_point[1];
      }
      if (use_view_position) {
        center_x = gv_points.points[frame].view_point[0];
        center_y = gv_points.points[frame].view_point[1];
      }

      sat_decoder.SampleFrameRectGPU(
          cl_output_buffer(), output_frame->width, output_frame->height,
          output_frame->linesize[0], cl_sat_buffer(),
          video_decoder.source_codec_ctx, center_x, center_y);
      // ret = cl::copy(
      //     cl_manager.command_queue, cl_output_buffer, output_frame->data[0],
      //     output_frame->data[0] + (output_frame->height *
      //     output_frame->linesize[0]));

      // SaveFramePNG(output_frame.get(), "test_output/buffer.png");

      sat_decoder.InterpolateFrameRectGPU(
          cl_source_frame(), rgb_frame->width, rgb_frame->height,
          rgb_frame->linesize[0], cl_output_buffer(), output_frame->width,
          output_frame->height, output_frame->linesize[0], center_x, center_y);
      ret = cl::copy(
          cl_manager.command_queue, cl_source_frame, rgb_frame->data[0],
          rgb_frame->data[0] + (rgb_frame->height * rgb_frame->linesize[0]));
      ret = video_encoder.EncodeFrameToFile(rgb_frame.get());
      // SaveFramePNG(rgb_frame.get(), "test_output/" + std::to_string(frame));
      frame++;
    } else {
      continue_loop = false;
    }
  }

  video_encoder.EncodeFrameToFile(NULL);
  video_encoder.WriteTrailerAndCloseFile();
  return EXIT_SUCCESS;
}

int EncodeLogCartesianVideo(const std::vector<std::string> &args) {
  std::string source_video =
      "360_em_dataset/1080p_videos/03_drone_d5d4gnuAJLo.mp4";
  std::string output_video = "drone_sat_foveated_logcartesian.mp4";
  std::string gaze_file = "360_em_dataset/reformatted_data/003/03.txt";
  auto length = std::chrono::hours(5);

  if (args.size() >= 4) {
    source_video = args[1];
    gaze_file = args[2];
    output_video = args[3];
    std::cout << "Using gaze" << gaze_file << std::endl;
  }

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  SATEncoder sat_encoder(&cl_manager);
  SATDecoder sat_decoder(&cl_manager);
  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;
  AVCodecContext output_codec_ctx = *source_codec_ctx;
  output_codec_ctx.width = REDUCED_BUFFER_WIDTH;
  output_codec_ctx.height = REDUCED_BUFFER_HEIGHT;
  VideoEncoder video_encoder(&output_codec_ctx, NULL, output_video);
  sat_decoder.InitializeGrid(REDUCED_BUFFER_WIDTH, REDUCED_BUFFER_HEIGHT,
                             source_codec_ctx->width, source_codec_ctx->height);

  GazeViewPoints gv_points(gaze_file);

  int ret = -1;
  bool continue_loop = true;
  int frame = 0;

  std::unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  std::unique_ptr<AVFrame, AVFrameDeleter> output_frame(av_frame_alloc());
  output_frame->format = AV_PIX_FMT_RGB0;
  output_frame->width = REDUCED_BUFFER_WIDTH;
  output_frame->height = REDUCED_BUFFER_HEIGHT;
  av_frame_get_buffer(output_frame.get(), 0);

  float center_x = 0.5f;
  float center_y = 0.5f;

  int cl_source_frame_size = cl_source_frame_size =
      4 * source_codec_ctx->width * source_codec_ctx->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_WRITE,
                             cl_source_frame_size);
  int cl_sat_buffer_size =
      3 * source_codec_ctx->width * source_codec_ctx->height * sizeof(uint32_t);
  cl::Buffer cl_sat_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                           cl_sat_buffer_size);
  int cl_output_buffer_size = output_frame->linesize[0] * output_frame->height;
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  // Get frame from decoder
  while (continue_loop && frame < 30 * std::chrono::seconds(length).count()) {
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
    if (ret == 0) {
      if (frame % 30 == 0) {
        std::cout << "Processing frame " << frame << std::endl;
      }

      // Copy RGB frame to GPU
      ret =
          cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                   rgb_frame->data[0] + cl_source_frame_size, cl_source_frame);

      sat_encoder.EncodeFrameGPU(cl_sat_buffer(), cl_source_frame(),
                                 rgb_frame->width, rgb_frame->height,
                                 rgb_frame->linesize[0]);

      memset(rgb_frame->data[0], 0, rgb_frame->height * rgb_frame->linesize[0]);

      center_x = gv_points.points[frame].gaze_point[0];
      center_y = gv_points.points[frame].gaze_point[1];

      sat_decoder.SampleFrameRectGPU(
          cl_output_buffer(), output_frame->width, output_frame->height,
          output_frame->linesize[0], cl_sat_buffer(),
          video_decoder.source_codec_ctx, center_x, center_y);
      ret = cl::copy(cl_manager.command_queue, cl_output_buffer,
                     output_frame->data[0],
                     output_frame->data[0] +
                         (output_frame->height * output_frame->linesize[0]));
      output_frame->pts = rgb_frame->pts;
      output_frame->pkt_dts = rgb_frame->pkt_dts;
      ret = video_encoder.EncodeFrameToFile(output_frame.get());

      frame++;
    } else {
      continue_loop = false;
    }
  }

  video_encoder.EncodeFrameToFile(NULL);
  video_encoder.WriteTrailerAndCloseFile();
  return EXIT_SUCCESS;
}

int EncodeLogCartesianVideoBitrate(const std::vector<std::string> &args) {
  using namespace std::filesystem;
  std::string source_video;
  // std::string output_video = "drone_sat_foveated_logcartesian.mp4";
  path output_video;
  std::string gaze_file;
  int bitrate = -1;
  auto length = std::chrono::hours(5);

  if (args.size() >= 6) {
    source_video = args[2];
    gaze_file = args[3];
    output_video = args[4];
    bitrate = std::stoi(args[5]);
  } else {
    exit(EXIT_FAILURE);
  }

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  SATEncoder sat_encoder(&cl_manager);
  SATDecoder sat_decoder(&cl_manager);
  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;
  AVCodecContext output_codec_ctx = *source_codec_ctx;
  output_codec_ctx.width = REDUCED_BUFFER_WIDTH;
  output_codec_ctx.height = REDUCED_BUFFER_HEIGHT;
  VideoEncoder video_encoder(&output_codec_ctx, NULL, output_video, bitrate);
  sat_decoder.InitializeGrid(REDUCED_BUFFER_WIDTH, REDUCED_BUFFER_HEIGHT,
                             source_codec_ctx->width, source_codec_ctx->height);

  GazeViewPoints gv_points(gaze_file);

  int ret = -1;
  bool continue_loop = true;
  int frame = 0;

  std::unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  std::unique_ptr<AVFrame, AVFrameDeleter> output_frame(av_frame_alloc());
  output_frame->format = AV_PIX_FMT_RGB0;
  output_frame->width = REDUCED_BUFFER_WIDTH;
  output_frame->height = REDUCED_BUFFER_HEIGHT;
  av_frame_get_buffer(output_frame.get(), 0);

  float center_x = 0.5f;
  float center_y = 0.5f;

  int cl_source_frame_size = cl_source_frame_size =
      4 * source_codec_ctx->width * source_codec_ctx->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_WRITE,
                             cl_source_frame_size);
  int cl_sat_buffer_size =
      3 * source_codec_ctx->width * source_codec_ctx->height * sizeof(uint32_t);
  cl::Buffer cl_sat_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                           cl_sat_buffer_size);
  int cl_output_buffer_size = output_frame->linesize[0] * output_frame->height;
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  // Get frame from decoder
  while (continue_loop && frame < 30 * std::chrono::seconds(length).count()) {
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
    if (ret == 0) {
      if (frame % 30 == 0) {
        std::cout << "Processing frame " << frame << std::endl;
      }

      // Copy RGB frame to GPU
      ret =
          cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                   rgb_frame->data[0] + cl_source_frame_size, cl_source_frame);

      sat_encoder.EncodeFrameGPU(cl_sat_buffer(), cl_source_frame(),
                                 rgb_frame->width, rgb_frame->height,
                                 rgb_frame->linesize[0]);

      memset(rgb_frame->data[0], 0, rgb_frame->height * rgb_frame->linesize[0]);

      center_x = gv_points.points[frame].gaze_point[0];
      center_y = gv_points.points[frame].gaze_point[1];

      sat_decoder.SampleFrameRectGPU(
          cl_output_buffer(), output_frame->width, output_frame->height,
          output_frame->linesize[0], cl_sat_buffer(),
          video_decoder.source_codec_ctx, center_x, center_y);
      ret = cl::copy(cl_manager.command_queue, cl_output_buffer,
                     output_frame->data[0],
                     output_frame->data[0] +
                         (output_frame->height * output_frame->linesize[0]));
      output_frame->pts = rgb_frame->pts;
      output_frame->pkt_dts = rgb_frame->pkt_dts;
      ret = video_encoder.EncodeFrameToFile(output_frame.get());

      frame++;
    } else {
      continue_loop = false;
    }
  }

  video_encoder.EncodeFrameToFile(NULL);
  video_encoder.WriteTrailerAndCloseFile();
  return EXIT_SUCCESS;
}

int DecodeLogCartesianVideo(const std::vector<std::string> &args) {
  fs::path source_video;
  fs::path output_video;
  fs::path gaze_file;
  auto length = std::chrono::hours(5);

  if (args.size() >= 5) {
    source_video = args[2];
    gaze_file = args[3];
    output_video = args[4];
  } else {
    return EXIT_FAILURE;
  }

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  SATEncoder sat_encoder(&cl_manager);
  SATDecoder sat_decoder(&cl_manager);
  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;
  AVCodecContext output_codec_ctx = *source_codec_ctx;
  output_codec_ctx.width = 1920;
  output_codec_ctx.height = 1080;
  VideoEncoder video_encoder(&output_codec_ctx, NULL, output_video);
  sat_decoder.InitializeGrid(REDUCED_BUFFER_WIDTH, REDUCED_BUFFER_HEIGHT,
                             source_codec_ctx->width, source_codec_ctx->height);

  GazeViewPoints gv_points(gaze_file);

  int ret = -1;
  bool continue_loop = true;
  int frame = 0;

  std::unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  std::unique_ptr<AVFrame, AVFrameDeleter> output_frame(av_frame_alloc());
  output_frame->format = AV_PIX_FMT_RGB0;
  output_frame->width = 1920;
  output_frame->height = 1080;
  av_frame_get_buffer(output_frame.get(), 0);

  float center_x = 0.5f;
  float center_y = 0.5f;

  int cl_source_frame_size = cl_source_frame_size =
      4 * source_codec_ctx->width * source_codec_ctx->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_WRITE,
                             cl_source_frame_size);
  int cl_output_buffer_size = output_frame->linesize[0] * output_frame->height;
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  // Get frame from decoder
  while (continue_loop && frame < 30 * std::chrono::seconds(length).count()) {
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
    if (ret == 0) {
      if (frame % 30 == 0) {
        std::cout << "Processing frame " << frame << std::endl;
      }

      // Copy RGB frame to GPU
      ret =
          cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                   rgb_frame->data[0] + cl_source_frame_size, cl_source_frame);

      center_x = gv_points.points[frame].gaze_point[0];
      center_y = gv_points.points[frame].gaze_point[1];

      sat_decoder.InterpolateFrameRectGPU(
          cl_output_buffer(), output_frame->width, output_frame->height,
          output_frame->linesize[0], cl_source_frame(), rgb_frame->width,
          rgb_frame->height, rgb_frame->linesize[0], center_x, center_y);

      ret = cl::copy(cl_manager.command_queue, cl_output_buffer,
                     output_frame->data[0],
                     output_frame->data[0] + cl_output_buffer_size);

      output_frame->pts = rgb_frame->pts;
      output_frame->pkt_dts = rgb_frame->pkt_dts;

      video_encoder.EncodeFrameToFile(output_frame.get());
      frame++;
    } else {
      continue_loop = false;
    }
  }

  video_encoder.EncodeFrameToFile(NULL);
  video_encoder.WriteTrailerAndCloseFile();
  return EXIT_SUCCESS;
}

int FoveateLogCartesianVideo(const std::vector<std::string> &args) {
  std::string source_video =
      "360_em_dataset/1080p_videos/03_drone_d5d4gnuAJLo.mp4";
  std::string gaze_file = "360_em_dataset/reformatted_data/003/03.txt";
  std::string output_video = "foveated_test.mp4";
  auto length = std::chrono::hours(5);

  if (args.size() >= 5) {
    source_video = args[2];
    gaze_file = args[3];
    output_video = args[4];
    std::cout << "Using gaze" << gaze_file << std::endl;
  } else {
    return EXIT_FAILURE;
  }

  OpenCLManager cl_manager;
  cl_manager.InitializeContext();
  VideoDecoder video_decoder;
  video_decoder.OpenVideo(source_video);
  SATEncoder sat_encoder(&cl_manager);
  SATDecoder sat_decoder(&cl_manager);
  AVCodecContext *source_codec_ctx = video_decoder.source_codec_ctx;
  AVCodecContext output_codec_ctx = *source_codec_ctx;
  VideoEncoder video_encoder(&output_codec_ctx, NULL, output_video);
  sat_decoder.InitializeGrid(REDUCED_BUFFER_WIDTH, REDUCED_BUFFER_HEIGHT,
                             source_codec_ctx->width, source_codec_ctx->height);

  GazeViewPoints gv_points(gaze_file);

  int ret = -1;
  bool continue_loop = true;
  int frame = 0;

  std::unique_ptr<AVFrame, AVFrameDeleter> rgb_frame(av_frame_alloc());
  std::unique_ptr<AVFrame, AVFrameDeleter> output_frame(av_frame_alloc());
  output_frame->format = AV_PIX_FMT_RGB0;
  output_frame->width = REDUCED_BUFFER_WIDTH;
  output_frame->height = REDUCED_BUFFER_HEIGHT;
  av_frame_get_buffer(output_frame.get(), 0);

  float center_x = 0.5f;
  float center_y = 0.5f;

  int cl_source_frame_size = cl_source_frame_size =
      4 * source_codec_ctx->width * source_codec_ctx->height;
  cl::Buffer cl_source_frame(cl_manager.context, CL_MEM_READ_WRITE,
                             cl_source_frame_size);
  int cl_sat_buffer_size =
      3 * source_codec_ctx->width * source_codec_ctx->height * sizeof(uint32_t);
  cl::Buffer cl_sat_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                           cl_sat_buffer_size);
  int cl_output_buffer_size = output_frame->linesize[0] * output_frame->height;
  cl::Buffer cl_output_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  // Get frame from decoder
  while (continue_loop && frame < 30 * std::chrono::seconds(length).count()) {
    ret = video_decoder.GetFrame(rgb_frame.get(), AV_PIX_FMT_RGB0);
    if (ret == 0) {
      if (frame % 30 == 0) {
        std::cout << "Processing frame " << frame << std::endl;
      }

      // Copy RGB frame to GPU
      ret =
          cl::copy(cl_manager.command_queue, rgb_frame->data[0],
                   rgb_frame->data[0] + cl_source_frame_size, cl_source_frame);

      sat_encoder.EncodeFrameGPU(cl_sat_buffer(), cl_source_frame(),
                                 rgb_frame->width, rgb_frame->height,
                                 rgb_frame->linesize[0]);

      memset(rgb_frame->data[0], 0, rgb_frame->height * rgb_frame->linesize[0]);

      center_x = gv_points.points[frame].gaze_point[0];
      center_y = gv_points.points[frame].gaze_point[1];

      sat_decoder.SampleFrameRectGPU(
          cl_output_buffer(), output_frame->width, output_frame->height,
          output_frame->linesize[0], cl_sat_buffer(),
          video_decoder.source_codec_ctx, center_x, center_y);

      sat_decoder.InterpolateFrameRectGPU(
          cl_source_frame(), rgb_frame->width, rgb_frame->height,
          rgb_frame->linesize[0], cl_output_buffer(), output_frame->width,
          output_frame->height, output_frame->linesize[0], center_x, center_y);

      ret = cl::copy(cl_manager.command_queue, cl_source_frame,
                     rgb_frame->data[0],
                     rgb_frame->data[0] + cl_source_frame_size);

      ret = video_encoder.EncodeFrameToFile(rgb_frame.get());

      frame++;
    } else {
      continue_loop = false;
    }
  }

  video_encoder.EncodeFrameToFile(NULL);
  video_encoder.WriteTrailerAndCloseFile();
  return EXIT_SUCCESS;
}