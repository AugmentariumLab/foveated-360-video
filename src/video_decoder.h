#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

class VideoDecoder {
 public:
  AVFormatContext *source_format_ctx = NULL;
  AVCodecContext *source_codec_ctx = NULL;
  AVStream *source_video_stream = NULL;
  int video_stream_idx = -1;
  bool av_format_opened = false;
  struct SwsContext *sws_ctx = NULL;
  VideoDecoder();
  ~VideoDecoder();
  void OpenVideo(std::string video_path);
  void OpenVideo(AVIOContext *);
  int GetFrame(AVFrame *target_frame, AVPixelFormat pixel_format);

 private:
  AVPixelFormat target_pixel_format = AV_PIX_FMT_NONE;
  AVFrame *source_frame = NULL;
  AVPacket source_packet;
  int OpenCodecContext(int *stream_idx, AVCodecContext **dec_ctx,
                       AVFormatContext *fmt_ctx, enum AVMediaType type);
};