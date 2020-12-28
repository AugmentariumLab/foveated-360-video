#include <array>
#include <cstdio>
#include <iostream>
#include <queue>
#include <string>
#include <limits>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <libavutil/version.h>
#if LIBAVCODEC_VERSION_MAJOR <= 57
  #include "FFmpeg34/libavcodec/nvenc.h"
#else
  #include "FFmpeg42/libavcodec/nvenc.h"
#endif
}

class VideoEncoder {
 private:
  struct PtsDts {
    int64_t pts;
    int64_t dts;
    PtsDts(int64_t pts, int64_t dts) : pts(pts), dts(dts){};
  };

  std::string output_filename;
  AVBufferRef *hw_device_ctx;

  AVCodecContext *audio_codec_ctx;
  AVCodec *audio_codec;

  AVFrame *hw_frame;
  AVFormatContext *out_format_ctx;
  AVStream *out_video_stream;
  AVStream *out_audio_stream;
  AVRational input_timebase;
  int64_t last_dts = 0;
  std::queue<PtsDts> pts_queue;
  static int SetHWFrameCtx(AVCodecContext *ctx, AVBufferRef *hw_device_ctx);

 public:
  AVCodec *video_codec;
  AVCodecContext *video_codec_ctx;
  VideoEncoder(AVCodecContext *source_codec_ctx);
  VideoEncoder(AVCodecContext *video_codec_ctx, AVCodecContext *audio_codec_ctx,
               std::string filename);
  VideoEncoder(AVCodecContext *video_codec_ctx, AVCodecContext *audio_codec_ctx,
               std::string filename, int bitrate);
  ~VideoEncoder();
  int EncodeFrame(AVPacket *out_packet, AVFrame *source_frame);
  int EncodeFrameToFile(AVFrame *source_frame);
  int EncodeFrameToFile(AVFrame *source_frame, AVPacketSideData side_data);
  int GetPacket(AVPacket *out_packet);
  void WriteTrailerAndCloseFile();
  void PrintSupportedPixelFormats();
};