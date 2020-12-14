#include "video_decoder.h"

VideoDecoder::VideoDecoder() {
#if LIBAVUTIL_VERSION_MAJOR <= 55
  av_register_all();
#endif
  av_init_packet(&source_packet);
  source_packet.data = NULL;
  source_packet.size = 0;
}

VideoDecoder::~VideoDecoder() {
  if (source_frame != NULL) {
    av_frame_free(&source_frame);
  }
  av_packet_unref(&source_packet);
  if (source_codec_ctx != NULL) {
    avcodec_close(source_codec_ctx);
    avcodec_free_context(&source_codec_ctx);
  }
  if (source_format_ctx != NULL) {
    avformat_close_input(&source_format_ctx);
  }
  if (sws_ctx != NULL) {
    sws_freeContext(sws_ctx);
  }
}

/**
 * Opens the video with ffmpeg and finds the appropriate codecs
 */
void VideoDecoder::OpenVideo(std::string video_path) {
  int ret;
  std::array<char, 256> err_buf;

  // Open Video File
  if ((ret = avformat_open_input(&source_format_ctx, video_path.c_str(), NULL,
                                 NULL)) != 0) {
    av_strerror(ret, err_buf.data(), err_buf.size());
    std::cerr << "Could not open video file" << err_buf.data() << std::endl;
    return;
  }
  // Find the video stream
  if (avformat_find_stream_info(source_format_ctx, NULL) < 0) {
    std::cerr << "Could not find stream information" << std::endl;
    return;
  }

  if (OpenCodecContext(&video_stream_idx, &source_codec_ctx, source_format_ctx,
                       AVMEDIA_TYPE_VIDEO) >= 0) {
    source_video_stream = source_format_ctx->streams[video_stream_idx];
    source_codec_ctx->time_base = source_video_stream->time_base;
    source_codec_ctx->framerate = source_video_stream->r_frame_rate;
  }
  av_format_opened = true;
}

void VideoDecoder::OpenVideo(AVIOContext *avio_ctx) {
  int ret;
  std::array<char, 256> err_buf;
  source_format_ctx = avformat_alloc_context();
  if (source_format_ctx == NULL) {
    std::cerr << "Failed to allocate avformat ctx" << std::endl;
    exit(EXIT_FAILURE);
  }
  source_format_ctx->pb = avio_ctx;
  AVDictionary *options = NULL;
  av_dict_set(&options, "pixel_format", "yuv420", 0);
  ret = avformat_open_input(&source_format_ctx, NULL, NULL, &options);
  if (ret < 0) {
    std::cerr << "![VideoDecoder::TryOpenInput] Failed to open input"
              << std::endl;
    return;
  }

  // ret = avformat_find_stream_info(source_format_ctx, NULL);
  // if (ret < 0) {
  //   std::cerr << "Could not find stream information" << std::endl;
  //   return;
  // }

  ret = OpenCodecContext(&video_stream_idx, &source_codec_ctx,
                         source_format_ctx, AVMEDIA_TYPE_VIDEO);
  if (ret >= 0) {
    source_video_stream = source_format_ctx->streams[video_stream_idx];
    source_codec_ctx->time_base = source_video_stream->time_base;
    source_codec_ctx->framerate = source_video_stream->r_frame_rate;

    source_codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  } else {
    std::cerr << "OpenCodecContext returned " << ret << std::endl;
    exit(EXIT_FAILURE);
  }
  av_format_opened = true;
}

int VideoDecoder::OpenCodecContext(int *stream_idx, AVCodecContext **dec_ctx,
                                   AVFormatContext *fmt_ctx,
                                   enum AVMediaType type) {
  int refcount = 0;
  int ret, stream_index;
  AVStream *st = NULL;
  AVCodec *dec = NULL;
  AVDictionary *opts = NULL;
  if ((ret = av_find_best_stream(fmt_ctx, type, -1, -1, &dec, 0)) < 0) {
    std::cerr << "Could not find type " << av_get_media_type_string(type)
              << std::endl;
    return ret;
  }
  stream_index = ret;
  st = fmt_ctx->streams[stream_index];
  std::cout << "[VideoDecoder] Codec: "
            << avcodec_get_name(st->codecpar->codec_id) << std::endl;
  std::cout << "[VideoDecoder] Decoder: " << dec->long_name << std::endl;

  /* Allocate a codec context for the decoder */
  *dec_ctx = avcodec_alloc_context3(dec);
  if (!*dec_ctx) {
    fprintf(stderr, "Failed to allocate the %s codec context\n",
            av_get_media_type_string(type));
    return AVERROR(ENOMEM);
  }
  /* Copy codec parameters from input stream to output codec context */
  if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
    fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
            av_get_media_type_string(type));
    return ret;
  }
  av_opt_set_int(*dec_ctx, "refcounted_frames", 1, 0);

  /* Init the decoders, with or without reference counting */
  av_dict_set(&opts, "refcounted_frames", refcount ? "1" : "0", 0);
  if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
    fprintf(stderr, "Failed to open %s codec\n",
            av_get_media_type_string(type));
    return ret;
  }
  *stream_idx = stream_index;

Error:
  if (opts != NULL) {
    av_dict_free(&opts);
  }
  return 0;
}

int VideoDecoder::GetFrame(AVFrame *target_frame, AVPixelFormat pixel_format) {
  using std::cerr;
  using std::cout;
  using std::endl;
  int ret = -1;
  std::array<char, 256> err_buf;
  if (source_frame == NULL) {
    source_frame = av_frame_alloc();
  }
  if (target_frame->data[0] == NULL || target_frame->format != pixel_format) {
    av_frame_unref(target_frame);
    target_frame->format = pixel_format;
    target_frame->width = source_codec_ctx->width;
    target_frame->height = source_codec_ctx->height;
    if ((ret = av_frame_get_buffer(target_frame, 1)) < 0) {
      cerr << "Allocate frame buffer failed" << endl;
    }
  }
  if (sws_ctx == NULL || target_pixel_format != pixel_format) {
    sws_freeContext(sws_ctx);
    sws_ctx = sws_getContext(source_codec_ctx->width, source_codec_ctx->height,
                             source_codec_ctx->pix_fmt, source_codec_ctx->width,
                             source_codec_ctx->height, pixel_format,
                             SWS_BILINEAR, NULL, NULL, NULL);
    target_pixel_format = pixel_format;
  }
  int return_value = -100;
  bool read_another_frame = true;
  bool send_another_packet = true;

  while (read_another_frame) {
    ret = av_read_frame(source_format_ctx, &source_packet);
    // if (ret == 0) {
    //   std::cout << "RECEIVED: " << std::dec << source_packet.size << " ----";
    //   for (int i = 0; i < 10; i++) {
    //     std::cout << std::setw(2) << std::setfill('0') << std::hex
    //               << (int)(source_packet.buf->data[i]);
    //   }
    //   std::cout << std::endl;
    // } else {
    //   std::cout << "Error reading packet: " << std::endl;
    //   av_make_error_string(err_buf.data(), err_buf.size(), ret);
    //   std::cout << "Return " << ret << "; " << err_buf.data() << std::endl;
    // }
    if (ret == 0 && source_packet.stream_index == video_stream_idx) {
      read_another_frame = false;
      if (send_another_packet &&
          (ret = avcodec_send_packet(source_codec_ctx, &source_packet)) != 0) {
        if (ret == AVERROR_EOF) {
          send_another_packet = false;
          return_value = AVERROR_EOF;
        } else {
          av_make_error_string(err_buf.data(), err_buf.size(), ret);
          std::cerr << "[VideoDecoder::GetFrame] Avcodec send packet failed;"
                    << err_buf.data() << std::endl;
        }
      }
      int attempts = 0;
      ret = -1;
      while (ret != 0 && attempts < 10) {
        if ((ret = avcodec_receive_frame(source_codec_ctx, source_frame)) !=
            0) {
          av_make_error_string(err_buf.data(), err_buf.size(), ret);
          attempts++;
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          if (ret == AVERROR(EAGAIN)) {
            attempts = 999;
            read_another_frame = true;
          }
        }
      }

      if (ret == 0) {
        target_frame->pts = source_frame->pts;
        target_frame->pkt_dts = source_frame->pkt_dts;
        sws_scale(sws_ctx, source_frame->data, source_frame->linesize, 0,
                  source_frame->height, target_frame->data,
                  target_frame->linesize);
        return_value = 0;
      } else {
        if (return_value == -100) {
          return_value = ret;
        }
      }
    } else if (ret != 0) {
      read_another_frame = false;
    }
    av_packet_unref(&source_packet);
  }
  av_packet_unref(&source_packet);
  return return_value;
}