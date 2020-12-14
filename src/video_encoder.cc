#include "video_encoder.h"

VideoEncoder::VideoEncoder(AVCodecContext *s_video_codec_ctx) {
  using namespace std;
  int ret = 0;
  std::array<char, 256> errstr;

  hw_frame = av_frame_alloc();
  hw_device_ctx = NULL;
  out_format_ctx = NULL;
  audio_codec = NULL;
  audio_codec_ctx = NULL;

  if ((ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, NULL,
                                    NULL, 0)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to create hw context" << endl;
    av_make_error_string(errstr.data(), errstr.size(), ret);
    cerr << "Ret: " << ret << "," << errstr.data() << endl;
    return;
  }

  if (!(video_codec = avcodec_find_encoder_by_name("h264_nvenc"))) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to find h264_nvenc encoder"
         << endl;
    return;
  }
  video_codec_ctx = avcodec_alloc_context3(video_codec);
  video_codec_ctx->bit_rate = std::pow(10, 8);
  // std::cerr << "Max bitrate " << video_codec_ctx->bit_rate << std::endl;
  video_codec_ctx->width = s_video_codec_ctx->width;
  video_codec_ctx->height = s_video_codec_ctx->height;
  std::cerr << "Encoded resolution " << s_video_codec_ctx->width << ", "
            << s_video_codec_ctx->height << std::endl;
  video_codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  video_codec_ctx->time_base = s_video_codec_ctx->time_base;
  input_timebase = s_video_codec_ctx->time_base;
  video_codec_ctx->framerate = s_video_codec_ctx->framerate;
  video_codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;
  video_codec_ctx->profile = FF_PROFILE_H264_MAIN;
  video_codec_ctx->max_b_frames = 0;
  video_codec_ctx->delay = 0;
  ret = av_opt_set(video_codec_ctx->priv_data, "cq", "25", 0);
  if (ret != 0) {
    std::cerr << "Failed to set cq in priv data" << std::endl;
  }

  if ((ret = SetHWFrameCtx(video_codec_ctx, hw_device_ctx)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to set hwframe context."
         << endl;
    return;
  }

  AVDictionary *opts = NULL;
  av_dict_set(&opts, "preset", "fast", 0);
  if (avcodec_open2(video_codec_ctx, video_codec, &opts) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to open H264 codec" << endl;
    return;
  }

  if ((ret = av_hwframe_get_buffer(video_codec_ctx->hw_frames_ctx, hw_frame,
                                   0)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Av_hwframe_get_buffer failed" << endl;
    return;
  }

  NvencContext *nv = (NvencContext *)video_codec_ctx->priv_data;
  // auto r = nv->output_surface_ready_queue;
  // auto p = nv->output_surface_queue;
  // std::cerr << "Ready  "
  //           << (uint32_t)(r->wndx - r->rndx) / sizeof(NvencSurface *)
  //           << std::endl;
  // std::cerr << "Pending  "
  //           << (uint32_t)(p->wndx - p->rndx) / sizeof(NvencSurface *)
  //           << std::endl;
  // std::cerr << "rc_lookahead " << nv->rc_lookahead << std::endl;
  // std::cerr << "nb_surfaces " << nv->nb_surfaces << std::endl;
  nv->async_depth = 1;
}

VideoEncoder::VideoEncoder(AVCodecContext *s_video_codec_ctx,
                           AVCodecContext *s_audio_codec_ctx,
                           std::string filename) {
  using namespace std;
  int ret = 0;

  hw_frame = av_frame_alloc();
  hw_device_ctx = NULL;
  this->output_filename = filename;
  if ((ret = avformat_alloc_output_context2(
           &out_format_ctx, NULL, NULL, this->output_filename.c_str())) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] avformat_alloc_output_context2 failed"
         << endl;
    return;
  }
  if ((ret = avio_open(&out_format_ctx->pb, this->output_filename.c_str(),
                       AVIO_FLAG_WRITE)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] AVIO Open failed" << endl;
    return;
  }

  if ((ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, NULL,
                                    NULL, 0)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to create hw context" << endl;
    return;
  }

  if (!(video_codec = avcodec_find_encoder_by_name("h264_nvenc"))) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to find h264_nvenc encoder"
         << endl;
    return;
  }
  video_codec_ctx = avcodec_alloc_context3(video_codec);
  video_codec_ctx->bit_rate = std::pow(10, 8);
  video_codec_ctx->width = s_video_codec_ctx->width;
  video_codec_ctx->height = s_video_codec_ctx->height;
  // video_codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  video_codec_ctx->time_base = s_video_codec_ctx->time_base;
  input_timebase = s_video_codec_ctx->time_base;
  video_codec_ctx->framerate = s_video_codec_ctx->framerate;
  video_codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;
  video_codec_ctx->profile = FF_PROFILE_H264_MAIN;
  video_codec_ctx->max_b_frames = 0;
  video_codec_ctx->delay = 0;
  ret = av_opt_set(video_codec_ctx->priv_data, "cq", "25", 0);

  if ((ret = SetHWFrameCtx(video_codec_ctx, hw_device_ctx)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to set hwframe context."
         << endl;
    return;
  }

  AVDictionary *opts = NULL;
  av_dict_set(&opts, "preset", "fast", 0);
  if (avcodec_open2(video_codec_ctx, video_codec, &opts) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to open H264 codec" << endl;
    return;
  }

  if (!(out_video_stream = avformat_new_stream(out_format_ctx, video_codec))) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to allocate output stream"
         << endl;
  }
  out_video_stream->time_base = video_codec_ctx->time_base;
  out_video_stream->r_frame_rate = video_codec_ctx->framerate;
  if ((ret = avcodec_parameters_from_context(out_video_stream->codecpar,
                                             video_codec_ctx)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to copy parameters to context"
         << endl;
    return;
  }

  if (s_audio_codec_ctx) {
    out_audio_stream = avformat_new_stream(out_format_ctx, NULL);
    audio_codec = avcodec_find_encoder(s_audio_codec_ctx->codec_id);
    if (!audio_codec) {
      std::cerr << __func__ << " Audio encoder not found" << std::endl;
      return;
    }
    audio_codec_ctx = avcodec_alloc_context3(audio_codec);
    if (!audio_codec_ctx) {
      std::cerr << __func__ << "Failed to allocate the encoder context\n"
                << std::endl;
      return;
    }
    /* In this example, we transcode to same properties (picture size,
     * sample rate etc.). These properties can be changed for output
     * streams easily using filters */
    audio_codec_ctx->sample_rate = s_audio_codec_ctx->sample_rate;
    audio_codec_ctx->channel_layout = s_audio_codec_ctx->channel_layout;
    audio_codec_ctx->channels =
        av_get_channel_layout_nb_channels(s_audio_codec_ctx->channel_layout);
    /* take first format from list of supported formats */
    audio_codec_ctx->sample_fmt = audio_codec->sample_fmts[0];
    audio_codec_ctx->time_base = {1, audio_codec_ctx->sample_rate};
    if (out_format_ctx->oformat->flags & AVFMT_GLOBALHEADER)
      audio_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    /* Third parameter can be used to pass settings to encoder */
    ret = avcodec_open2(audio_codec_ctx, audio_codec, NULL);
    if (ret < 0) {
      std::cerr << __func__ << "Cannot open video encoder for audio stream"
                << std::endl;
      return;
    }
    ret = avcodec_parameters_from_context(out_audio_stream->codecpar,
                                          audio_codec_ctx);
    if (ret < 0) {
      std::cerr << __func__
                << "Failed to copy encoder parameters to audio stream"
                << std::endl;
      return;
    }
    out_audio_stream->time_base = audio_codec_ctx->time_base;
  } else {
    audio_codec_ctx = NULL;
    out_audio_stream = NULL;
  }
  if ((ret = av_hwframe_get_buffer(video_codec_ctx->hw_frames_ctx, hw_frame,
                                   0)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Av_hwframe_get_buffer failed" << endl;
    return;
  }

  if ((ret = avformat_write_header(out_format_ctx, NULL)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to copy parameters to context"
         << endl;
    return;
  }
}

VideoEncoder::VideoEncoder(AVCodecContext *s_video_codec_ctx,
                           AVCodecContext *s_audio_codec_ctx,
                           std::string filename, int bitrate) {
  using namespace std;
  int ret = 0;

  hw_frame = av_frame_alloc();
  hw_device_ctx = NULL;
  this->output_filename = filename;
  if ((ret = avformat_alloc_output_context2(
           &out_format_ctx, NULL, NULL, this->output_filename.c_str())) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] avformat_alloc_output_context2 failed"
         << endl;
    return;
  }
  if ((ret = avio_open(&out_format_ctx->pb, this->output_filename.c_str(),
                       AVIO_FLAG_WRITE)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] AVIO Open failed" << endl;
    return;
  }

  if ((ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, NULL,
                                    NULL, 0)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to create hw context" << endl;
    return;
  }

  if (!(video_codec = avcodec_find_encoder_by_name("h264_nvenc"))) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to find h264_nvenc encoder"
         << endl;
    return;
  }
  video_codec_ctx = avcodec_alloc_context3(video_codec);
  video_codec_ctx->width = s_video_codec_ctx->width;
  video_codec_ctx->height = s_video_codec_ctx->height;
  // video_codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  video_codec_ctx->time_base = s_video_codec_ctx->time_base;
  input_timebase = s_video_codec_ctx->time_base;
  video_codec_ctx->framerate = s_video_codec_ctx->framerate;
  video_codec_ctx->pix_fmt = AV_PIX_FMT_CUDA;
  video_codec_ctx->profile = FF_PROFILE_H264_MAIN;
  video_codec_ctx->max_b_frames = 0;
  video_codec_ctx->delay = 0;
  if (bitrate > 0) {
    video_codec_ctx->bit_rate = bitrate;
  } else {
    video_codec_ctx->bit_rate = std::pow(10, 8);
    ret = av_opt_set(video_codec_ctx->priv_data, "cq", "25", 0);
  }

  if ((ret = SetHWFrameCtx(video_codec_ctx, hw_device_ctx)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to set hwframe context."
         << endl;
    return;
  }

  AVDictionary *opts = NULL;
  av_dict_set(&opts, "preset", "fast", 0);
  if (avcodec_open2(video_codec_ctx, video_codec, &opts) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to open H264 codec" << endl;
    return;
  }

  if (!(out_video_stream = avformat_new_stream(out_format_ctx, video_codec))) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to allocate output stream"
         << endl;
  }
  out_video_stream->time_base = video_codec_ctx->time_base;
  out_video_stream->r_frame_rate = video_codec_ctx->framerate;
  if ((ret = avcodec_parameters_from_context(out_video_stream->codecpar,
                                             video_codec_ctx)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to copy parameters to context"
         << endl;
    return;
  }

  if (s_audio_codec_ctx) {
    out_audio_stream = avformat_new_stream(out_format_ctx, NULL);
    audio_codec = avcodec_find_encoder(s_audio_codec_ctx->codec_id);
    if (!audio_codec) {
      std::cerr << __func__ << " Audio encoder not found" << std::endl;
      return;
    }
    audio_codec_ctx = avcodec_alloc_context3(audio_codec);
    if (!audio_codec_ctx) {
      std::cerr << __func__ << "Failed to allocate the encoder context\n"
                << std::endl;
      return;
    }
    /* In this example, we transcode to same properties (picture size,
     * sample rate etc.). These properties can be changed for output
     * streams easily using filters */
    audio_codec_ctx->sample_rate = s_audio_codec_ctx->sample_rate;
    audio_codec_ctx->channel_layout = s_audio_codec_ctx->channel_layout;
    audio_codec_ctx->channels =
        av_get_channel_layout_nb_channels(s_audio_codec_ctx->channel_layout);
    /* take first format from list of supported formats */
    audio_codec_ctx->sample_fmt = audio_codec->sample_fmts[0];
    audio_codec_ctx->time_base = {1, audio_codec_ctx->sample_rate};
    if (out_format_ctx->oformat->flags & AVFMT_GLOBALHEADER)
      audio_codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    /* Third parameter can be used to pass settings to encoder */
    ret = avcodec_open2(audio_codec_ctx, audio_codec, NULL);
    if (ret < 0) {
      std::cerr << __func__ << "Cannot open video encoder for audio stream"
                << std::endl;
      return;
    }
    ret = avcodec_parameters_from_context(out_audio_stream->codecpar,
                                          audio_codec_ctx);
    if (ret < 0) {
      std::cerr << __func__
                << "Failed to copy encoder parameters to audio stream"
                << std::endl;
      return;
    }
    out_audio_stream->time_base = audio_codec_ctx->time_base;
  } else {
    audio_codec_ctx = NULL;
    out_audio_stream = NULL;
  }
  if ((ret = av_hwframe_get_buffer(video_codec_ctx->hw_frames_ctx, hw_frame,
                                   0)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Av_hwframe_get_buffer failed" << endl;
    return;
  }

  if ((ret = avformat_write_header(out_format_ctx, NULL)) < 0) {
    cerr << "[VideoEncoder::VideoEncoder] Failed to copy parameters to context"
         << endl;
    return;
  }
}

VideoEncoder::~VideoEncoder() {
  if (hw_frame != NULL) {
    av_frame_free(&hw_frame);
  }
  if (video_codec_ctx != NULL) {
    avcodec_close(video_codec_ctx);
    avcodec_free_context(&video_codec_ctx);
  }
  av_buffer_unref(&hw_device_ctx);
  if (out_format_ctx != NULL) {
    WriteTrailerAndCloseFile();
  }
}

int VideoEncoder::EncodeFrame(AVPacket *out_packet, AVFrame *source_frame) {
  int ret = 0;
  std::array<char, 256> err_buf;

  if (source_frame == NULL) {
    // std::cerr << __func__ << " Source frame is null" << std::endl;
    avcodec_send_frame(video_codec_ctx, NULL);
    ret = avcodec_receive_packet(video_codec_ctx, out_packet);
    if (ret == 0 && !pts_queue.empty()) {
      PtsDts pts = pts_queue.front();
      pts_queue.pop();
      out_packet->pts = pts.pts;
      out_packet->dts = pts.dts;
    }
    return ret;
  }

  AVFrame *yuv_frame = source_frame;
  if (!hw_frame->hw_frames_ctx) {
    std::cerr << "[VideoEncoder::EncodeFrame] Error no memory" << std::endl;
    goto Error;
  }
  if (yuv_frame->format != AV_PIX_FMT_YUV420P) {
    yuv_frame = av_frame_alloc();
    yuv_frame->width = source_frame->width;
    yuv_frame->height = source_frame->height;
    yuv_frame->format = AV_PIX_FMT_YUV420P;
    yuv_frame->pts = source_frame->pts;
    yuv_frame->pkt_dts = source_frame->pkt_dts;
    av_frame_get_buffer(yuv_frame, 0);
    SwsContext *to_yuv =
        sws_getContext(source_frame->width, source_frame->height,
                       (AVPixelFormat)source_frame->format, source_frame->width,
                       source_frame->height, AV_PIX_FMT_YUV420P, SWS_BILINEAR,
                       NULL, NULL, NULL);
    sws_scale(to_yuv, source_frame->data, source_frame->linesize, 0,
              source_frame->height, yuv_frame->data, yuv_frame->linesize);
    sws_freeContext(to_yuv);
  }

  if ((ret = av_hwframe_transfer_data(hw_frame, yuv_frame, 0)) < 0) {
    av_make_error_string(err_buf.data(), 256, ret);
    std::cerr
        << "[VideoEncoder::EncodeFrame] Error transfering frame from software "
           "to hardware; "
        << ret << "; " << err_buf.data() << std::endl;
    if (source_frame != NULL) {
      AVPixelFormat *format;
      const char *pix_fmt_name =
          av_get_pix_fmt_name((AVPixelFormat)source_frame->format);
      std::cout << "[VideoEncoder::EncodeFrame] Received format "
                << pix_fmt_name << ", " << source_frame->format << std::endl;
    } else {
      std::cerr << "[VideoEncoder::EncodeFrame] Source frame was null"
                << std::endl;
    }
    goto Error;
  }
  av_init_packet(out_packet);
  out_packet->data = NULL;
  out_packet->size = 0;
  pts_queue.emplace(yuv_frame->pts, yuv_frame->pkt_dts);
  if ((ret = avcodec_send_frame(video_codec_ctx, hw_frame)) < 0) {
    std::cerr << "[VideoEncoder::EncodeFrame] Error in send frame" << std::endl;
    goto Error;
  }
  if (yuv_frame != source_frame) {
    av_frame_free(&yuv_frame);
  }
  ret = avcodec_receive_packet(video_codec_ctx, out_packet);
  if (ret == 0 && !pts_queue.empty()) {
    PtsDts pts = pts_queue.front();
    pts_queue.pop();
    out_packet->pts = pts.pts;
    out_packet->dts = pts.dts;
  }
  return ret;
Error:
  return -1;
}

int VideoEncoder::EncodeFrameToFile(AVFrame *source_frame) {
  using namespace std;
  if (out_format_ctx == NULL) {
    cerr << "[VideoEncoder::EncodeFrameToFile] No file opened" << endl;
    return -1;
  }
  int ret = 0;
  char err_buf[256];
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = NULL;
  packet.size = 0;
  ret = EncodeFrame(&packet, source_frame);
  if (source_frame == NULL) {
    while (ret == 0) {
      packet.stream_index = 0;
      av_packet_rescale_ts(&packet, input_timebase,
                           out_video_stream->time_base);
      ret = av_interleaved_write_frame(out_format_ctx, &packet);
      if (ret < 0) {
        cerr
            << "[VideoEncoder::EncodeFrameToFile] Error writing to output file."
            << endl;
        return -1;
      }
      av_packet_unref(&packet);
      ret = GetPacket(&packet);
    }
    return 0;
  }
  if (ret == 0) {
    packet.stream_index = 0;
    av_packet_rescale_ts(&packet, input_timebase, out_video_stream->time_base);
    ret = av_interleaved_write_frame(out_format_ctx, &packet);
    if (ret < 0) {
      cerr << "[VideoEncoder::EncodeFrameToFile] Error writing to output file."
           << endl;
      return -1;
    }
  }
  return 0;
}

int VideoEncoder::EncodeFrameToFile(AVFrame *source_frame,
                                    AVPacketSideData side_data) {
  using namespace std;
  if (out_format_ctx == NULL) {
    cerr << "[VideoEncoder::EncodeFrameToFile] No file opened" << endl;
    return -1;
  }
  int ret = 0;
  char err_buf[256];
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = NULL;
  packet.size = 0;
  ret = EncodeFrame(&packet, source_frame);
  if (ret == 0) {
    packet.stream_index = 0;
    av_packet_rescale_ts(&packet, input_timebase, out_video_stream->time_base);
    cout << "Adding side data to packet: " << side_data.size << endl;
    ret = av_packet_add_side_data(&packet, side_data.type, side_data.data,
                                  side_data.size);
    if (ret != 0) {
      av_make_error_string(err_buf, 256, ret);
      cout << "Add side data error " << err_buf << endl;
    }
    cout << "Side_data_size: " << packet.side_data->size << endl;
    ret = av_interleaved_write_frame(out_format_ctx, &packet);
    if (ret < 0) {
      cerr << "[VideoEncoder::EncodeFrameToFile] Error writing to output file."
           << endl;
      return -1;
    }
  }
  return 0;
}

int VideoEncoder::GetPacket(AVPacket *out_packet) {
  using namespace std;
  int ret = -1;
  av_init_packet(out_packet);
  out_packet->data = NULL;
  out_packet->size = 0;
  ret = avcodec_receive_packet(video_codec_ctx, out_packet);
  // ret = video_codec_ctx->codec->receive_packet(video_codec_ctx, out_packet);
  if (ret == 0 && !pts_queue.empty()) {
    PtsDts pts = pts_queue.front();
    pts_queue.pop();
    out_packet->pts = pts.pts;
    out_packet->dts = pts.dts;
  }
  return ret;
}

int VideoEncoder::SetHWFrameCtx(AVCodecContext *ctx,
                                AVBufferRef *hw_device_ctx) {
  using namespace std;
  AVBufferRef *hw_frames_ref;
  AVHWFramesContext *frames_ctx = NULL;
  int err = 0;

  if (!(hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx))) {
    fprintf(stderr,
            "[VideoEncoder::SetHWFrameCtx] Failed to create hw "
            "frame context.\n");
    return -1;
  }
  frames_ctx = (AVHWFramesContext *)(hw_frames_ref->data);
  frames_ctx->format = AV_PIX_FMT_CUDA;
  frames_ctx->sw_format = AV_PIX_FMT_YUV420P;
  frames_ctx->width = ctx->width;
  frames_ctx->height = ctx->height;
  frames_ctx->initial_pool_size = 20;
  if ((err = av_hwframe_ctx_init(hw_frames_ref)) < 0) {
    cerr << "[VideoEncoder::SetHWFrameCtx] Failed to initialize hw frame "
            "context."
         << endl;
    av_buffer_unref(&hw_frames_ref);
    return err;
  }
  ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
  if (!ctx->hw_frames_ctx) err = AVERROR(ENOMEM);

  // av_buffer_unref(&hw_frames_ref);
  return err;
}

void VideoEncoder::WriteTrailerAndCloseFile() {
  using namespace std;
  if (out_format_ctx == NULL) {
    cerr << "[VideoEncoder::EncodeFrameToFile] No file opened" << endl;
  }
  av_write_trailer(out_format_ctx);
  avformat_close_input(&out_format_ctx);
}

void VideoEncoder::PrintSupportedPixelFormats() {
  using namespace std;
  AVPixelFormat *formats;
  cout << "Supported Formats: ";
  av_hwframe_transfer_get_formats(
      hw_frame->hw_frames_ctx, AV_HWFRAME_TRANSFER_DIRECTION_TO, &formats, 0);
  for (int i = 0; i < 99 && formats[i] != AV_PIX_FMT_NONE; i++) {
    const char *pix_fmt_name = av_get_pix_fmt_name(formats[i]);
    cout << pix_fmt_name;
    if (formats[i + 1] != AV_PIX_FMT_NONE) {
      cout << ", ";
    }
  }
  cout << endl;
  free(formats);
}
