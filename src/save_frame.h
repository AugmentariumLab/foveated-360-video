#pragma once
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

// Save an ffmpeg AVFrame as a PNG file
inline void SaveFramePNG(AVFrame *pFrame, std::string outputFilepath) {
  // Write file as PNG
  if (outputFilepath.compare(outputFilepath.length() - 4, 4, ".png") != 0) {
    outputFilepath = outputFilepath + ".png";
  }
  FILE *output_file;
  std::string frameFilename = outputFilepath;
  AVFrame *rgb_frame = av_frame_alloc();
  rgb_frame->format = AV_PIX_FMT_RGB24;
  rgb_frame->width = pFrame->width;
  rgb_frame->height = pFrame->height;
  av_frame_get_buffer(rgb_frame, 1);

  // Open File
  output_file = fopen(frameFilename.c_str(), "wb");
  if (output_file == NULL) return;
  struct SwsContext *sws_ctx = sws_getContext(
      pFrame->width, pFrame->height, (AVPixelFormat)pFrame->format,
      pFrame->width, pFrame->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL,
      NULL);
  sws_scale(sws_ctx, pFrame->data, pFrame->linesize, 0, pFrame->height,
            rgb_frame->data, rgb_frame->linesize);
  sws_freeContext(sws_ctx);

  // Allocate PNG codec
  AVCodec *out_codec = avcodec_find_encoder(AV_CODEC_ID_PNG);
  AVCodecContext *out_codec_ctx = avcodec_alloc_context3(out_codec);
  out_codec_ctx->width = pFrame->width;
  out_codec_ctx->height = pFrame->height;
  out_codec_ctx->pix_fmt = AV_PIX_FMT_RGB24;
  out_codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  out_codec_ctx->time_base.num = 1;
  out_codec_ctx->time_base.den = 25;
  if (avcodec_open2(out_codec_ctx, out_codec, NULL) < 0) {
    return;
  }

  AVPacket out_packet;
  av_init_packet(&out_packet);
  out_packet.size = 0;
  out_packet.data = NULL;

  avcodec_send_frame(out_codec_ctx, rgb_frame);
  avcodec_receive_packet(out_codec_ctx, &out_packet);

  avcodec_close(out_codec_ctx);
  avcodec_free_context(&out_codec_ctx);

  // Write PNG to disk and close the stream
  fwrite(out_packet.data, out_packet.size, 1, output_file);
  fclose(output_file);

  av_packet_unref(&out_packet);
  av_frame_free(&rgb_frame);
}

// Load a PNG file as an FFMpeg AVFrame
inline void LoadFramePNG(AVFrame *frame, std::string inputFilepath) {
#if LIBAVUTIL_VERSION_MAJOR <= 55
  av_register_all();
#endif
  // Read PNG file
  int ret = 0;
  std::vector<uint8_t> png_binary;
  {
    if (inputFilepath.compare(inputFilepath.size() - 4, 4, ".png") != 0) {
      inputFilepath = inputFilepath + ".png";
    }
    std::ifstream input(inputFilepath, std::ios::in | std::ios::binary);
    if (!input) {
      std::cerr << "Failed to open input: " << inputFilepath << std::endl;
      std::exit(EXIT_FAILURE);
    }
    input.seekg(0, std::ios::end);
    png_binary.resize(input.tellg());
    input.seekg(0, std::ios::beg);
    input.read((char *)png_binary.data(), png_binary.size());
    input.close();
  }

  AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_PNG);
  if (!codec) {
    fprintf(stderr, "PNG Codec not found\n");
    exit(1);
  }
  AVCodecContext *c = avcodec_alloc_context3(codec);
  if (!c) {
    fprintf(stderr, "Could not allocate video codec context\n");
    exit(1);
  }
  if (avcodec_open2(c, codec, NULL) < 0) {
    fprintf(stderr, "Could not open codec\n");
    exit(1);
  }

  AVPacket out_packet;
  av_init_packet(&out_packet);
  out_packet.size = 0;
  out_packet.data = NULL;

  out_packet.data = png_binary.data();
  out_packet.size = png_binary.size();

  ret = avcodec_send_packet(c, &out_packet);
  if (ret != 0) {
    std::cerr << "Failed to receive PNG frame" << std::endl;
    exit(EXIT_FAILURE);
  }

  frame->format = AV_PIX_FMT_RGB24;
  ret = avcodec_receive_frame(c, frame);
  if (ret != 0) {
    std::cerr << "Failed to receive PNG frame" << std::endl;
    exit(EXIT_FAILURE);
  }

  avcodec_close(c);
  avcodec_free_context(&c);

  av_packet_unref(&out_packet);
}