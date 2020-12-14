#pragma once

#include <cpp-base64/base64.h>
#include <zlib.h>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "opencl_manager.h"
#include "parameters.h"
#include "sat_decoder.h"
#include "sat_encoder.h"
#include "save_frame.h"
#include "video_decoder.h"
#include "video_encoder.h"

class VideoServer {
 public:
  struct frame_metadata {
    float center_x;
    float center_y;
  };
  struct connection_data {
    int sessionid;
    int current_frame;

    OpenCLManager *cl_manager;
    VideoDecoder *video_decoder;
    VideoEncoder *video_encoder;
    SATEncoder *sat_encoder;
    SATDecoder *sat_decoder;
    AVFrame *rgb_frame;
    AVFrame *output_frame;
    float center_x = 0.0;
    float center_y = 0.0;
    bool exit_thread = false;

    std::thread *thread;
    std::mutex wait_mutex;
    std::mutex center_xy_mutex;
    // Do not kill the thread while this mutex is locked
    std::mutex kill_thread_mutex;
    std::queue<VideoServer::frame_metadata> metadata_queue;
  };

  VideoServer();
  ~VideoServer();
  typedef websocketpp::server<websocketpp::config::asio> server;
  void on_open(websocketpp::connection_hdl hdl);
  void on_message(websocketpp::connection_hdl, server::message_ptr msg);
  void on_close(websocketpp::connection_hdl hdl);
  void Run(uint16_t port);
  static int WritePacket(void *opaque, uint8_t *buffer, int buf_size);

 private:
 struct IOOutput {
    uint8_t* outBuffer;
    int bytesSet;
    int maxSize;
  };
  typedef std::map<websocketpp::connection_hdl, connection_data *,
                   std::owner_less<websocketpp::connection_hdl>>
      con_list;

  int m_next_sessionid;
  server m_server;
  con_list m_connections;
  connection_data *GetConnectionDataFromHdl(websocketpp::connection_hdl hdl);
  void HandleTextMessage(websocketpp::connection_hdl hdl,
                         nlohmann::json received_arr);
  void HandleFrameRequest(websocketpp::connection_hdl hdl,
                          nlohmann::json received_arr);
  void SendFrameLoop(websocketpp::connection_hdl hdl,
                     connection_data *conn_data);
  void InitializeConnectionData(websocketpp::connection_hdl hdl, connection_data *data, std::string video_request);
  void DestroyConnectionData(websocketpp::connection_hdl hdl);
};
