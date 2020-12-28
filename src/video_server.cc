#include "video_server.h"

VideoServer::VideoServer() {
  using namespace std;
  using websocketpp::lib::bind;
  using websocketpp::lib::placeholders::_1;
  using websocketpp::lib::placeholders::_2;

  m_server.init_asio();

  m_server.set_open_handler(bind(&VideoServer::on_open, this, _1));
  m_server.set_close_handler(bind(&VideoServer::on_close, this, _1));
  m_server.set_message_handler(bind(&VideoServer::on_message, this, _1, _2));
  m_server.set_access_channels(websocketpp::log::alevel::none);
  m_server.set_error_channels(websocketpp::log::elevel::all);
  m_next_sessionid = 1;
}

VideoServer::~VideoServer() {}

void VideoServer::on_open(websocketpp::connection_hdl hdl) {
  using namespace std;
  using json = nlohmann::json;
  std::cout << "Connection established" << std::endl;

  connection_data *data = new connection_data();
  data->sessionid = m_next_sessionid;
  m_connections[hdl] = data;

  json message;
  message["type"] = "text";
  message["message"] =
      string("Your connection id is ") + to_string(m_next_sessionid);
  message["packetNumber"] = 0;
  try {
    m_server.send(hdl, message.dump(), websocketpp::frame::opcode::text);
  } catch (websocketpp::exception const &e) {
    std::cout << "Websocket send failed: "
              << "(" << e.what() << ")" << std::endl;
  }

  m_next_sessionid++;
}

void VideoServer::InitializeConnectionData(websocketpp::connection_hdl hdl,
                                           connection_data *data,
                                           std::string video_request) {
  std::cout << "Client requested" << video_request << std::endl;
  if (data->thread != NULL) {
    std::cerr << "Connection already initialized" << std::endl;
    return;
  }
  std::string video_filename = "1080p_videos/" + video_request + ".mp4";
  {
    // Check for file existance.
    std::ifstream f(video_filename);
    if (!f.good()) {
      std::cerr << "Cannot find " << video_filename << std::endl;
      return;
    }
  }
  OpenCLManager *cl_manager = data->cl_manager = new OpenCLManager();
  cl_manager->InitializeContext();
  data->video_decoder = new VideoDecoder();
  data->sat_encoder = new SATEncoder(data->cl_manager);
  data->sat_decoder = new SATDecoder(data->cl_manager);

  data->video_decoder->OpenVideo(video_filename.c_str());
  AVCodecContext *source_codec_ctx = data->video_decoder->source_codec_ctx;
  AVCodecContext output_codec_ctx = *source_codec_ctx;
  output_codec_ctx.width = REDUCED_BUFFER_WIDTH;
  output_codec_ctx.height = REDUCED_BUFFER_HEIGHT;
  data->video_encoder = new VideoEncoder(&output_codec_ctx);

  data->rgb_frame = av_frame_alloc();
  data->output_frame = av_frame_alloc();
  data->output_frame->format = AV_PIX_FMT_RGB0;
  data->output_frame->width = REDUCED_BUFFER_WIDTH;
  data->output_frame->height = REDUCED_BUFFER_HEIGHT;
  av_frame_get_buffer(data->output_frame, 1);
  int attempts = 0;
  int ret = -1;
  int width = source_codec_ctx->width;
  int height = source_codec_ctx->height;
  data->thread = new std::thread(&VideoServer::SendFrameLoop, this, hdl, data);
}

void VideoServer::DestroyConnectionData(websocketpp::connection_hdl hdl) {
  connection_data *data = GetConnectionDataFromHdl(hdl);
  data->exit_thread = true;
  data->kill_thread_mutex.lock();
  delete data->video_decoder;
  delete data->sat_decoder;
  delete data->sat_encoder;
  delete data->video_encoder;
  delete data->cl_manager;
  av_frame_free(&data->rgb_frame);
  av_frame_free(&data->output_frame);
  delete data;
}

void VideoServer::on_message(websocketpp::connection_hdl hdl,
                             server::message_ptr msg) {
  using namespace std;
  using json = nlohmann::json;

  json received_arr = json::parse(msg->get_payload());

  if (received_arr["type"] == "text") {
    HandleTextMessage(hdl, received_arr);
  } else if (received_arr["type"] == "frameRequest") {
    new std::thread(&VideoServer::HandleFrameRequest, this, hdl, received_arr);
  } else if (received_arr["type"] == "videoRequest") {
    connection_data *data = GetConnectionDataFromHdl(hdl);
    InitializeConnectionData(hdl, data, received_arr["video"]);
  }
}

void VideoServer::on_close(websocketpp::connection_hdl hdl) {
  std::cout << "Client disconnected" << std::endl;
  DestroyConnectionData(hdl);
  m_connections.erase(hdl);
}

void VideoServer::Run(uint16_t port) {
  m_server.listen(port);
  std::cout << "Listening on port " << port << std::endl;
  m_server.start_accept();
  m_server.run();
}

VideoServer::connection_data *VideoServer::GetConnectionDataFromHdl(
    websocketpp::connection_hdl hdl) {
  auto it = m_connections.find(hdl);

  if (it == m_connections.end()) {
    // this connection is not in the list. This really shouldn't happen
    // and probably means something else is wrong.
    throw std::invalid_argument("No data available for session");
  }

  return it->second;
}

void VideoServer::HandleTextMessage(websocketpp::connection_hdl hdl,
                                    nlohmann::json received_arr) {
  using namespace std;
  using json = nlohmann::json;

  std::cout << "Message received: " << received_arr["message"].get<string>()
            << std::endl;

  json to_return;
  to_return["type"] = "text";
  to_return["message"] =
      "I got your message: " + received_arr["message"].get<string>();

  try {
    m_server.send(hdl, to_return.dump(), websocketpp::frame::opcode::text);
  } catch (websocketpp::exception const &e) {
    std::cout << "Websocket send failed: "
              << "(" << e.what() << ")" << std::endl;
  }
}

void VideoServer::HandleFrameRequest(websocketpp::connection_hdl hdl,
                                     nlohmann::json received_arr) {
  // std::cout << "-------------Handle frame request received--------------" <<
  // std::endl;
  connection_data *conn_data = GetConnectionDataFromHdl(hdl);
  conn_data->center_xy_mutex.lock();
  conn_data->center_x = received_arr["centerX"].get<double>();
  conn_data->center_y = received_arr["centerY"].get<double>();
  conn_data->center_xy_mutex.unlock();
  // Acknowledge that this information has been processed
  nlohmann::json to_return;
  to_return["type"] = "ack";
  to_return["packetNumber"] = received_arr["packetNumber"].get<int32_t>();
  try {
    m_server.send(hdl, to_return.dump(), websocketpp::frame::opcode::text);
  } catch (websocketpp::exception const &e) {
    std::cerr << "Websocket send failed: "
              << "(" << e.what() << ")" << std::endl;
  }
}

int VideoServer::WritePacket(void *opaque, uint8_t *buf, int buf_size) {
  IOOutput *out = reinterpret_cast<IOOutput *>(opaque);
  if (buf_size + out->bytesSet < out->maxSize) {
    memcpy(out->outBuffer + out->bytesSet, buf, buf_size);
    out->bytesSet += buf_size;
    return buf_size;
  }
  return 0;
}

void VideoServer::SendFrameLoop(websocketpp::connection_hdl hdl,
                                connection_data *conn_data) {
  using namespace std::chrono;
  using json = nlohmann::json;
  conn_data->kill_thread_mutex.lock();
  int ret = -1;
  char err_buf[256];
  int attempts = 0;
  // high_resolution_clock::time_point start, stop;
  // high_resolution_clock::time_point loop_start_time, loop_stop_time;
  high_resolution_clock::time_point checkpoint_time =
      high_resolution_clock::now();
  bool continue_loop = true;
  double elapsed_time;

  VideoDecoder *video_decoder = conn_data->video_decoder;
  OpenCLManager *cl_manager = conn_data->cl_manager;
  VideoEncoder *video_encoder = conn_data->video_encoder;
  SATDecoder *sat_decoder = conn_data->sat_decoder;
  SATEncoder *sat_encoder = conn_data->sat_encoder;
  AVFrame *output_frame = conn_data->output_frame;
  AVFrame *rgb_frame = conn_data->rgb_frame;

  int sv_count = 30;

  int height = video_decoder->source_codec_ctx->height;
  int width = video_decoder->source_codec_ctx->width;
  int cl_source_frame_size = 4 * width * height * sizeof(uint8_t);
  cl::Buffer cl_source_frame(cl_manager->context, CL_MEM_READ_WRITE,
                             cl_source_frame_size);
  int cl_sat_buffer_size = 3 * width * height * sizeof(uint32_t);
  cl::Buffer cl_sat_buffer(cl_manager->context, CL_MEM_READ_WRITE,
                           cl_sat_buffer_size);
  int cl_output_buffer_size = output_frame->linesize[0] * output_frame->height;
  cl::Buffer cl_output_buffer(cl_manager->context, CL_MEM_READ_WRITE,
                              cl_output_buffer_size);

  // Setup muxing variables to mux to fMP4
  AVOutputFormat *out_fmt = av_guess_format("mp4", NULL, NULL);
  AVFormatContext *out_fmt_ctx = NULL;
  ret = avformat_alloc_output_context2(&out_fmt_ctx, out_fmt, NULL, NULL);
  if (out_fmt_ctx == NULL || ret != 0) {
    std::cerr << "Failed to allocate output ctx2" << std::endl;
  }
  uint8_t *avio_ctx_buffer = NULL;
  size_t avio_ctx_buffer_size = 1000000;
  avio_ctx_buffer = (uint8_t *)av_malloc(avio_ctx_buffer_size);
  IOOutput buffer;
  buffer.bytesSet = 0;
  buffer.maxSize = 1000000;
  buffer.outBuffer = (uint8_t *)av_malloc(buffer.maxSize);

  AVIOContext *avio_ctx =
      avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 1, &buffer,
                         NULL, &WritePacket, NULL);
  if (avio_ctx == NULL) {
    std::cerr << "Failed to allocate avio ctx" << std::endl;
  }
  avio_ctx->seekable = false;
  out_fmt_ctx->pb = avio_ctx;
  AVDictionary *encode_opts = NULL;
  // Flags for fMP4
  av_dict_set(&encode_opts, "movflags",
              "frag_keyframe+empty_moov+default_base_moof", 0);

  AVStream *st = avformat_new_stream(out_fmt_ctx, video_encoder->video_codec);
  if (st == NULL) {
    std::cerr << "Failled to initialize new video stream" << std::endl;
  }
  st->id = (0);
  st->time_base = video_encoder->video_codec_ctx->time_base;
  avcodec_parameters_from_context(st->codecpar, video_encoder->video_codec_ctx);
  std::cerr << "stframerate: " << st->avg_frame_rate.num << "/"
            << st->avg_frame_rate.den << std::endl;
  st->duration = 0;

  ret = avformat_write_header(out_fmt_ctx, &encode_opts);
  if (ret < 0) {
    std::cerr << "Failed to write mp4 header" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cerr << "Header size: " << buffer.bytesSet << std::endl;
  m_server.send(hdl, buffer.outBuffer, buffer.bytesSet,
                websocketpp::frame::opcode::binary);
  buffer.bytesSet = 0;
  // Finished setting up muxing parameters to mux to fMP4.

  // Demux, decode, and convert from the video file into an RGB frame
  int frame_number = 0;
  int sent_frame_number = 0;
  while (continue_loop) {
    conn_data->wait_mutex.lock();
    attempts = 0;
    ret = -1;
    ret = video_decoder->GetFrame(rgb_frame, AV_PIX_FMT_RGB0);
    bool got_rgb_frame = (ret == 0);
    if (got_rgb_frame) {
      frame_number++;

      // Create Summed Area Table
      ret =
          cl::copy(cl_manager->command_queue, rgb_frame->data[0],
                   rgb_frame->data[0] + cl_source_frame_size, cl_source_frame);
      sat_encoder->EncodeFrameGPU(cl_sat_buffer(), cl_source_frame(), width,
                                  height, rgb_frame->linesize[0]);
      clFlush(cl_manager->command_queue());
      clFinish(cl_manager->command_queue());
    } else {
      // std::cout << "Failed to get rgb frame" << std::endl;
    }

    // Sleep until we're ready
    conn_data->wait_mutex.unlock();
    double time_since_checkpoint =
        duration<double, std::milli>(high_resolution_clock::now() -
                                     checkpoint_time)
            .count();
    double time_to_sleep = (1000.0 / 30.0) - time_since_checkpoint;
    if (time_to_sleep > 0) {
      std::this_thread::sleep_for(
          std::chrono::duration<double, std::milli>(time_to_sleep));
    }
    if (conn_data->exit_thread) {
      break;
    }

    // Done sleeping. Grab the latest gaze position.
    conn_data->wait_mutex.lock();
    conn_data->center_xy_mutex.lock();
    double center_x = conn_data->center_x;
    double center_y = conn_data->center_y;
    conn_data->center_xy_mutex.unlock();
    checkpoint_time = high_resolution_clock::now();

    // Sample from the summed area table based on the gaze position.
    std::vector<uint32_t> tt(3 * width * height);

    clFlush(cl_manager->command_queue());
    clFinish(cl_manager->command_queue());
    sat_decoder->SampleFrameRectGPU(
        cl_output_buffer(), output_frame->width, output_frame->height,
        output_frame->linesize[0], cl_sat_buffer(),
        video_decoder->source_codec_ctx, center_x, center_y);
    output_frame->pts = rgb_frame->pts;
    output_frame->pkt_dts = rgb_frame->pkt_dts;
    ret = cl::copy(cl_manager->command_queue, cl_output_buffer,
                   output_frame->data[0],
                   output_frame->data[0] +
                       (output_frame->height * output_frame->linesize[0]));
    if (ret != CL_SUCCESS) {
      std::cerr << "Failed to copy output frame out. " << ret << " "
                << OpenCLManager::GetCLErrorString(ret) << std::endl;
      exit(EXIT_FAILURE);
    }

    // This will be the new gaze position.
    frame_metadata new_metadata;
    new_metadata.center_x = center_x;
    new_metadata.center_y = center_y;
    conn_data->metadata_queue.push(new_metadata);

    AVPacket out_packet;
    av_init_packet(&out_packet);
    out_packet.size = 0;
    out_packet.data = NULL;

    attempts = 0;
    ret = video_encoder->EncodeFrame(&out_packet, output_frame);
    while ((ret < 0 || out_packet.size == 0) && attempts < 20) {
      // av_make_error_string(err_buf, 256, ret);
      // cout << "Failed to receive packet from encoder, trying again. " <<
      // err_buf
      //  << std::endl;
      // Sleep for 1ms
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      ret = video_encoder->GetPacket(&out_packet);
      attempts++;
    }
    if (ret < 0 || out_packet.size == 0) {
      std::cerr << "Final attempt to receive packet failed" << std::endl;
    } else {
      new_metadata = conn_data->metadata_queue.front();
      conn_data->metadata_queue.pop();
    }

    // std::cerr << "Out packet size: " << out_packet.size << std::endl;

    if (out_packet.size > 0) {
      out_packet.stream_index = 0;
      ret = av_write_frame(out_fmt_ctx, &out_packet);
      av_write_frame(out_fmt_ctx, nullptr);
      if (ret < 0) {
        av_make_error_string(err_buf, 256, ret);
        std::cerr << "Muxxing failed " << err_buf << std::endl;
      }
    } else {
      std::cerr << "Out pack size is 0, ret:" << ret << std::endl;
    }

    json to_return;
    to_return["type"] = "image";
    to_return["centerX"] = new_metadata.center_x;
    to_return["centerY"] = new_metadata.center_y;
    to_return["frameNum"] = sent_frame_number;
    sent_frame_number = (sent_frame_number + 1) % 256;
    std::string to_return_string = to_return.dump();
    try {
      m_server.send(hdl, to_return_string, websocketpp::frame::opcode::text);
      m_server.send(hdl, buffer.outBuffer, buffer.bytesSet,
                    websocketpp::frame::opcode::binary);
      buffer.bytesSet = 0;
    } catch (websocketpp::exception const &e) {
      std::cerr << "Websocket send failed: "
                << "(" << e.what() << ")" << std::endl;
    }
    av_packet_unref(&out_packet);
    std::memset(output_frame->data[0], 0,
                output_frame->height * output_frame->linesize[0]);
    conn_data->wait_mutex.unlock();
  }

  av_write_frame(out_fmt_ctx, nullptr);

  av_free(avio_ctx);
  av_dict_free(&encode_opts);
  av_free(avio_ctx_buffer);
  av_free(buffer.outBuffer);

  std::cerr << "Exiting Send Frame Loop" << std::endl;
  conn_data->kill_thread_mutex.unlock();
}
