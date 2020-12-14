#include "video_client.h"

using namespace std::chrono;
using json = nlohmann::json;

typedef websocketpp::client<websocketpp::config::asio_client> client;
typedef websocketpp::config::asio_client::message_type::ptr message_ptr;

const std::string VideoClient::vertex_shader =
    "#version 130\n"
    "in vec2 i_position;\n"
    "in vec2 uv;\n"
    "out vec2 vUv;\n"
    "uniform mat4 u_projection_matrix;\n"
    "void main() {\n"
    "    vUv = uv;\n"
    "    gl_Position = u_projection_matrix * vec4( i_position, 0.0, 1.0 );\n"
    "}\n";

const std::string VideoClient::fragment_shader =
    "#version 130\n"
    "in vec2 vUv;\n"
    "out vec4 o_color;\n"
    "uniform sampler2D tex;"
    "void main() {\n"
    "    o_color = vec4(texture(tex, vUv).xyz, 1.0);\n"
    "}\n";

VideoClient::VideoClient(std::string uri)
    : uri(uri),
      window(NULL),
      renderer(NULL),
      texture(NULL),
      frame(av_frame_alloc()),
      gaze_vec(2048) {
  io_buffer.bytesSet = 0;
  io_buffer.inBuffer.reserve(1000000);
  // av_log_set_level(AV_LOG_QUIET);
}

VideoClient::~VideoClient() {
  if (texture != NULL) {
    SDL_DestroyTexture(texture);
  }
  if (renderer != NULL) {
    SDL_DestroyRenderer(renderer);
  }
  if (window != NULL) {
    SDL_DestroyWindow(window);
  }
  SDL_Quit();
  if (frame != NULL) {
    av_frame_free(&frame);
    frame = NULL;
  }
  if (avio_ctx != NULL) {
    av_free(avio_ctx->buffer);
    av_free(avio_ctx);
    avio_ctx = NULL;
  }
}

void VideoClient::on_message(websocketpp::connection_hdl hdl, message_ptr msg) {
  if (msg->get_opcode() == websocketpp::frame::opcode::text) {
    json parsed = json::parse(msg->get_payload());
    if (parsed["type"] == "image") {
      last_received_pos = GazePos(parsed["centerX"], parsed["centerY"]);
      gaze_vec[parsed["frameNum"]] =
          GazePos(parsed["centerX"], parsed["centerY"]);
    } else if (parsed["type"] == "ack") {
    } else {
      std::cout << "Message received" << std::endl;
      std::cout << msg->get_payload() << std::endl;
    }
  } else {
    auto curr_time = high_resolution_clock::now();
    if (last_received_pos.x >= 0) {
      // gaze_vec[gaze_rec_pos + 1] = last_received_pos;
      gaze_rec_pos = (gaze_rec_pos + 1) % gaze_vec.size();
    }
    int64_t idx = GazeToIndex(last_received_pos);
    recv_time[idx] = curr_time;
    if (sent_time.find(idx) != sent_time.end()) {
      total_recv_time +=
          duration<double, std::milli>(curr_time - sent_time[idx]).count();
      total_recv_count++;
      sent_time.erase(idx);
    }
    // if (total_recv_count > 0) {
    //   std::cout << "Current avg time: " << total_recv_time / total_recv_count
    //             << std::endl;
    // }

    int ret = 0;
    auto payload = msg->get_payload();
    if (payload.size() + io_buffer.bytesSet > io_buffer.inBuffer.capacity()) {
      std::cerr << "[VideoClient::on_message] IO Buffer overcapacity"
                << std::endl;
      exit(EXIT_FAILURE);
    } else {
      std::memcpy(io_buffer.inBuffer.data() + io_buffer.bytesSet,
                  payload.data(), payload.size());
      io_buffer.bytesSet += payload.size();
      avio_ctx->eof_reached = false;
    }
    TryOpenInput();
    last_received_pos = GazePos(-1, -1);
  }
}

void VideoClient::on_open(websocketpp::connection_hdl hdl) {
  this->hdl = hdl;
  using json = nlohmann::json;
  std::cout << "Connection opened" << std::endl;
  json video_request;
  video_request["type"] = "videoRequest";
  video_request["video"] = "03_drone_d5d4gnuAJLo";
  try {
    ws_client.send(hdl, video_request.dump(), websocketpp::frame::opcode::text);
  } catch (websocketpp::exception const& e) {
    std::cerr << "![VideoClient::on_open]" << e.what() << std::endl;
  }
}

void VideoClient::UpdateGazePosition(float x, float y) {
  const float eps = std::pow(10, -5);
  if (std::abs(last_sent_pos.x - x) < eps &&
      std::abs(last_sent_pos.y - y) < eps) {
    return;
  }
  try {
    GazePos gp(x, y);
    int64_t idx = GazeToIndex(gp);
    sent_time[idx] = high_resolution_clock::now();
    json frame_request;
    frame_request["type"] = "frameRequest";
    frame_request["centerX"] = x;
    frame_request["centerY"] = y;
    frame_request["packetNumber"] = nextPacketNumber++;
    ws_client.send(hdl, frame_request.dump(), websocketpp::frame::opcode::text);
    last_sent_pos = {x, y};
  } catch (websocketpp::exception const& e) {
    std::cerr << "Failed to update gaze position" << std::endl;
    // std::cerr << "![VideoClient::connect]" << e.what() << std::endl;
  }
}

int VideoClient::ReadPacket(void* opaque, uint8_t* buf, int buf_size) {
  IOOutput* io_buffer = reinterpret_cast<IOOutput*>(opaque);
  buf_size = std::min(buf_size, io_buffer->bytesSet);
  if (buf_size > 0) {
    std::memcpy(buf, io_buffer->inBuffer.data(), buf_size);
    int64_t remaining_bytes = io_buffer->bytesSet - buf_size;
    if (remaining_bytes > 0) {
      std::memmove(io_buffer->inBuffer.data(),
                   io_buffer->inBuffer.data() + buf_size, remaining_bytes);
    }
    io_buffer->bytesSet = remaining_bytes;
  }
  if (buf_size == 0) {
    std::cerr << "[VideoClient::ReadPacket] Buf size 0" << std::endl;
    // exit(EXIT_FAILURE);
  }
  return buf_size;
}

int VideoClient::TryOpenInput() {
  // Do not try to open until we have enough bytes...
  if (io_buffer.bytesSet < 5000) {
    // std::cout << "Skipping a few times " << std::endl;
    return 0;
  }
  if (decoder.av_format_opened) {
    return 0;
  }
  if (avio_ctx == NULL) {
    return -1;
  }
  decoder.OpenVideo(avio_ctx);
  return 0;
}

void VideoClient::run() {
  using namespace std::filesystem;
  // Initialize SDL
  int ret;
  std::array<char, 256> errbuf;
  SDL_Event event;
  SetupSDL();

  path gaze_file = "360_em_dataset/reformatted_data/003/03.txt";
  GazeViewPoints gv_points(gaze_file);

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  auto last_time = high_resolution_clock::now();

  // Initialize OpenCL Stuff
  OpenCLManager cl_manager;
  cl_manager.gl_context = (cl_context_properties)glXGetCurrentContext();
  cl_manager.gl_display = (cl_context_properties)glXGetCurrentDisplay();
  cl_manager.InitializeContext();
  SATDecoder sat_decoder(&cl_manager);

  size_t avio_ctx_buffer_size = 1000000;
  uint8_t* avio_ctx_buffer = (uint8_t*)av_malloc(avio_ctx_buffer_size);
  avio_ctx = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 0,
                                &io_buffer, &ReadPacket, NULL, NULL);
  avio_ctx->seekable = false;

  connection_thread = std::thread(&VideoClient::connect, this);

  while (!decoder.av_format_opened) {
    duration<float, std::milli> time_passed(high_resolution_clock::now() -
                                            last_time);
    std::cerr << "Waiting for format to open" << std::endl;
    std::this_thread::sleep_for(duration<float, std::milli>(100));
    if (time_passed.count() > 1000 * 10) {
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "Format opened" << std::endl;

  int reduced_width = decoder.source_codec_ctx->width;
  int reduced_height = decoder.source_codec_ctx->height;
  std::cout << "Width, height: " << reduced_width << ", " << reduced_height
            << std::endl;
  std::cout << "Pixel format " << decoder.source_codec_ctx->pix_fmt
            << std::endl;
  // exit(EXIT_FAILURE);
  sat_decoder.InitializeGrid(reduced_width, reduced_height, full_width,
                             full_height);

  frame->format = AV_PIX_FMT_RGB0;
  frame->width = reduced_width;
  frame->height = reduced_height;
  av_frame_get_buffer(frame, 1);

  AVFrame* rgb_frame = av_frame_alloc();
  rgb_frame->format = AV_PIX_FMT_RGB0;
  rgb_frame->width = full_width;
  rgb_frame->height = full_height;
  av_frame_get_buffer(rgb_frame, 1);

  AVFrame* yuv_frame = av_frame_alloc();
  yuv_frame->format = AV_PIX_FMT_YUV420P;
  yuv_frame->width = full_width;
  yuv_frame->height = full_height;
  av_frame_get_buffer(yuv_frame, 1);

  cl::Buffer reduced_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                            frame->height * frame->linesize[0]);
  cl::Buffer rgb_buffer(cl_manager.context, CL_MEM_READ_WRITE,
                        rgb_frame->height * rgb_frame->linesize[0]);

  cl::ImageGL gl_mem(cl_manager.context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0,
                     gltexture, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "Failed to create mem: " << cl_manager.GetCLErrorString(ret)
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cerr << "Mem size " << ret << "," << gl_mem.getInfo<CL_MEM_SIZE>()
            << std::endl;

  SwsContext* sws_ctx = sws_getContext(
      full_width, full_height, (AVPixelFormat)rgb_frame->format, full_width,
      full_height, (AVPixelFormat)yuv_frame->format, 0, NULL, NULL, NULL);

  ret = 0;
  AVPacket packet;
  av_init_packet(&packet);
  bool exit_window = false;
  int frame_num = 0;
  while (!ws_client.stopped() && !exit_window) {
    glClear(GL_COLOR_BUFFER_BIT);
    last_time = high_resolution_clock::now();
    bool frame_available = io_buffer.bytesSet > 0;

    if (frame_available) {
      t1 = std::chrono::high_resolution_clock::now();
      ret = decoder.GetFrame(frame, (AVPixelFormat)frame->format);
      if (ret < 0) {
        std::cerr << "Failed to get frame" << std::endl;
        exit(EXIT_FAILURE);
      }
      t2 = std::chrono::high_resolution_clock::now();

      frame_num = (frame_num + 1) % 256;
      GazePos gp = gaze_vec[frame_num];

      int64_t idx = GazeToIndex(gp);
      if (recv_time.find(idx) != recv_time.end()) {
        auto prev = recv_time[idx];
        auto now = high_resolution_clock::now();
        total_decode_time += duration<float, std::milli>(now - prev).count();
        total_decode_count++;
        // std::cout << "Average decode time: "
        //           << (total_decode_time / total_decode_count) << std::endl;
      }

      auto decoded_time = high_resolution_clock::now();
      ret = cl::copy(cl_manager.command_queue, frame->data[0],
                     frame->data[0] + (frame->linesize[0] * frame->height),
                     reduced_buffer);
      if (ret != CL_SUCCESS) {
        std::cerr << "Cl copy failed" << std::endl;
        std::cerr << ret << " " << OpenCLManager::GetCLErrorString(ret)
                  << std::endl;
        exit(EXIT_FAILURE);
      }

      glFinish();
      clEnqueueAcquireGLObjects(cl_manager.command_queue(), 1, &gl_mem(), 0, 0,
                                NULL);
      sat_decoder.InterpolateFrameRectGPU(
          rgb_buffer(), full_width, full_height, rgb_frame->linesize[0],
          reduced_buffer(), reduced_width, reduced_height, frame->linesize[0],
          gp.x, gp.y);
      const size_t dst_origin[]{0, 0, 0};
      const size_t region[]{(size_t)full_width, (size_t)full_height, 1};
      ret = clEnqueueCopyBufferToImage(cl_manager.command_queue(), rgb_buffer(),
                                       gl_mem(), 0, dst_origin, region, 0, NULL,
                                       NULL);
      if (ret != CL_SUCCESS) {
        std::cerr << "Failure " << cl_manager.GetCLErrorString(ret)
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      cl_manager.command_queue.finish();
      clEnqueueReleaseGLObjects(cl_manager.command_queue(), 1, &gl_mem(), 0, 0,
                                NULL);
      auto unwarped_time = high_resolution_clock::now();
      total_unwarp_time +=
          duration<float, std::milli>(unwarped_time - decoded_time).count();
      total_unwarp_count++;
    }

    glEnable(GL_TEXTURE_2D);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glFlush();
    SDL_GL_SwapWindow(window);

    int mouse_x, mouse_y;
    SDL_GetMouseState(&mouse_x, &mouse_y);
    float mouse_xf = mouse_x / (float)full_width;
    float mouse_yf = mouse_y / (float)full_height;
    UpdateGazePosition(mouse_xf, mouse_yf);

    // Wait until we're ready.
    duration<double, std::milli> time_elapsed(high_resolution_clock::now() -
                                              last_time);
    if (time_elapsed.count() < MIN_LOOP_TIME) {
      SDL_Delay(MIN_LOOP_TIME - time_elapsed.count());
    }
    if (frame_available) {
      std::cout << "Loop time: " << time_elapsed.count() << std::endl;
    }

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        exit_window = true;
      }
    }
  }
  if (ws_client.stopped()) {
    std::cout << "client stopped" << std::endl;
  } else {
    ws_client.close(hdl, websocketpp::close::status::normal, "");
    // ws_client.stop();
    connection_thread.join();
  }

  std::cout << "Average receive time: "
            << (total_recv_time / std::max((uint64_t)1, total_recv_count))
            << " ms" << std::endl;
  std::cout << "Average decode time: "
            << (total_decode_time / std::max((uint64_t)1, total_decode_count))
            << " ms" << std::endl;
  std::cout << "Average unwarp time: "
            << (total_unwarp_time / std::max((uint64_t)1, total_unwarp_count))
            << " ms" << std::endl;

  CleanupSDL();
}

/**
 * @brief Connect to the websocket defined by url and run forever.
 * Run this in another thread if you want it to be nonblocking.
 *
 * @return int
 */
int VideoClient::connect() {
  using websocketpp::lib::bind;
  using websocketpp::lib::placeholders::_1;
  using websocketpp::lib::placeholders::_2;

  try {
    // Set logging to be pretty verbose (everything except message payloads)
    ws_client.set_access_channels(websocketpp::log::alevel::none);
    ws_client.clear_access_channels(websocketpp::log::alevel::frame_payload);

    // Initialize ASIO
    ws_client.init_asio();

    // Register our message handler
    ws_client.set_message_handler(bind(&VideoClient::on_message, this, _1, _2));
    ws_client.set_open_handler(bind(&VideoClient::on_open, this, _1));

    websocketpp::lib::error_code ec;
    websocketpp::client<websocketpp::config::asio_client>::connection_ptr con =
        ws_client.get_connection(uri, ec);
    if (ec) {
      std::cout << "could not create connection because: " << ec.message()
                << std::endl;
      exit(EXIT_FAILURE);
    }

    // Note that connect here only requests a connection. No network messages
    // are exchanged until the event loop starts running in the next line.
    ws_client.connect(con);

    // Start the ASIO io_service run loop
    // this will cause a single connection to be made to the server.
    // ws_client.run() will exit when this connection is closed.
    ws_client.run();
  } catch (websocketpp::exception const& e) {
    std::cerr << "![VideoClient::connect]" << e.what() << std::endl;
  }
  return EXIT_SUCCESS;
}

int64_t VideoClient::GazeToIndex(const VideoClient::GazePos& gp) {
  int x_num = std::clamp(gp.x, 0.0f, 1.0f) * 10000;
  int y_num = std::clamp(gp.y, 0.0f, 1.0f) * 10000;
  return x_num + y_num * 10000;
}

void VideoClient::SetupSDL() {
  SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
  SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
  SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  window = SDL_CreateWindow("Foveated", SDL_WINDOWPOS_UNDEFINED,
                            SDL_WINDOWPOS_UNDEFINED, full_width, full_height,
                            SDL_WINDOW_OPENGL);
  if (!window) {
    std::cerr << "Failed to create SDL Window" << std::endl;
    exit(EXIT_FAILURE);
  }

  glcontext = SDL_GL_CreateContext(window);
  if (glcontext == NULL) {
    printf("OpenGL context could not be created! SDL Error: %s\n",
           SDL_GetError());
    exit(EXIT_FAILURE);
  }

  GLenum err = glewInit();
  if (err != GLEW_OK) {
    std::cerr << "Failed to init glew: " << glewGetErrorString(err)
              << std::endl;
    exit(EXIT_FAILURE);
  }

  vs = glCreateShader(GL_VERTEX_SHADER);
  fs = glCreateShader(GL_FRAGMENT_SHADER);

  const char* vs_source = vertex_shader.c_str();
  const int vs_size = vertex_shader.size();
  glShaderSource(vs, 1, (const GLchar**)&vs_source, &vs_size);
  glCompileShader(vs);

  GLint status;
  glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
  if (status == GL_FALSE) {
    fprintf(stderr, "vertex shader compilation failed\n");
    exit(EXIT_FAILURE);
  }

  const char* fs_source = fragment_shader.c_str();
  const int fs_size = fragment_shader.size();
  glShaderSource(fs, 1, (const GLchar**)&fs_source, &fs_size);
  glCompileShader(fs);

  glGetShaderiv(fs, GL_COMPILE_STATUS, &status);
  if (status == GL_FALSE) {
    fprintf(stderr, "fragment shader compilation failed\n");
    exit(EXIT_FAILURE);
  }

  program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);

  typedef enum t_attrib_id { attrib_position, attrib_uv } t_attrib_id;

  glBindAttribLocation(program, attrib_position, "i_position");
  glBindAttribLocation(program, attrib_uv, "i_uv");
  glLinkProgram(program);

  glUseProgram(program);

  glDisable(GL_DEPTH_TEST);
  glClearColor(0.5, 0.0, 0.0, 0.0);
  glViewport(0, 0, full_width, full_height);

  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);

  glEnableVertexAttribArray(attrib_position);
  glEnableVertexAttribArray(attrib_uv);

  glVertexAttribPointer(attrib_uv, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, 0);
  glVertexAttribPointer(attrib_position, 2, GL_FLOAT, GL_FALSE,
                        sizeof(float) * 4, (void*)(2 * sizeof(float)));

  // clang-format off
  const GLfloat g_vertex_buffer_data[] = {
      /*  R, G, X, Y  */
      0, 0, 0, 0,
      1, 0, (GLfloat)full_width, 0, 
      1, 1, (GLfloat)full_width, (GLfloat)full_height,
      0, 0, 0, 0,
      1, 1, (GLfloat)full_width, (GLfloat)full_height,
      0, 1, 0, (GLfloat)full_height};
  // clang-format on

  glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data),
               g_vertex_buffer_data, GL_STATIC_DRAW);

  t_mat4x4 projection_matrix;
  mat4x4_ortho(projection_matrix, 0.0f, (float)full_width, (float)full_height,
               0.0f, 0.0f, 100.0f);
  glUniformMatrix4fv(glGetUniformLocation(program, "u_projection_matrix"), 1,
                     GL_FALSE, projection_matrix);

  glGenTextures(1, &gltexture);
  glBindTexture(GL_TEXTURE_2D, gltexture);
  glEnable(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLint)full_width,
               (GLint)full_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glUniform1i(glGetUniformLocation(program, "tex"), 0);
}

void VideoClient::mat4x4_ortho(t_mat4x4 out, float left, float right,
                               float bottom, float top, float znear,
                               float zfar) {
#define T(a, b) (a * 4 + b)

  out[T(0, 0)] = 2.0f / (right - left);
  out[T(0, 1)] = 0.0f;
  out[T(0, 2)] = 0.0f;
  out[T(0, 3)] = 0.0f;

  out[T(1, 1)] = 2.0f / (top - bottom);
  out[T(1, 0)] = 0.0f;
  out[T(1, 2)] = 0.0f;
  out[T(1, 3)] = 0.0f;

  out[T(2, 2)] = -2.0f / (zfar - znear);
  out[T(2, 0)] = 0.0f;
  out[T(2, 1)] = 0.0f;
  out[T(2, 3)] = 0.0f;

  out[T(3, 0)] = -(right + left) / (right - left);
  out[T(3, 1)] = -(top + bottom) / (top - bottom);
  out[T(3, 2)] = -(zfar + znear) / (zfar - znear);
  out[T(3, 3)] = 1.0f;

#undef T
}

void VideoClient::CleanupSDL() {
  SDL_GL_DeleteContext(glcontext);
  SDL_DestroyWindow(window);
  SDL_Quit();
}