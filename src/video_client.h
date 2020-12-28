#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls_client.hpp>

// clang-format off
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_opengl_glext.h>
#include <SDL2/SDL_thread.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avassert.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#include "gaze_view_points.h"
#include "opencl_manager.h"
#include "parameters.h"
#include "sat_decoder.h"
#include "save_frame.h"
#include "video_decoder.h"
#include "video_encoder.h"
#include <CL/cl_gl.h>
// clang-format on

class VideoClient {
  typedef websocketpp::config::asio_client::message_type::ptr message_ptr;

 public:
  VideoClient(std::string url);
  ~VideoClient();
  void run();
  void on_message(websocketpp::connection_hdl hdl, message_ptr msg);
  void on_open(websocketpp::connection_hdl hdl);

 private:
  typedef float t_mat4x4[16];
  struct IOOutput {
    std::vector<uint8_t> inBuffer;
    int bytesSet;
  };
  struct GazePos {
    float x;
    float y;
    GazePos() : x(-1), y(-1){};
    GazePos(float x, float y) : x(x), y(y){};
  };
  const int MIN_LOOP_TIME = 5;
  std::string uri;
  static const std::string vertex_shader;
  static const std::string fragment_shader;

  int full_width = 1920;
  int full_height = 1080;

  double total_recv_time = 0;
  uint64_t total_recv_count = 0;
  double total_decode_time = 0;
  uint64_t total_decode_count = 0;
  double total_unwarp_time = 0;
  uint64_t total_unwarp_count = 0;

  GazePos last_sent_pos;
  GazePos last_received_pos;
  std::vector<GazePos> gaze_vec;
  int gaze_rec_pos = 0;
  uint64_t nextPacketNumber = 0;
  std::unordered_map<int64_t, std::chrono::high_resolution_clock::time_point>
      sent_time;
  std::unordered_map<int64_t, std::chrono::high_resolution_clock::time_point>
      recv_time;
  websocketpp::client<websocketpp::config::asio_client> ws_client;

  AVIOContext* avio_ctx = NULL;
  AVFrame* frame = NULL;
  SDL_Window* window = NULL;
  SDL_Renderer* renderer = NULL;
  SDL_Texture* texture = NULL;
  SDL_GLContext glcontext = NULL;
  GLuint vs, fs, program;
  GLuint vao, vbo;
  GLuint gltexture;
  VideoDecoder decoder;

  std::thread connection_thread;
  websocketpp::connection_hdl hdl;
  IOOutput io_buffer;

  int connect();
  void UpdateGazePosition(float x, float y);
  static int ReadPacket(void* opaque, uint8_t* buf, int buf_size);
  int TryOpenInput();
  static int64_t GazeToIndex(const GazePos& gp);
  void SetupSDL();
  void CleanupSDL();
  static void mat4x4_ortho(t_mat4x4 out, float left, float right, float bottom,
                           float top, float znear, float zfar);
};