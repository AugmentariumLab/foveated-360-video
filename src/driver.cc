
#include <iostream>
#include "parameters.h"
#include "video_server.h"

int main(int argc, char **argv) {
  int port = SERVER_PORT;
  if (argc > 1) {
    port = std::atoi(argv[1]);
  }
  VideoServer *server = new VideoServer();
  server->Run(port);
}