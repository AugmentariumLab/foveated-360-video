#include <iostream>
#include <string>
#include "video_client.h"

int main(int argc, char* argv[]) {
  std::vector<std::string> args(argv, argv + argc);
  std::string uri = "ws://localhost:9562";
  // uri = "ws://192.168.1.33:9562";
  if (args.size() >= 2) {
    uri = args[1];
  }
  VideoClient my_client(uri);
  my_client.run();
  return EXIT_SUCCESS;
}