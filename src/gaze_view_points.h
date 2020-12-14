
#pragma once

#include <vector>
#include <string>
#include <regex>
#include <iostream>
#include <fstream>

class GazeViewPoints {
 public:
  struct GazeViewPoint {
    unsigned int frame = 0;
    float view_point[2];
    float gaze_point[2];
    float pred_view_point[2];
    float pred_gaze_point[2];
  };

  std::vector<GazeViewPoint> points;
  GazeViewPoints();
  GazeViewPoints(std::string file_path);
};