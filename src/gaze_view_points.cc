#include "gaze_view_points.h"

GazeViewPoints::GazeViewPoints(std::string file_path) {
  std::ifstream file(file_path);
  const std::string float_regex = R"(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))";
  const std::regex xu_regex(
      R"(frame,(\d+),forward,([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))"
      R"(,eye,([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))");

  if (file.good()) {
    std::string line;
    while (std::getline(file, line)) {
      std::smatch xu_match;
      if (std::regex_search(line, xu_match, xu_regex)) {
        GazeViewPoint newPoint;
        newPoint.frame = std::stoul(xu_match.str(1));
        newPoint.view_point[0] = std::stof(xu_match.str(2));
        newPoint.view_point[1] = std::stof(xu_match.str(3));
        newPoint.gaze_point[0] = std::stof(xu_match.str(4));
        newPoint.gaze_point[1] = std::stof(xu_match.str(5));
        newPoint.pred_view_point[0] = newPoint.view_point[0];
        newPoint.pred_view_point[1] = newPoint.view_point[1];
        newPoint.pred_gaze_point[0] = newPoint.gaze_point[0];
        newPoint.pred_gaze_point[1] = newPoint.gaze_point[1];
        if (points.size() > 0) {
          GazeViewPoint oldPoint = points[points.size() - 1];
          newPoint.pred_view_point[0] = oldPoint.view_point[0];
          newPoint.pred_view_point[1] = oldPoint.view_point[1];
          newPoint.pred_gaze_point[0] = oldPoint.gaze_point[0];
          newPoint.pred_gaze_point[1] = oldPoint.gaze_point[1];
        }
        points.push_back(newPoint);
      }
    }
  } else {
    std::cerr << "Cannot open file: " << file_path << std::endl;
  }
}