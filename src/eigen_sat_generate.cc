#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <iostream>

int main() {
  using namespace std;
  using namespace std::chrono;
  using namespace Eigen;
  int height = 1080;
  int width = 1920;
  int sv_count = 30;

  system_clock::time_point start, stop;
  double elapsed_time;

  int u_buffer_size = 3 * height * sv_count * sizeof(float);
  float *u_buffer = (float *)malloc(u_buffer_size);
  int v_buffer_size = 3 * sv_count * width * sizeof(float);
  float *v_buffer = (float *)malloc(v_buffer_size);
  int sv_buffer_size = 3 * sv_count * sizeof(float);
  float *sv_buffer = (float *)malloc(sv_buffer_size);
  int svd_buffer_size = 3 * width * height * sizeof(float);
  float *svd_buffer = (float *)malloc(svd_buffer_size);

  start = high_resolution_clock::now();
  std::ifstream svd_file("SVD_metadata_10/1.bin", std::ios::binary);
  svd_file.read((char *)sv_buffer, sv_buffer_size);
  svd_file.read((char *)u_buffer, u_buffer_size);
  svd_file.read((char *)v_buffer, v_buffer_size);
  svd_file.close();
  stop = high_resolution_clock::now();
  elapsed_time = duration<double, std::milli>(stop - start).count();
  cout << "Time to read file: " << elapsed_time << endl;

  Map<Matrix<float, Dynamic, Dynamic, RowMajor>> eigen_u_buffer(
      u_buffer, 3 * height, sv_count);
  Map<Matrix<float, Dynamic, Dynamic, RowMajor>> eigen_v_buffer(
      v_buffer, 3 * sv_count, width);
  Map<Matrix<float, Dynamic, 1>> eigen_svd_buffer(svd_buffer, 3 * sv_count);
  start = high_resolution_clock::now();
  for (int color = 0; color < 3; color++) {
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>, 0, InnerStride<3>> eigen_svd(
        svd_buffer + color, height, width);
    eigen_svd =
        eigen_u_buffer.block(color * height, 0, height, sv_count) *
        eigen_svd_buffer.segment(color * sv_count, sv_count).asDiagonal() *
        eigen_v_buffer.block(color * sv_count, 0, sv_count, width);
  }
  stop = high_resolution_clock::now();
  elapsed_time = duration<double, std::milli>(stop - start).count();
  cout << "Time to recover SAT: " << elapsed_time << endl;

  free(u_buffer);
  free(v_buffer);
  free(sv_buffer);
  free(svd_buffer);
}