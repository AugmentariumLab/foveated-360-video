__kernel void sample_rect_kernel(
    __global uchar *output_buffer, int output_width, int output_height,
    int output_linesize, __global uchar *source_buffer, int source_width,
    int source_height, int source_linesize, __global short *grid_buffer,
    float center_x, float center_y) {
  int rect_buffer_width = output_width;
  int rect_buffer_height = output_height;

  int source_bytes_per_pixel = source_linesize / source_width;
  int output_bytes_per_pixel = output_linesize / output_width;

  int grid_bytes_per_pixel = 2;
  int grid_linesize = grid_bytes_per_pixel * output_width;

  float lambdaX = (float)source_width / (exp(1.0f) - 1);
  float lambdaY = (float)source_height / (exp(1.0f) - 1);

  for (int i = get_global_id(0); i < output_width; i += get_global_size(0)) {
    for (int j = get_global_id(1); j < output_height; j += get_global_size(1)) {
      int u = i - output_width / 2;
      int v = j - output_height / 2;

      int delta_x = grid_buffer[(j)*grid_linesize + (i)*grid_bytes_per_pixel];
      int delta_y =
          grid_buffer[(j)*grid_linesize + (i)*grid_bytes_per_pixel + 1];
      int x_pos = center_x * source_width + delta_x;
      int y_pos = center_y * source_height + delta_y;

      if (x_pos >= source_width) {
        x_pos -= source_width;
      } else if (x_pos < 0) {
        x_pos += source_width;
      }

      if (x_pos >= 0 && x_pos < source_width && y_pos >= 0 &&
          y_pos < source_height) {
        int target_coord = j * output_linesize + i * output_bytes_per_pixel;
        int bottom_right_coord =
            y_pos * source_linesize + x_pos * source_bytes_per_pixel;
        output_buffer[target_coord] = source_buffer[bottom_right_coord];
        output_buffer[target_coord + 1] = source_buffer[bottom_right_coord + 1];
        output_buffer[target_coord + 2] = source_buffer[bottom_right_coord + 2];
      }
    }
  }
}

__kernel void create_grid_kernel(__global short *grid_buffer, int output_width,
                                 int output_height, int source_width,
                                 int source_height) {
  int grid_width = output_width;
  int grid_height = output_height;
  int grid_bytes_per_pixel = 2;
  int grid_linesize = grid_width * grid_bytes_per_pixel;

  int thread_x = get_global_id(0);
  int thread_y = get_global_id(1);
  int total_threads_x = get_global_size(0);
  int total_threads_y = get_global_size(1);

  int i = thread_x;
  int j = thread_y;

  if (i >= output_width || j >= output_width) {
    return;
  }

  int u = i - output_width / 2;
  int v = j - output_height / 2;

  float lambdaX = (float)source_width / (exp(1.0f) - 1);
  float lambdaY = (float)source_height / (exp(1.0f) - 1);

  int delta_x =
      max((int)abs(u),
          (int)(lambdaX *
                (exp(pow((float)(2.0f * abs(u) / output_width), 4.0f)) - 1))) *
      ((u > 0) - (u < 0));
  int delta_y =
      max((int)abs(v),
          (int)(lambdaY *
                (exp(pow((float)(2.0f * abs(v) / output_height), 4.0f)) - 1))) *
      ((v > 0) - (v < 0));

  int target_pos = thread_y * grid_linesize + thread_x * grid_bytes_per_pixel;
  grid_buffer[target_pos] = delta_x;
  grid_buffer[target_pos + 1] = delta_y;
}
