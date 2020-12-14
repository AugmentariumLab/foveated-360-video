float sample_sat_value_from_svd(int x, int y, __global uchar *source_buffer,
                                __global float *sv_buffer,
                                __global float *u_buffer,
                                __global float *v_buffer, int source_width,
                                int source_height, int source_linesize,
                                int bytes_per_pixel, int sv_count, int color,
                                float range) {
  float final_value = 0;
  int sv_buffer_position = color * sv_count;
  int u_buffer_position = (color * source_height + y) * sv_count;
  int v_buffer_position = (color * sv_count) * source_width + x;
  for (int i = 0; i < sv_count; i++) {
    final_value += u_buffer[u_buffer_position + i] *
                   sv_buffer[sv_buffer_position + i] *
                   v_buffer[v_buffer_position + i * source_width];
  }
  final_value +=
      (source_buffer[y * source_linesize + x * bytes_per_pixel + color] *
           (range / 255.0f) -
       (range / 2.0f));
  return max(final_value, 0.0f);
}

// Sample from the reduced sat
__kernel void sample_rect_from_reduced_sat_kernel(
    __global uchar *output_buffer, int output_width, int output_height,
    int output_linesize, __global float *source_buffer) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  int input_bytes_per_pixel = 5;
  int input_linesize = input_bytes_per_pixel * (output_width + 1);

  int top_left_coord = j * input_linesize + i * input_bytes_per_pixel;
  int top_right_coord = top_left_coord + input_bytes_per_pixel;
  int bottom_left_coord = top_left_coord + input_linesize;
  int bottom_right_coord = bottom_left_coord + input_bytes_per_pixel;

  int output_bytes_per_pixel = output_linesize / output_width;
  int output_coordinate = j * output_linesize + i * output_bytes_per_pixel;

  float min_value = 0;
  float max_value = 255;

  int rect_x = source_buffer[bottom_right_coord + 3] -
               source_buffer[bottom_left_coord + 3];
  int rect_y = source_buffer[bottom_right_coord + 4] -
               source_buffer[top_right_coord + 4];

  for (int color = 0; color < 3; color++) {
    float top_left_value = source_buffer[top_left_coord + color] * (rect_x > 0);
    float top_right_value =
        source_buffer[top_right_coord + color] * (rect_y > 0);
    float bottom_left_value =
        source_buffer[bottom_left_coord + color] * (rect_x > 0 && rect_y > 0);
    float bottom_right_value =
        source_buffer[bottom_right_coord + color] * (rect_x > 0 || rect_y > 0);

    int rectangle_size = max(rect_x, 1) * max(rect_y, 1);
    output_buffer[output_coordinate + color] =
        (uchar)clamp((bottom_right_value - top_right_value + top_left_value -
                      bottom_left_value) /
                         rectangle_size,
                     min_value, max_value);
  }
  //   if (output_buffer[output_coordinate] > 250) {
  //     printf("Color is white at %d,%d\n", i, j);
  //     //   printf("rect size %d %d, %d\n\n", rectangle_size, rect_x, rect_y);
  //     //   printf("Values %f \n\n", (bottom_right_value - top_right_value +
  //     //                             top_left_value - bottom_left_value) /
  //     //                                rectangle_size);
  //     output_buffer[output_coordinate + 0] = 255;
  //     output_buffer[output_coordinate + 1] = 0;
  //     output_buffer[output_coordinate + 2] = 0;
  //   }
}

// Recreate a sat along the destination points from the SVD
__kernel void create_reduced_sat_kernel(
    __global float *output_buffer, int output_width, int output_height,
    __global uchar *source_buffer, int source_width, int source_height,
    int source_linesize, __global short *grid_buffer, float center_x,
    float center_y, __global float *u_buffer, __global float *v_buffer,
    __global float *sv_buffer, int sv_count, float3 delta_range) {
  int grid_width = output_width + 1;
  int grid_height = output_height + 1;
  int grid_bytes_per_pixel = 2;
  int grid_linesize = grid_bytes_per_pixel * grid_width;

  int output_bytes_per_pixel = 5;
  int output_linesize = output_bytes_per_pixel * grid_width;

  int source_bytes_per_pixel = source_linesize / source_width;

  int i = get_global_id(0);
  int j = get_global_id(1);

  if (i >= output_width || j >= output_height) {
    return;
  }

  int delta_x = grid_buffer[(j)*grid_linesize + (i)*grid_bytes_per_pixel];
  int delta_x_minus =
      grid_buffer[(j)*grid_linesize + (i - 1) * grid_bytes_per_pixel];
  int delta_y = grid_buffer[(j)*grid_linesize + (i)*grid_bytes_per_pixel + 1];
  int delta_y_minus =
      grid_buffer[(j - 1) * grid_linesize + (i)*grid_bytes_per_pixel + 1];
  int x_pos = center_x * source_width + delta_x;
  int x_pos_minus = center_x * source_width + delta_x_minus;
  int y_pos = center_y * source_height + delta_y;
  int y_pos_minus = center_y * source_height + delta_y_minus;

  if (((x_pos >= 0 && x_pos < source_width) ||
       (x_pos_minus >= 0 && x_pos_minus < source_width)) &&
      ((y_pos >= 0 && y_pos < source_height) ||
       (y_pos_minus >= 0 && y_pos_minus < source_height))) {
    x_pos = clamp(x_pos, 0, source_width - 1);
    y_pos = clamp(y_pos, 0, source_height - 1);

    int target_coord = j * output_linesize + i * output_bytes_per_pixel;
    output_buffer[target_coord] = sample_sat_value_from_svd(
        x_pos, y_pos, source_buffer, sv_buffer, u_buffer, v_buffer,
        source_width, source_height, source_linesize, source_bytes_per_pixel,
        sv_count, 0, delta_range.x);
    output_buffer[target_coord + 1] = sample_sat_value_from_svd(
        x_pos, y_pos, source_buffer, sv_buffer, u_buffer, v_buffer,
        source_width, source_height, source_linesize, source_bytes_per_pixel,
        sv_count, 1, delta_range.y);
    output_buffer[target_coord + 2] = sample_sat_value_from_svd(
        x_pos, y_pos, source_buffer, sv_buffer, u_buffer, v_buffer,
        source_width, source_height, source_linesize, source_bytes_per_pixel,
        sv_count, 2, delta_range.z);
    output_buffer[target_coord + 3] = x_pos;
    output_buffer[target_coord + 4] = y_pos;
  }
}

__kernel void sample_rect_kernel(__global uchar4 *output_buffer,
                                 int output_width, int output_height,
                                 int output_linesize,
                                 __global uint *source_buffer, int source_width,
                                 int source_height,
                                 __global short *grid_buffer, float2 center) {
  int rect_buffer_width = output_width;
  int rect_buffer_height = output_height;

  int grid_width = output_width + 1;
  int grid_height = output_height + 1;
  int grid_bytes_per_pixel = 2;
  int grid_linesize = grid_width * grid_bytes_per_pixel;


  int o_linesize = output_linesize / 4;
  int i_linesize = source_width;

  float lambdaX = (float)source_width / (exp(1.0f) - 1);
  float lambdaY = (float)source_height / (exp(1.0f) - 1);

  int i = get_global_id(0);
  int j = get_global_id(1);
  int u = i - output_width / 2;
  int v = j - output_height / 2;

  if (i >= output_width || j >= output_height) {
    return;
  }

  int delta_x =
      grid_buffer[(j + 1) * grid_linesize + (i + 1) * grid_bytes_per_pixel];
  int delta_x_minus =
      grid_buffer[(j + 1) * grid_linesize + (i)*grid_bytes_per_pixel];
  int delta_y =
      grid_buffer[(j + 1) * grid_linesize + (i + 1) * grid_bytes_per_pixel + 1];
  int delta_y_minus =
      grid_buffer[(j)*grid_linesize + (i + 1) * grid_bytes_per_pixel + 1];
  int2 pos = (int2)(center.x * source_width, center.y * source_height) +
             (int2)(delta_x, delta_y);
  int2 pos_minus =
      (int2)(center.x * source_width, center.y * source_height) + (int2)(delta_x_minus, delta_y_minus);

  if (pos.x >= source_width && pos_minus.x >= source_width) {
    pos.x -= source_width;
    pos_minus.x -= source_width;
  } else if (pos.x < 0 && pos_minus.x < 0) {
    pos.x += source_width;
    pos_minus.x += source_width;
  }

  //   if (y_pos >= source_height && pos_minus.y >= source_height) {
  //       y_pos -= source_height;
  //       pos_minus.y -= source_height;
  //   } else if (y_pos < 0 && pos_minus.y < 0) {
  //       y_pos += source_height;
  //       pos_minus.y += source_height;
  //   }

  if (((pos.x >= 0 && pos.x < source_width) ||
       (pos_minus.x >= 0 && pos_minus.x < source_width)) &&
      ((pos.y >= 0 && pos.y < source_height) ||
       (pos_minus.y >= 0 && pos_minus.y < source_height))) {
    pos.x = clamp(pos.x, 1, source_width - 1);
    pos.y = clamp(pos.y, 1, source_height - 1);
    pos_minus.x = clamp(pos_minus.x, 0, pos.x - 1);
    pos_minus.y = clamp(pos_minus.y, 0, pos.y - 1);
    int target_coord = j * o_linesize + i;
    if (pos.x > 0 && pos.y > 0) {
      int top_left_coord = pos_minus.y * i_linesize + pos_minus.x;
      int top_right_coord = pos_minus.y * i_linesize + pos.x;
      int bottom_left_coord = pos.y * i_linesize + pos_minus.x;
      int bottom_right_coord = pos.y * i_linesize + pos.x;
      int rectangle_size = (pos.x - pos_minus.x) * (pos.y - pos_minus.y);
      output_buffer[target_coord].xyz =
          convert_uchar3((vload3(bottom_right_coord, source_buffer) -
                          vload3(top_right_coord, source_buffer) +
                          vload3(top_left_coord, source_buffer) -
                          vload3(bottom_left_coord, source_buffer)) /
                         (uint3)(rectangle_size));
    } else if (pos.x > 0) {
      // pos.y is 0
      int right_coordinate = pos.x;
      int left_coordinate = pos_minus.x;
      int rectangle_size = (pos.x - pos_minus.x);
      output_buffer[target_coord].xyz =
          convert_uchar3((vload3(right_coordinate, source_buffer) -
                          vload3(left_coordinate, source_buffer)) /
                         (uint3)(rectangle_size));
    } else if (pos.y > 0) {
      // pos.x is 0
      int top_coordinate = pos_minus.y * i_linesize;
      int bottom_coordinate = pos.y * i_linesize;
      int rectangle_size = pos.y - pos_minus.y;
      output_buffer[target_coord].xyz =
          convert_uchar3((vload3(bottom_coordinate, source_buffer) -
                          vload3(top_coordinate, source_buffer)) /
                         (uint3)(rectangle_size));
    } else {
      output_buffer[target_coord].xyz =
          convert_uchar3(vload3(0, source_buffer));
    }
  }
}

__kernel void create_grid_kernel(__global short *grid_buffer, int output_width,
                                 int output_height, int source_width,
                                 int source_height) {
  int grid_width = output_width + 1;
  int grid_height = output_height + 1;
  int grid_bytes_per_pixel = 2;
  int grid_linesize = grid_width * grid_bytes_per_pixel;

  int thread_x = get_global_id(0);
  int thread_y = get_global_id(1);
  int total_threads_x = get_global_size(0);
  int total_threads_y = get_global_size(1);

  if (thread_x >= grid_width || thread_y >= grid_height) {
    return;
  }

  int i = thread_x - 1;
  int j = thread_y - 1;

  int u = i - output_width / 2;
  int v = j - output_height / 2;

  float lambdaX = (float)source_width / (exp(1.0f) - 1);
  float lambdaY = (float)source_height / (exp(1.0f) - 1);

  int delta_x =
      max((int)abs(u),
          (int)(lambdaX *
                (exp(pow((float)(2.0f * abs(u) / output_width), 4.0f)) - 1))) *
      ((u > 0) - (u < 0));
  int delta_x_plus =
      max((int)abs(u + 1),
          (int)(lambdaX *
                (exp(pow((float)(2.0f * abs(u + 1) / output_width), 4.0f)) -
                 1))) *
      ((u + 1 > 0) - (u + 1 < 0));
  int delta_y =
      max((int)abs(v),
          (int)(lambdaY *
                (exp(pow((float)(2.0f * abs(v) / output_height), 4.0f)) - 1))) *
      ((v > 0) - (v < 0));
  int delta_y_plus =
      max((int)abs(v + 1),
          (int)(lambdaY *
                (exp(pow((float)(2.0f * abs(v + 1) / output_height), 4.0f)) -
                 1))) *
      ((v + 1 > 0) - (v + 1 < 0));

  int target_pos = thread_y * grid_linesize + thread_x * grid_bytes_per_pixel;
  grid_buffer[target_pos] = floor((delta_x + delta_x_plus) / 2.0f);
  grid_buffer[target_pos + 1] = floor((delta_y + delta_y_plus) / 2.0f);
}

// This is similar to sample_rect_kernel.
__kernel void sample_rect_360_kernel(
    __global uchar4 *output_buffer, int output_width, int output_height,
    int output_linesize, __global uint *source_buffer, int source_width,
    int source_height, __global short2 *grid_buffer, float2 center) {
  int rect_buffer_width = output_width;
  int rect_buffer_height = output_height;

  int grid_width = output_width + 1;
  int grid_height = output_height + 1;
  int grid_bytes_per_pixel = 2;

  int o_linesize = output_linesize / 4;
  int i_linesize = source_width;

  float lambdaX = (float)source_width / (exp(1.0f) - 1);
  float lambdaY = (float)source_height / (exp(1.0f) - 1);

  int i = get_global_id(0);
  int j = get_global_id(1);
  int u = i - output_width / 2;
  int v = j - output_height / 2;

  if (i >= output_width || j >= output_height) {
    return;
  }

  short2 delta = grid_buffer[(j + 2) * grid_width + (i + 2)];
  short2 delta_minus = grid_buffer[(j + 2) * grid_width + (i - 1)];
  int2 pos = (int2)(center.x * source_width, center.y * source_height) + convert_int2(delta);
  int2 pos_minus =
      (int2)(center.x * source_width, center.y * source_height) + convert_int2(delta_minus);

  if (pos.x >= source_width && pos_minus.x >= source_width) {
    pos.x -= source_width;
    pos_minus.x -= source_width;
  } else if (pos.x < 0 && pos_minus.x < 0) {
    pos.x += source_width;
    pos_minus.x += source_width;
  }

  if (((pos.x >= 0 && pos.x < source_width) ||
       (pos_minus.x >= 0 && pos_minus.x < source_width)) &&
      ((pos.y >= 0 && pos.y < source_height) ||
       (pos_minus.y >= 0 && pos_minus.y < source_height))) {
    pos.x = clamp(pos.x, 1, source_width - 1);
    pos.y = clamp(pos.y, 1, source_height - 1);
    pos_minus.x = clamp(pos_minus.x, 0, pos.x - 1);
    pos_minus.y = clamp(pos_minus.y, 0, pos.y - 1);
    int target_coord = j * o_linesize + i;
    if (pos.x > 0 && pos.y > 0) {
      int top_left_coord = pos_minus.y * i_linesize + pos_minus.x;
      int top_right_coord = pos_minus.y * i_linesize + pos.x;
      int bottom_left_coord = pos.y * i_linesize + pos_minus.x;
      int bottom_right_coord = pos.y * i_linesize + pos.x;
      int rectangle_size = (pos.x - pos_minus.x) * (pos.y - pos_minus.y);
      output_buffer[target_coord].xyz =
          convert_uchar3((vload3(bottom_right_coord, source_buffer) -
                          vload3(top_right_coord, source_buffer) +
                          vload3(top_left_coord, source_buffer) -
                          vload3(bottom_left_coord, source_buffer)) /
                         (uint3)(rectangle_size));
    } else if (pos.x > 0) {
      // pos.y is 0
      int right_coordinate = pos.x;
      int left_coordinate = pos_minus.x;
      int rectangle_size = (pos.x - pos_minus.x);
      output_buffer[target_coord].xyz =
          convert_uchar3((vload3(right_coordinate, source_buffer) -
                          vload3(left_coordinate, source_buffer)) /
                         (uint3)(rectangle_size));
    } else if (pos.y > 0) {
      // pos.x is 0
      int top_coordinate = pos_minus.y * i_linesize;
      int bottom_coordinate = pos.y * i_linesize;
      int rectangle_size = pos.y - pos_minus.y;
      output_buffer[target_coord].xyz =
          convert_uchar3((vload3(bottom_coordinate, source_buffer) -
                          vload3(top_coordinate, source_buffer)) /
                         (uint3)(rectangle_size));
    } else {
      output_buffer[target_coord].xyz =
          convert_uchar3(vload3(0, source_buffer));
    }
  }
}