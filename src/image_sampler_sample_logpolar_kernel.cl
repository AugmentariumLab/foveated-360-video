
#define _PI 3.14159265359
#define _ALPHA 4.0

__kernel void create_logpolar_grid_kernel(__global short *grid_buffer,
                                          int output_width, int output_height,
                                          int source_width, int source_height) {
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

  if (i >= output_width || j >= output_height) {
    return;
  }

  int u = i - output_width / 2;
  int v = j - output_height / 2;

  float lambdaX = (float)source_width / (exp(1.0f) - 1);
  float lambdaY = (float)source_height / (exp(1.0f) - 1);

  int delta_x = exp(10.0f * pow((float)i / output_width, (float)_ALPHA)) *
                cos((float)((float)j / output_height * 2.0f * _PI));
  int delta_y = exp(10.0f * pow((float)i / output_width, (float)_ALPHA)) *
                sin((float)((float)j / output_height * 2.0f * _PI));

  int target_pos = thread_y * grid_linesize + thread_x * grid_bytes_per_pixel;
  grid_buffer[target_pos] = delta_x;
  grid_buffer[target_pos + 1] = delta_y;
}

__kernel void sample_logpolar_kernel(
    __global uchar *output_buffer, int output_width, int output_height,
    int output_linesize, __global uchar *source_buffer, int source_width,
    int source_height, int source_linesize, __global short *grid_buffer,
    float center_x, float center_y) {
  int rect_buffer_width = output_width;
  int rect_buffer_height = output_height;

  int source_bytes_per_pixel = source_linesize / source_width;
  int output_bytes_per_pixel = output_linesize / output_width;

  int thread_x = get_global_id(0);
  int thread_y = get_global_id(1);
  int total_threads_x = get_global_size(0);
  int total_threads_y = get_global_size(1);

  int i = thread_x;
  int j = thread_y;

  if (i >= output_width || j >= output_height) {
    return;
  }

  int grid_bytes_per_pixel = 2;
  int grid_linesize = grid_bytes_per_pixel * output_width;

  int x_pos = center_x * source_width +
              grid_buffer[j * grid_linesize + i * grid_bytes_per_pixel];
  int y_pos = center_y * source_height +
              grid_buffer[j * grid_linesize + i * grid_bytes_per_pixel + 1];

  // x_pos = clamp(x_pos, 0, source_width - 1);
  x_pos = (x_pos + 10 * source_width) % source_width;
  y_pos = clamp(y_pos, 0, source_height - 1);

  if (x_pos >= 0 && x_pos < source_width && y_pos >= 0 && y_pos >= 0 &&
      y_pos < source_height) {
    int target_coord = j * output_linesize + i * output_bytes_per_pixel;
    int bottom_right_coord =
        y_pos * source_linesize + x_pos * source_bytes_per_pixel;

    output_buffer[target_coord] = source_buffer[bottom_right_coord];
    output_buffer[target_coord + 1] = source_buffer[bottom_right_coord + 1];
    output_buffer[target_coord + 2] = source_buffer[bottom_right_coord + 2];
  }
}

__kernel void logpolar_gaussian_blur_kernel(__global uchar3 *output_buffer,
                                            int output_width, int output_height,
                                            int output_linesize,
                                            __global uchar3 *source_buffer) {
  int rect_buffer_width = output_width;
  int rect_buffer_height = output_height;

  int output_bytes_per_pixel = output_linesize / output_width;

  int thread_x = get_global_id(0);
  int thread_y = get_global_id(1);
  int total_threads_x = get_global_size(0);
  int total_threads_y = get_global_size(1);

  int i = thread_x;
  int j = thread_y;

  if (i >= output_width || j >= output_height) {
    return;
  }

  int target_pixel = j * output_width + i;
  if (i >= output_width / 2) {
    const float PARA1 = 0.3377, PARA2 = 0.1217, PARA3 = 0.0439;
    int pixel_11 = max(j - 1, 0) * output_width + max(i - 1, 0);
    int pixel_12 = max(j - 1, 0) * output_width + i;
    int pixel_13 = max(j - 1, 0) * output_width + min(i + 1, output_width - 1);
    int pixel_21 = j * output_width + max(i - 1, 0);
    int pixel_22 = j * output_width + i;
    int pixel_23 = j * output_width + min(i + 1, output_width - 1);
    int pixel_31 = min(j + 1, output_height - 1) * output_width + max(i - 1, 0);
    int pixel_32 = min(j + 1, output_height - 1) * output_width + i;
    int pixel_33 = min(j + 1, output_height - 1) * output_width +
                   min(i + 1, output_width - 1);

    float3 frag_color11 = convert_float3(source_buffer[pixel_11]);
    float3 frag_color12 = convert_float3(source_buffer[pixel_12]);
    float3 frag_color13 = convert_float3(source_buffer[pixel_13]);
    float3 frag_color21 = convert_float3(source_buffer[pixel_21]);
    float3 frag_color22 = convert_float3(source_buffer[pixel_22]);
    float3 frag_color23 = convert_float3(source_buffer[pixel_23]);
    float3 frag_color31 = convert_float3(source_buffer[pixel_31]);
    float3 frag_color32 = convert_float3(source_buffer[pixel_32]);
    float3 frag_color33 = convert_float3(source_buffer[pixel_33]);
    
    float3 newColor =
        PARA3 * (frag_color11 + frag_color13 + frag_color31 + frag_color33) +
        PARA2 * (frag_color12 + frag_color21 + frag_color23 + frag_color32) +
        PARA1 * (frag_color22);
    output_buffer[target_pixel] = convert_uchar3(newColor);
  } else {
    output_buffer[target_pixel] = source_buffer[target_pixel];
    // output_buffer[target_pixel] = (uchar3)(255, 0, 0);
  }
}
