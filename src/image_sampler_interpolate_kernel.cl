__kernel void interpolate_logpolar_kernel(__global uchar3 *output_buffer,
                                          int output_width, int output_height,
                                          __global uchar3 *source_buffer,
                                          int source_width, int source_height,
                                          float2 center) {
  float center_x = center.x;
  float center_y = center.y;

  float alpha = 1.0f;

  int x_pos = get_global_id(0);
  int y_pos = get_global_id(1);
  int target_coord = y_pos * output_width + x_pos;

  int rect_buffer_width = source_width;
  int rect_buffer_height = source_height;

  // Step 1: Get the UV coordinates corresponding to this XY position
  int center_x_pos = center_x * output_width;
  int center_y_pos = center_y * output_height;
  if (x_pos - center_x_pos > output_width / 2) {
    x_pos -= output_width;
  } else if (x_pos - center_x_pos < -output_width / 2) {
    x_pos += output_width;
  }
  int delta_x = x_pos - center_x_pos;
  int delta_y = y_pos - center_y_pos;
  float i_float =
      delta_x == 0 && delta_y == 0
          ? 0.0f
          : rect_buffer_width *
                pow(log(sqrt(pow(delta_x, 2.0f) + pow(delta_y, 2.0f))) / 10.0f,
                    (1.0f / alpha));
  int i = clamp((int)round(i_float), 0, rect_buffer_width - 1);
  float j_float = 0.0f;
  if (delta_x != 0) {
    j_float = (atan((float)delta_y / delta_x) + M_PI * (delta_x < 0)) *
              ((float)rect_buffer_height / (2.0 * M_PI));
    j_float = fmod(j_float + 2 * rect_buffer_height, source_height);
  } else {
    j_float =
        (M_PI_2 + M_PI * (delta_y < 0)) * (rect_buffer_height / (2.0 * M_PI));
  }
  int j = clamp((int)round(j_float), 0, rect_buffer_height - 1);

  int calculated_x_pos = center_x * output_width +
                         exp(10.0f * pow((float)i / source_width, alpha)) *
                             cos((float)j / source_height * 2.0f * M_PI);
  int calculated_y_pos = center_y * output_height +
                         exp(10.0f * pow((float)i / source_width, alpha)) *
                             sin((float)j / source_height * 2.0f * M_PI);

  if (calculated_x_pos == x_pos && calculated_y_pos == y_pos) {
    int source_coord = (j)*source_width + (i);
    output_buffer[target_coord] = source_buffer[source_coord];
  } else {
    // Bottom Right

    int min_i = clamp((int)floor(i_float), 0, source_width - 1);
    int min_j = (int)floor(j_float + source_height) % source_height;
    int max_i = clamp((int)ceil(i_float), 0, source_width - 1);
    int max_j = (int)ceil(j_float + source_height) % source_height;

    int top_left_coord = (min_j)*source_width + (min_i);
    int top_right_coord = (min_j)*source_width + (max_i);
    int bottom_left_coord = (max_j)*source_width + (min_i);
    int bottom_right_coord = (max_j)*source_width + (max_i);

    float i_ratio = i_float - floor(i_float);
    float j_ratio = j_float - floor(j_float);
    float3 left_color =
        mix(convert_float3(source_buffer[top_left_coord]),
            convert_float3(source_buffer[bottom_left_coord]), j_ratio);
    float3 right_color =
        mix(convert_float3(source_buffer[top_right_coord]),
            convert_float3(source_buffer[bottom_right_coord]), j_ratio);

    output_buffer[target_coord] =
        convert_uchar3(mix(left_color, right_color, i_ratio));
  }
}