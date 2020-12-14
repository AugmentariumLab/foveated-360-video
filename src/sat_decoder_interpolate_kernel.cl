__kernel void interpolate_rect_kernel(__global uchar3 *output_buffer,
                                      int output_width, int output_height,
                                      __global uchar3 *source_buffer,
                                      int source_width, int source_height,
                                      float2 center) {
  float center_x = center.x;
  float center_y = center.y;
  int rect_buffer_width = source_width;
  int rect_buffer_height = source_height;

  float lambdaX = output_width / (exp(1.0f) - 1);
  float lambdaY = output_height / (exp(1.0f) - 1);

  int x_pos = get_global_id(0);
  int y_pos = get_global_id(1);
  int target_coord = y_pos * output_width + x_pos;

  if (x_pos < 0 || x_pos >= output_width || y_pos < 0 ||
      y_pos >= output_height) {
    return;
  }

  // Step 1: Get the UV coordinates corresponding to this XY position)
  int center_x_pos = center_x * output_width;
  int center_y_pos = center_y * output_height;
  bool x_offset = false;
  if (x_pos - center_x_pos > output_width / 2) {
    x_pos -= output_width;
    x_offset = true;
  } else if (x_pos - center_x_pos < -output_width / 2) {
    x_pos += output_width;
    x_offset = true;
  }

  // x_pos = ((x_pos - center_x_pos + output_width / 2) % output_width) +
  //         (center_x_pos - output_width / 2);

  int delta_x = x_pos - center_x_pos;
  // if (abs(delta_x) > output_width / 2 + 5) {
  //   printf("ERROR %d\n", delta_x);
  // }
  int delta_y = y_pos - center_y_pos;
  int u = ceil(0.5 * rect_buffer_width *
               pow(log(abs(delta_x) / lambdaX + 1), 0.25f)) *
          ((delta_x > 0) - (delta_x < 0));
  int v = ceil(0.5 * rect_buffer_height *
               pow(log(abs(delta_y) / lambdaY + 1), 0.25f)) *
          ((delta_y > 0) - (delta_y < 0));

  if (abs(u) > abs(delta_x) || u == 0) {
    u = delta_x;
  }
  if (abs(v) > abs(delta_y) || v == 0) {
    v = delta_y;
  }
  int delta_x_calculated =
      max((int)abs(u),
          (int)(lambdaX *
                (exp(pow(2.0 * abs(u) / rect_buffer_width, 4.0)) - 1))) *
      ((u > 0) - (u < 0));
  int delta_y_calculated =
      max((int)abs(v),
          (int)(lambdaY *
                (exp(pow(2.0 * abs(v) / rect_buffer_height, 4.0)) - 1))) *
      ((v > 0) - (v < 0));

  if (delta_x_calculated == delta_x && delta_y_calculated == delta_y) {
    int source_coord =
        clamp(v + rect_buffer_height / 2, 0, source_height - 1) * source_width +
        clamp(u + rect_buffer_width / 2, 0, source_width - 1);

    output_buffer[target_coord] = source_buffer[source_coord];
  } else {
    // Bottom Right
    int delta_u = (x_pos < center_x_pos) - (x_pos > center_x_pos);
    int delta_v = (y_pos < center_y_pos) - (y_pos > center_y_pos);
    int delta_x_min =
        max((int)abs(u + delta_u),
            (int)(lambdaX *
                  (exp(pow(2.0f * abs(u + delta_u) / rect_buffer_width, 4.0f)) -
                   1))) *
        ((u > 0) - (u < 0));
    int delta_y_min =
        max((int)abs(v + delta_v),
            (int)(lambdaY *
                  (exp(pow(2.0f * abs(v + delta_v) / rect_buffer_height,
                           4.0f)) -
                   1))) *
        ((v > 0) - (v < 0));

    int min_x =
        min(center_x_pos + delta_x_min, center_x_pos + delta_x_calculated);
    int min_y =
        min(center_y_pos + delta_y_min, center_y_pos + delta_y_calculated);
    int max_x =
        max(center_x_pos + delta_x_min, center_x_pos + delta_x_calculated);
    int max_y =
        max(center_y_pos + delta_y_min, center_y_pos + delta_y_calculated);

    int min_u = min(u, u + delta_u);
    int min_v = min(v, v + delta_v);
    int max_u = max(u, u + delta_u);
    int max_v = max(v, v + delta_v);

    if (min_x < 0 && !x_offset) {
      min_u = max_u;
    }
    if (max_x >= output_width && !x_offset) {
      max_u = min_u;
    }
    if (min_y < 0) {
      min_v = max_v;
    }
    if (max_y >= output_height) {
      max_v = min_v;
    }

    int top_left_coord =
        clamp(min_v + rect_buffer_height / 2, 0, source_height - 1) *
            source_width +
        clamp(min_u + rect_buffer_width / 2, 0, source_width - 1);
    int top_right_coord =
        clamp(min_v + rect_buffer_height / 2, 0, source_height - 1) *
            source_width +
        clamp(max_u + rect_buffer_width / 2, 0, source_width - 1);
    int bottom_left_coord =
        clamp(max_v + rect_buffer_height / 2, 0, source_height - 1) *
            source_width +
        clamp(min_u + rect_buffer_width / 2, 0, source_width - 1);
    int bottom_right_coord =
        clamp(max_v + rect_buffer_height / 2, 0, source_height - 1) *
            source_width +
        clamp(max_u + rect_buffer_width / 2, 0, source_width - 1);

    float y_ratio = max_y == min_y
                        ? 0
                        : clamp((float)(y_pos - min_y) / (max_y - min_y),
                                (float)0, (float)1);
    float x_ratio = max_x == min_x
                        ? 0
                        : clamp((float)(x_pos - min_x) / (max_x - min_x),
                                (float)0, (float)1);
    float3 left_color =
        mix(convert_float3(source_buffer[top_left_coord]),
            convert_float3(source_buffer[bottom_left_coord]), y_ratio);
    float3 right_color =
        mix(convert_float3(source_buffer[top_right_coord]),
            convert_float3(source_buffer[bottom_right_coord]), y_ratio);
    output_buffer[target_coord] =
        convert_uchar3(mix(left_color, right_color, x_ratio));
  }
}