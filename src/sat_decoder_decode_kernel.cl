__kernel void decode_kernel(
    __global uchar *outputBuffer,
    int target_linesize,
    __global uint *sourceBuffer,
    int width,
    int height,
    int source_linesize)
{
    int x_pos = get_global_id(0);
    int y_pos = get_global_id(1);

    int target_bytes_per_pixel = target_linesize / width;

    int deltaX = 1;
    int deltaY = 1;

    int target_coord = y_pos * target_linesize + x_pos * target_bytes_per_pixel;
    uint min_value = 0;
    uint max_value = 255;

    if (x_pos > 0 && y_pos > 0)
    {
        int true_delta_x = min(x_pos, deltaX);
        int true_delta_y = min(y_pos, deltaY);
        int rectangleSize = true_delta_x * true_delta_y;
        int bottom_right_coord = y_pos * source_linesize + x_pos * 3;
        int top_right_coord = (y_pos - true_delta_y) * source_linesize + x_pos * 3;
        int bottom_left_coord = y_pos * source_linesize + (x_pos - true_delta_x) * 3;
        int top_left_coord = (y_pos - true_delta_y) * source_linesize + (x_pos - true_delta_x) * 3;
        outputBuffer[target_coord] = clamp((uint)(sourceBuffer[bottom_right_coord] - sourceBuffer[top_right_coord] + sourceBuffer[top_left_coord] - sourceBuffer[bottom_left_coord]) / rectangleSize, min_value, max_value);
        outputBuffer[target_coord + 1] = clamp((uint)(sourceBuffer[bottom_right_coord + 1] - sourceBuffer[top_right_coord + 1] + sourceBuffer[top_left_coord + 1] - sourceBuffer[bottom_left_coord + 1]) / rectangleSize, min_value, max_value);
        outputBuffer[target_coord + 2] = clamp((uint)(sourceBuffer[bottom_right_coord + 2] - sourceBuffer[top_right_coord + 2] + sourceBuffer[top_left_coord + 2] - sourceBuffer[bottom_left_coord + 2]) / rectangleSize, min_value, max_value);
    }
    else if (x_pos > 0)
    {
        int true_delta_x = min(x_pos, deltaX);
        int right_coord = y_pos * source_linesize + x_pos * 3;
        int left_coord = y_pos * source_linesize + (x_pos - true_delta_x) * 3;
        outputBuffer[target_coord] = clamp((uint)(sourceBuffer[right_coord] - sourceBuffer[left_coord]) / true_delta_x, min_value, max_value);
        outputBuffer[target_coord + 1] = clamp((uint)(sourceBuffer[right_coord + 1] - sourceBuffer[left_coord + 1]) / true_delta_x, min_value, max_value);
        outputBuffer[target_coord + 2] = clamp((uint)(sourceBuffer[right_coord + 2] - sourceBuffer[left_coord + 2]) / true_delta_x, min_value, max_value);
    }
    else if (y_pos > 0)
    {
        int true_delta_y = min(y_pos, deltaY);
        int bottom_coord = y_pos * source_linesize + x_pos * 3;
        int top_coord = (y_pos - true_delta_y) * source_linesize + x_pos * 3;
        outputBuffer[target_coord] = clamp((uint)(sourceBuffer[bottom_coord] - sourceBuffer[top_coord]) / true_delta_y, min_value, max_value);
        outputBuffer[target_coord + 1] = clamp((uint)(sourceBuffer[bottom_coord + 1] - sourceBuffer[top_coord + 1]) / true_delta_y, min_value, max_value);
        outputBuffer[target_coord + 2] = clamp((uint)(sourceBuffer[bottom_coord + 2] - sourceBuffer[top_coord + 2]) / true_delta_y, min_value, max_value);
    }
    else
    {
        outputBuffer[target_coord] = clamp((uint)sourceBuffer[0], min_value, max_value);
        outputBuffer[target_coord + 1] = clamp((uint)sourceBuffer[1], min_value, max_value);
        outputBuffer[target_coord + 2] = clamp((uint)sourceBuffer[2], min_value, max_value);
    }
}
