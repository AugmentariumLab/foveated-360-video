__kernel void copy_image_kernel(
	__global uint *output_buffer,
	int target_linesize,
    __global uchar *source_buffer,
    int source_width,
    int source_height,
    int source_linesize)
{
	int source_bytes_per_pixel = source_linesize / source_width;

	for (int y = get_global_id(1); y < source_height; y += get_global_size(1)) {
		for (int x = get_global_id(0); x < source_width; x += get_global_size(0)) {
			int in_position = y * source_linesize + x * source_bytes_per_pixel;
			int out_position = y * target_linesize + x * 3;
			output_buffer[out_position] = source_buffer[in_position];
			output_buffer[out_position + 1] = source_buffer[in_position + 1];
			output_buffer[out_position + 2] = source_buffer[in_position + 2];
		}
	}
}

__kernel void copy_image_back_kernel(__global uchar *output_buffer,
    __global uint *source_buffer,
    int width,
    int height,
    int linesize)
{
	int block = get_global_id(0);
	int total_blocks = get_global_size(0);
	int lines_per_block = (height + total_blocks - 1) / total_blocks;
	int thread = get_local_id(0);
	int total_threads = get_local_size(0);

	for (int y = block * lines_per_block; y < height && y < (block + 1) * lines_per_block; y++) {
		for (int x = thread; x < width; x += total_threads) {
			int out_position = y * linesize + x * 3;
			output_buffer[out_position] = source_buffer[out_position];
			output_buffer[out_position + 1] = source_buffer[out_position + 1];
			output_buffer[out_position + 2] = source_buffer[out_position + 2];
		}
	}
}

__kernel void scan_rows_kernel(__global uint* output_buffer, int width, int height, int source_line_size)
{
	for (int y = get_global_id(0); y < height; y += get_global_size(0)) {
		uint currentSum[3] = {0, 0, 0};
		for (int x = 0; x < width; x++) {
			int position = y * source_line_size + x * 3;
			currentSum[0] += output_buffer[position];
			currentSum[1] += output_buffer[position + 1];
			currentSum[2] += output_buffer[position + 2];
			output_buffer[position] = currentSum[0];
			output_buffer[position + 1] = currentSum[1];
			output_buffer[position + 2] = currentSum[2];
		}
	}
}

__kernel void scan_columns_kernel(__global uint* output_buffer, int width, int height, int source_line_size)
{
	for (int x = get_global_id(0); x < width; x += get_global_size(0)) {
		uint currentSum[3] = {0, 0, 0};
		for (int y = 0; y < height; y++) {
			int position = y * source_line_size + x * 3;
			currentSum[0] += output_buffer[position];
			currentSum[1] += output_buffer[position + 1];
			currentSum[2] += output_buffer[position + 2];
			output_buffer[position] = currentSum[0];
			output_buffer[position + 1] = currentSum[1];
			output_buffer[position + 2] = currentSum[2];
		}
	}
}
