#define PI 3.141592653589793
#define PI_2 1.5707963267948966
#define DEG2RAD 0.017453292519943295
#define RAD2DEG 57.29577951308232

// Inverse gnomonic formula
__kernel void gnomonic_kernel(__global uchar3 *target_buffer, int target_width,
                              int target_height, __global uchar3 *source_buffer,
                              int source_width, int source_height,
                              float2 center) {
  // Center is the center of the viewport in [0,1] x [0,1]
  int i = get_global_id(0);
  int j = get_global_id(1);

  if (i >= target_width || j >= target_height) {
    return;
  }
  // Note you will need to calculate this if you wish to know exact
  // degrees for fov
  float2 scale = (float2)(6, 3);

  float2 uv = (float2)((float)i / target_width, (float)j / target_height);
  float2 convertedScreenCoord = scale * (uv - (float2)(0.5));
  float x = convertedScreenCoord.x, y = convertedScreenCoord.y;

  //[-pi/2, pi/2]
  float phi1 = (center.y - 0.5) * PI;
  //[-pi, pi]
  float lambda0 = (center.x - 0.5) * 2.0 * PI;
  float rho = sqrt(x * x + y * y);
  float c = atan(rho);
  float phi = asin(cos(c) * sin(phi1) + (y * sin(c) * cos(phi1)) / rho);
  float lambda =
      lambda0 +
      atan2(x * sin(c), (rho * cos(phi1) * cos(c) - y * sin(phi1) * sin(c)));
    phi = fmod(phi + PI_2 + 10 * PI, 2 * PI);
  lambda = fmod(lambda + PI + 10 * PI, 2 * PI);
  float2 source_uv = (float2)(lambda / (2.0 * PI), phi / (PI));
  source_uv = clamp(source_uv, (float2)(0.0f, 0.0f), (float2)(0.999f, 0.999f));

  int source_coord = (int)(source_uv.y * source_height) * source_width +
                     (int)(source_uv.x * source_width);

  int target_coord = j * target_width + i;
  target_buffer[target_coord] = source_buffer[source_coord];
  // target_buffer[target_coord] = (uchar3)(255 * source_uv.y, 0.0f, 0.0f);
}