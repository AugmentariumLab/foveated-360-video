# Foveated 360 Video

## Requirements
* Ubuntu 18.04 system with an NVIDIA GPU
* OpenCL headers and NVIDIA drivers
* C++ build tools including g++

### Libraries
* FFmpeg 3.x libraries (libavformat-dev, libavcodec-dev, libswscale-dev, ...)
* SDL
* Boost
* Eigen
* zlib
```
sudo apt install libavformat-dev libavcodec-dev libswscale-dev libavfilter-dev libavutil-dev libavresample-dev
sudo apt install libsdl2-dev libglew-dev libglvnd-dev
sudo apt install libboost-dev
sudo apt install zlib1g-dev
```

## Getting started
Install all prerequisites and run the following:
* `make clean`
* `make all`

Then the server can be started with:
* `./driver.x`

The client can be started with:
* `./client_driver.x <server_addr>`
* For example, to run on localhost `./client_driver.x ws://localhost:9562`

## Source code layout
* All source files are in the `src` folder.
* The transformation functions are written in the OpenCL `.cl` files.
* 1080p versions of benchmark videos are in the `1080p_videos` folder.
