# Foveated 360 Video

Foveated 360 Video is a client-server system prototype for streaming 360° videos which leverages parallel algorithms over real-time video transcoding.
See <https://augmentariumlab.github.io/foveated-360-video> for the corresponding video, paper, and slides published in IEEE Virtual Reality 2021.

## Requirements

* Ubuntu 20.04 system with an NVIDIA GPU
* OpenCL headers and NVIDIA drivers
* C++ build tools including g++

### Libraries

* FFmpeg libraries (libavformat-dev, libavcodec-dev, libswscale-dev, ...)
* SDL
* Boost
* Eigen
* zlib

```sh
sudo apt install ocl-icd-opencl-dev opencl-headers opencl-c-headers opencl-clhpp-headers
sudo apt install libavformat-dev libavcodec-dev libswscale-dev libavfilter-dev libavutil-dev libavresample-dev
sudo apt install libsdl2-dev libglew-dev libglvnd-dev freeglut3-dev
sudo apt install libboost-dev
sudo apt install zlib1g-dev
```

## Getting started

Download and extract the [1080p video dataset](https://drive.google.com/file/d/13C7-47pQBd_qcqtJ8FxUunOgboaFxpzH/view?usp=sharing).

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

## Related Publication

Please refer to <https://augmentariumlab.github.io/foveated-360-video> for our paper published in IEEE VR 2021 (TVCG): "A Log-Rectilinear Transformation for Foveated 360-degree Video Streaming".

## References

If you use ARCore Depth Lab in your research, please reference it as:

    @article{Li2021LogRectilinear,
      author={Li, David and Du, Ruofei and Babu, Adharsh and Brumar, Camelia D. and Varshney, Amitabh},
      journal={IEEE Transactions on Visualization and Computer Graphics},
      title={A Log-Rectilinear Transformation for Foveated 360-degree Video Streaming},
      year={2021},
      volume={27},
      number={5},
      pages={2638-2647},
      doi={10.1109/TVCG.2021.3067762}
    }

or

    David Li, Ruofei Du, Adharsh Babu, Camelia Brumar, and Amitabh Varshney. 2021. A Log-Rectilinear Transformation for Foveated 360-Degree Video Streaming. IEEE Transactions on Visualization and Computer Graphics, 27(5), 2638-2647. DOI: <https://doi.org/10.1109/TVCG.2021.3067762>
