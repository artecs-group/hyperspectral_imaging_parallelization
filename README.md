# Hyperspectral imaging parallelization
<img alt="license" src="https://img.shields.io/github/license/mashape/apistatus.svg"/>

Hyperspectral imaging parallelization with different programming models such as OpenMP, SYCL or Kokkos

## 1. Requirements
To run the code, you will need to install the following dependencies beforehand:

- \>= Make 4.2
- \>= CMake 3.13
- \>= [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) 2023.0, which contains the Intel C++ compiler and the oneMKL library.

Up till this point, you should be able to run the Sequential, OpenMP and SYCL(on CPU and Intel GPU) codes.

### 1.1 Kokkos Requirements
To run with Kokkos implementation you should install [Kokkos](https://github.com/kokkos/kokkos) and [Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) libraries. The versions tested were:

- Kokkos 4.0.00
- Kokkos Kernels 3.7.01

Since the Kokkos implementation were thought to run under CPU, NVIDIA GPU and Intel GPU, we greatly recommend you to create two builds of both Kokkos and Kokkos Kernels. The first one could run under CUDA (nvcc) supporting CPU under OpenMP and NVIDIA GPU under CUDA. The second build should support Intel GPUs using oneAPI SYCL as backend. 

### 1.2 CUDA Requirements
To enable CUDA GPU support for running with OpenMP, SYCL, and Kokkos, you will need to install the following:

- \>= [CUDA 11.4](https://developer.nvidia.com/cuda-11-4-0-download-archive)

Additionally for SYCL, you will require:

- [The Intel standalon compiler](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md).
- [The oneMKL toolchain](https://oneapi-src.github.io/oneMKL/building_the_project.html#building-with-cmake) for the compiler.

## 2. How to run it?
You first need to download the repository:

```bash
> git clone https://github.com/artecs-group/hyperspectral_imaging_parallelization
> cd hyperspectral_imaging_parallelization
> git checkout develop
```

Now, set the environment variables for the standalon compiler, Kokkos or oneAPI toolkit. In the case of oneAPI:

```bash
> source path/to/oneapi/setvars.sh
```

To build the project you can use the following variables to specify in which device you want to run and what programmin model to use.

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| IMPL     | Selects the implementation to run. | sequential, sycl, openmp | non-default |
| DEVICE   | Selects the device whre to run the code. | cpu, igpu (Intel GPU), ngpu (NVIDIA GPU) | cpu |
| PDEBUG   | Used to show debug info during the execution. | yes, no | no |
| KOKKOS_INSTALL_DIR | Path where the user had installed Kokkos | path | /opt/kokkos/build/ |
| KOKKOS_KERNELS_INSTALL_DIR | Path where the user had installed Kokkos Kernels | path | /opt/kokkos-kernels/build |

Then, to build and run the code in sequential mode for the CPU:

```bash
> mkdir build
> cd build
> cmake .. -DIMPL=sequential
> make
> make run
```

### 2.1 Running with Kokkos
In order to execute with Kokkos framework, you have to set where Kokkos is installed, e.g.:

```bash
> mkdir build
> cd build
> cmake .. -DIMPL=kokkos -DKOKKOS_INSTALL_DIR=/opt/kokkos/build -DKOKKOS_KERNELS_INSTALL_DIR=/opt/kokkos-kernels/build
> make
> make run
```