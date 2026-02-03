# powerspectrum

Cosmological Power Spectrum Analysis & MCMC Parameter Estimation

## Overview
このリポジトリは、銀河パワースペクトルの解析および宇宙論的パラメータ推定を行うためのコードセットです。
**Dark Quest Simulation** のハローデータと **SDSS/BOSS CMASS** の観測データを比較し、MCMC（Metropolis-Hastings法）を用いて銀河-ハロー関係（HOD等）のパラメータ推定を行います。また、パワースペクトルの多重極モーメント（Multipole moments）を用いたダークエネルギーモデルの検証も目的としています。

## Features
* **Power Spectrum Calculation**: FFTWを用いたパワースペクトルおよび多重極モーメントの計算
* **MCMC Sampling**: C++によるMetropolis-Hastingsアルゴリズムの実装（OpenMP並列化対応）
* **Halo Analysis**: Friends-of-Friends (FoF) ハローファインダーおよび $V_{\text{max}}$ 等のハロープロパティの解析
* **Visualization**: Python (`getdist`, `matplotlib`) を用いた事後分布の等高線図（Contour plots）およびパワースペクトルのプロット作成

## Requirements

### C++ (Simulation & MCMC)
* **Compiler**: GCC (g++) with C++11/14 support
* **Libraries**:
    * [Eigen 3](https://eigen.tuxfamily.org/) (Linear algebra)
    * [GSL (GNU Scientific Library)](https://www.gnu.org/software/gsl/) (Random number generation, Integration)
    * [FFTW 3](http://www.fftw.org/) (Fast Fourier Transform)
    * [OpenMP](https://www.openmp.org/) (Parallelization)

### Python (Analysis & Plotting)
* Python 3.x
* numpy
* pandas
* matplotlib
* [GetDist](https://getdist.readthedocs.io/) (MCMC plot visualization)
* (Optional) blackjax, jax

## Installation & Build

```bash
# Clone the repository
git clone git@github.com:hr17291109/powerspectrum.git
cd powerspectrum

# Compile the C++ code (Example)
# Ensure paths to Eigen, GSL, and FFTW are correctly set
g++ -O3 -fopenmp -I/usr/include/eigen3 main.cpp -o run_mcmc -lgsl -lgslcblas -lm -lfftw3
