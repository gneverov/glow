/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdlib>
#include <iostream>
#include <random>

#include "Bench.h"

using namespace glow;

extern "C" {
// Forward declare functions from libjit.
void libjit_convolution_f(
    float *outW, const float *inW, const float *filterW, const float *biasW,
    const size_t *outWdims, const size_t *inWdims, const size_t *filterWdims,
    const size_t *biasWdims, const size_t *filterSizes, const size_t *strides,
    const size_t *pads, size_t group, unsigned depthUnroll);
}

/// Benchmark convolution.
/// Input image size: n x h x w x inC
/// Filter size: outC x kernel x kernel x inC + outC (bias)
/// Filter stride: stride, stride
/// Output image size: n x h / stride x w / stride x outC
class ConvBench : public Benchmark {
  /// Matrices.
  std::vector<float> filter;
  std::vector<float> in;
  std::vector<float> out;
  std::vector<float> bias;

  /// Dimensions expressed in libjit's format.
  const std::vector<size_t> filterDims;
  const std::vector<size_t> inDims;
  const std::vector<size_t> outDims;
  const std::vector<size_t> biasDims;

public:
  ConvBench(
      size_t n, size_t h, size_t w, size_t inC, size_t outC, size_t kernel,
      size_t stride) :

      filterDims({outC, kernel, kernel, inC}),
      inDims({n, h, w, inC}),
      outDims({n, h / stride, w / stride, outC}),
      biasDims({outC}) {}

  virtual void setup() override {
    random_initialize(filterDims, &filter);
    random_initialize(inDims, &in);
    random_initialize(outDims, &out);
    random_initialize(biasDims, &bias);
  }

  virtual void run() override {
    size_t stride = inDims[1] / outDims[1];
    size_t strides[] = {stride, stride};
    size_t pads[] = {0, 0};
    libjit_convolution_f(
        out.data(), in.data(), filter.data(), bias.data(), outDims.data(),
        inDims.data(), filterDims.data(), biasDims.data(), &filterDims[1],
        strides, pads, 4, 4);
  }

  virtual void teardown() override {}

  double gflops() const {
    return
        2.0 * filterDims[0] * filterDims[1] * filterDims[2] * filterDims[3] *
            outDims[0] * outDims[1] * outDims[2] / 1e9;
  }

private:
  void random_initialize(
      const std::vector<size_t>& dims, std::vector<float>* v) {
    std::mt19937 gen;
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    size_t size = std::accumulate(
      dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    v->resize(size);
    for (size_t i = 0; i < size; i++) {
      (*v)[i] = dis(gen);
    }
  }
};

int main() {
  constexpr int reps = 100;
  printf(
    "     N,    InW,    InH,    InC,   OutC, Kernel, Stride, gflops/s, \n");

  size_t batch = 16;
  // In order to have some guidance on what convolution shapes are practically
  // useful, we benchmark all the convolution layers of MobileNet
  // (https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py).
  // This struct defines the kernel size, stride, and output depth of each
  // convolution. MobileNet uses depthwise separable convolution layers. Each
  // such layer is represented here as two convolutions: a kxk depthwise
  // convolution with batch size multiplied by the current depth,  followed by a
  // 1x1 pointwise convolution. A depth value of 0 in this struct indicates a
  // depthwise convolution.
  struct {
    size_t kernel;
    size_t stride;
    size_t depth;
  } layers[] = {
    {3, 2, 32},
    {3, 1, 0},
    {1, 1, 64},
    {3, 2, 0},
    {1, 1, 128},
    {3, 1, 0},
    {1, 1, 128},
    {3, 2, 0},
    {1, 1, 256},
    {3, 1, 0},
    {1, 1, 256},
    {3, 2, 0},
    {1, 1, 512},
    {3, 1, 0},
    {1, 1, 512},
    {3, 2, 0},
    {1, 1, 1024},
    {3, 1, 0},
    {1, 1, 1024},
  };

  size_t size = 224;
  size_t depth = 3;
  for (auto& layer : layers) {
    size_t n, inC, outC;
    if (layer.depth == 0) {
      n = batch * depth;
      inC = 1;
      outC = 1;
    } else {
      n = batch;
      inC = depth;
      outC = layer.depth;
    }
    ConvBench b(n, size, size, inC, outC, layer.kernel, layer.stride);
    auto time = bench(&b, reps);
    printf(
        "%6zu, %6zu, %6zu, %6zu, %6zu, %6zu, %6zu, %6.2lf\n", n, size, size,
        inC, outC, layer.kernel, layer.stride, b.gflops() / time);

    size /= layer.stride;
    if (layer.depth != 0) {
      depth = layer.depth;
    }
  }
}
