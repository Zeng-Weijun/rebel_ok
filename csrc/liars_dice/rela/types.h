
#pragma once

#include <torch/extension.h>
#include <unordered_map>

namespace rela {

using TensorDict = std::unordered_map<std::string, torch::Tensor>;
using TensorVecDict =
    std::unordered_map<std::string, std::vector<torch::Tensor>>;

using TorchTensorDict = torch::Dict<std::string, torch::Tensor>;
using TorchJitInput = std::vector<torch::jit::IValue>;
using TorchJitOutput = torch::jit::IValue;
using TorchJitModel = torch::jit::script::Module;

inline torch::Tensor quantize(torch::Tensor tensor) {
  return (tensor * 255 + 0.5).clamp_(0., 255.5).to(torch::kByte);
}

inline torch::Tensor dequantize(torch::Tensor tensor) {
  return tensor.to(torch::kFloat32) / 255;
}

class ValueTransition {
 public:
  ValueTransition() = default;

  ValueTransition(const torch::Tensor& query, const torch::Tensor& values)
      : query(query), values(values) {}

  std::vector<torch::Tensor> toVector();
  static ValueTransition fromVector(const std::vector<torch::Tensor>& tensors);

  static ValueTransition makeBatch(
      const std::vector<ValueTransition>& transitions,
      const std::string& device);

  ValueTransition index(int i) const;

  ValueTransition padLike() const;

  TorchJitInput toJitInput(const torch::Device& device) const;

  void write(FILE* file) const;
  static ValueTransition load(FILE* file, bool* success);

  torch::Tensor query;
  torch::Tensor values;
};

}  // namespace rela
