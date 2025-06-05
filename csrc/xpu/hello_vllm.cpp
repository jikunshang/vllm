#include <torch/all.h>
#include <iostream>

void hello_vllm(at::Tensor& a) {
  std::cout << "hello vllm: " << a.sizes() << std::endl;
}
