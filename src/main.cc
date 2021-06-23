/* Copyright 2019-2025 by Bitmain Technologies Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

==============================================================================*/
#include <iostream>
#include <vector>
#include "paddle_api.h"            //NOLINT
#include "config.h"

using namespace paddle::lite_api;  // NOLINT

void ConvertModel(Config config_obj) {
  CxxConfig config;
#if 1
  config.set_model_file(config_obj.model_path_ + "/model");
  config.set_param_file(config_obj.model_path_ + "/params");
#else // for __model__
  config.set_model_dir(config_obj.model_path_);
#endif 
 
 
  config.set_valid_places({Place{TARGET(kBM), PRECISION(kFloat)},
                           Place{TARGET(kHost), PRECISION(kFloat)}});

  std::shared_ptr<PaddlePredictor> predictor =
                 CreatePaddlePredictor<CxxConfig>(config);

  for (size_t i = 0; i < config_obj.input_shapes_.size(); i++) {
    std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(i)));
    shape_t in_tensor_shape;
    for (size_t j = 0; j < config_obj.input_shapes_[i].size(); j++) {
      in_tensor_shape.push_back(config_obj.input_shapes_[i][j]);
    }
    input_tensor->Resize(in_tensor_shape);
    auto* data = input_tensor->mutable_data<float>();
  }

  predictor->Run();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0] << " config_path\n";
    exit(1);
  }
  std::string config_path = argv[1];
  Config config_obj(config_path);

  ConvertModel(config_obj);
  std::cout << "=========================Convert Success, Happy" << std::endl;
  return 0;
}
