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

#pragma once

#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <cstring>
#include <vector>
#include "utility.h"

class Config {
public:
  explicit Config(const std::string &config_file) {
    config_map_ = LoadConfig(config_file);
    PrintConfigInfo();

    this->model_path_ = config_map_["model_path"];
    this->model_format_ = config_map_["model_format"];
    auto str_input_shapes = config_map_["input_shapes"];
    auto str_input_shape = split(str_input_shapes, ":");
    for (size_t i = 0; i < str_input_shape.size(); i++) {
      std::vector<int> input_shape;
      auto str_input_dim = split(str_input_shape[i], ",");
      for (size_t j = 0; j < str_input_dim.size(); j++) {
        input_shape.push_back(std::stoi(str_input_dim[j]));
      }
      input_shapes_.push_back(input_shape);
    }
  }

  std::string model_path_;
  std::string model_format_;
  std::vector<std::vector<int> > input_shapes_;

private:
  // Load configuration
  std::map<std::string, std::string> LoadConfig(const std::string &config_file);

  std::vector<std::string> split(const std::string &str,
                                 const std::string &delim);

  std::map<std::string, std::string> config_map_;
  void PrintConfigInfo();
};
