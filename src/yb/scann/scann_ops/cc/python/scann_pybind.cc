// Copyright 2025 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <string>

#include "absl/types/optional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "scann/scann_ops/cc/scann_npy.h"

PYBIND11_MODULE(scann_pybind, py_module) {
  py_module.doc() = "pybind11 wrapper for ScaNN";
  pybind11::class_<yb::ScannNumpy>(py_module, "ScannNumpy")
      .def(pybind11::init<const std::string&, const std::string&>())
      .def(pybind11::init<const yb::np_row_major_arr<float>&, const std::string&, int>())
      .def("search", &yb::ScannNumpy::Search)
      .def("search_batched", &yb::ScannNumpy::SearchBatched)

      .def("upsert", &yb::ScannNumpy::Upsert)
      .def("delete", &yb::ScannNumpy::Delete)
      .def("rebalance", &yb::ScannNumpy::Rebalance)
      .def_static("suggest_autopilot", &yb::ScannNumpy::SuggestAutopilot)
      .def("size", &yb::ScannNumpy::Size)
      .def("reserve", &yb::ScannNumpy::Reserve)
      .def("set_num_threads", &yb::ScannNumpy::SetNumThreads)
      .def("config", &yb::ScannNumpy::Config)
      .def("serialize", &yb::ScannNumpy::Serialize)
      .def("get_health_stats", &yb::ScannNumpy::GetHealthStats)
      .def("initialize_health_stats", &yb::ScannNumpy::InitializeHealthStats);
}
