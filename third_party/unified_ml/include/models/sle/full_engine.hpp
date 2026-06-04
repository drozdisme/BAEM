#pragma once

#include "models/sle/circuit.hpp"
#include "sle/full_engine.hpp"

namespace core::models::sle {

using EngineConfig = ::sle::FullEngineConfig;
using FullEngine = ::sle::FullEngineModel;
using FullEngineTrainResult = ::sle::FullEngineTrainResult;
using HardLogicContract = ::sle::HardLogicContract;

using ::sle::run_full_engine;
using ::sle::train_full_engine;

} // namespace core::models::sle
