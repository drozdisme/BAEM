#pragma once

#include "models/sle/full_engine.hpp"
#include "sle/framework.hpp"

namespace core::models::sle {

using TaskKind = ::sle::TaskKind;
using Sample = ::sle::Sample;
using Dataset = ::sle::Dataset;
using TrainerConfig = ::sle::TrainerConfig;
using Metrics = ::sle::Metrics;
using FrameworkModel = ::sle::FrameworkModel;
using Trainer = ::sle::Trainer;

using ::sle::make_boolean_dataset;

} // namespace core::models::sle
