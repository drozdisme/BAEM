#pragma once

#include "models/sle/circuit.hpp"
#include "sle/synthesis.hpp"

namespace core::models::sle {

using SynthesisConfig = ::sle::SynthesisConfig;
using SynthesisResult = ::sle::SynthesisResult;
using ResidualSynthesisResult = ::sle::ResidualSynthesisResult;

using ::sle::synthesize_local;
using ::sle::synthesize_multigate;
using ::sle::synthesize_with_residual;

} // namespace core::models::sle
