// unified_ml — ABI version symbol
// This TU bakes the version string into the shared library binary.
// The symbol is exported so consumers can verify ABI compatibility at runtime.
#include "unified_ml_abi.hpp"

namespace unified_ml {

UNIFIED_ML_API const char* abi_version() noexcept {
    return UNIFIED_ML_VERSION_STRING;
}

} // namespace unified_ml
