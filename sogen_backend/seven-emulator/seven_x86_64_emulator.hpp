#pragma once

#include <memory>
#include <arch_emulator.hpp>
#include "platform/platform.hpp"

#ifdef SEVEN_EMULATOR_IMPL
#define SEVEN_EMULATOR_DLL_STORAGE EXPORT_SYMBOL
#else
#define SEVEN_EMULATOR_DLL_STORAGE IMPORT_SYMBOL
#endif

namespace seven_backend
{
#if !SOGEN_BUILD_STATIC
    SEVEN_EMULATOR_DLL_STORAGE
#endif
    std::unique_ptr<x86_64_emulator> create_x86_64_emulator();
}
