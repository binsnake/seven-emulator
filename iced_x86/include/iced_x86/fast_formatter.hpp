// SPDX-License-Identifier: MIT
// Copyright (C) 2018-present iced project and contributors

#pragma once
#ifndef ICED_X86_FAST_FORMATTER_HPP
#define ICED_X86_FAST_FORMATTER_HPP

#include "fast_formatter_options.hpp"
#include "fast_string_output.hpp"
#include "symbol_resolver.hpp"
#include "code.hpp"
#include "code_size.hpp"
#include "register.hpp"
#include "op_kind.hpp"
#include "mnemonic.hpp"
#include "memory_size.hpp"
#include "rounding_control.hpp"
#include "internal/formatter_regs.hpp"
#include "internal/formatter_mnemonics.hpp"
#include "internal/formatter_memory_size.hpp"
#include <string>
#include <string_view>
#include <cstdint>
#include <optional>

namespace iced_x86 {

// Forward declaration
struct Instruction;

/// @brief Fast formatter with less formatting options and with a masm-like syntax.
///
/// Use it if formatting speed is more important than being able to re-assemble
/// formatted instructions.
///
/// This formatter is optimized for speed by:
/// - Using simpler code paths
/// - Pre-computed lookup tables
/// - Minimal virtual calls
/// - Direct string operations
///
/// Example:
/// @code
/// FastFormatter formatter;
/// FastStringOutput output;
/// formatter.format(instruction, output);
/// std::cout << output.view() << std::endl;
/// @endcode
class FastFormatter {
public:
  /// @brief Creates a new fast formatter with default options
  FastFormatter() = default;

  /// @brief Creates a new fast formatter with the specified options
  /// @param options Formatter options
  explicit FastFormatter( const FastFormatterOptions& options ) : options_( options ) {}

  /// @brief Creates a new fast formatter with a symbol resolver
  /// @param symbol_resolver Symbol resolver (can be nullptr)
  explicit FastFormatter( SymbolResolver* symbol_resolver )
      : symbol_resolver_( symbol_resolver ) {}

  /// @brief Creates a new fast formatter with options and symbol resolver
  /// @param options Formatter options
  /// @param symbol_resolver Symbol resolver (can be nullptr)
  FastFormatter( const FastFormatterOptions& options, SymbolResolver* symbol_resolver )
      : options_( options ), symbol_resolver_( symbol_resolver ) {}

  /// @brief Gets the formatter options (mutable)
  /// @return Formatter options
  FastFormatterOptions& options() noexcept { return options_; }

  /// @brief Gets the formatter options (const)
  /// @return Formatter options
  [[nodiscard]] const FastFormatterOptions& options() const noexcept { return options_; }

  /// @brief Gets the symbol resolver
  /// @return Symbol resolver or nullptr
  [[nodiscard]] SymbolResolver* symbol_resolver() const noexcept { return symbol_resolver_; }

  /// @brief Sets the symbol resolver
  /// @param resolver Symbol resolver (can be nullptr)
  void set_symbol_resolver( SymbolResolver* resolver ) noexcept { symbol_resolver_ = resolver; }

  /// @brief Formats the whole instruction: prefixes, mnemonic, operands
  /// @param instruction Instruction to format
  /// @param output Output buffer
  void format( const Instruction& instruction, FastStringOutput& output );

  /// @brief Formats the instruction and returns it as a string
  /// @param instruction Instruction to format
  /// @return Formatted instruction string
  [[nodiscard]] std::string format_to_string( const Instruction& instruction );

private:
  // Format helpers
  [[clang::noinline]] void format_register( FastStringOutput& output, Register reg );
  [[clang::noinline]] void format_number( FastStringOutput& output, uint64_t value );
  [[clang::noinline]] void format_memory( FastStringOutput& output, const Instruction& instruction, uint32_t operand,
                      Register seg_reg, Register base_reg, Register index_reg,
                      uint32_t scale, uint32_t displ_size, int64_t displ, uint32_t addr_size );
  [[clang::noinline]] void write_symbol( FastStringOutput& output, uint64_t address, const SymbolResult& symbol,
                     bool write_minus_if_signed = true );

  [[nodiscard]] std::string_view get_mnemonic( Mnemonic mnemonic ) const;
  [[nodiscard]] std::string_view get_memory_size_string( MemorySize size ) const;
  [[nodiscard]] bool show_segment_prefix( const Instruction& instruction, uint32_t op_count ) const;
  [[nodiscard]] uint32_t get_address_size( Register base_reg, Register index_reg,
                                            uint32_t displ_size, CodeSize code_size ) const;

  FastFormatterOptions options_;
  SymbolResolver* symbol_resolver_ = nullptr;
};

// ============================================================================
// Implementation
// ============================================================================







} // namespace iced_x86

#endif // ICED_X86_FAST_FORMATTER_HPP
