// SPDX-License-Identifier: MIT
// Copyright (C) 2018-present iced project and contributors

#pragma once
#ifndef ICED_X86_INTEL_FORMATTER_HPP
#define ICED_X86_INTEL_FORMATTER_HPP

#include "formatter_options.hpp"
#include "formatter_output.hpp"
#include "formatter_text_kind.hpp"
#include "symbol_resolver.hpp"
#include "register.hpp"
#include "op_kind.hpp"
#include "mnemonic.hpp"
#include "internal/formatter_regs.hpp"
#include "internal/formatter_mnemonics.hpp"
#include "internal/formatter_memory_size.hpp"
#include <string>
#include <string_view>
#include <cstdint>
#include <format>
#include <optional>

namespace iced_x86 {

// Forward declaration
struct Instruction;

/// @brief Intel (XED) formatter
/// 
/// Formats instructions using Intel syntax (destination, source order).
/// Example: @c mov eax, [ebx+ecx*4+10h]
class IntelFormatter {
public:
  /// @brief Creates a new Intel formatter with default options
  IntelFormatter() = default;

  /// @brief Creates a new Intel formatter with the specified options
  /// @param options Formatter options
  explicit IntelFormatter( const FormatterOptions& options ) : options_( options ) {}

  /// @brief Creates a new Intel formatter with a symbol resolver
  /// @param symbol_resolver Symbol resolver (can be nullptr)
  explicit IntelFormatter( SymbolResolver* symbol_resolver )
      : symbol_resolver_( symbol_resolver ) {}

  /// @brief Creates a new Intel formatter with options and symbol resolver
  /// @param options Formatter options
  /// @param symbol_resolver Symbol resolver (can be nullptr)
  IntelFormatter( const FormatterOptions& options, SymbolResolver* symbol_resolver )
      : options_( options ), symbol_resolver_( symbol_resolver ) {}

  /// @brief Gets the formatter options
  /// @return Formatter options (mutable)
  FormatterOptions& options() noexcept { return options_; }

  /// @brief Gets the formatter options
  /// @return Formatter options (const)
  const FormatterOptions& options() const noexcept { return options_; }

  /// @brief Gets the symbol resolver
  /// @return Symbol resolver or nullptr
  [[nodiscard]] SymbolResolver* symbol_resolver() const noexcept { return symbol_resolver_; }

  /// @brief Sets the symbol resolver
  /// @param resolver Symbol resolver (can be nullptr)
  void set_symbol_resolver( SymbolResolver* resolver ) noexcept { symbol_resolver_ = resolver; }

  /// @brief Formats the instruction
  /// @param instruction Instruction to format
  /// @param output Output to write to
  void format( const Instruction& instruction, FormatterOutput& output );

  /// @brief Formats the instruction to a string
  /// @param instruction Instruction to format
  /// @return Formatted string
  std::string format_to_string( const Instruction& instruction );

  /// @brief Formats a register
  /// @param reg Register
  /// @return Register name
  std::string_view format_register( Register reg ) const noexcept;

private:
  void format_mnemonic( const Instruction& instruction, FormatterOutput& output );
  void format_operands( const Instruction& instruction, FormatterOutput& output );
  void format_operand( const Instruction& instruction, uint32_t operand, FormatterOutput& output );
  void format_register_operand( const Instruction& instruction, uint32_t operand, Register reg,
                                FormatterOutput& output );
  void format_immediate( const Instruction& instruction, uint32_t operand, FormatterOutput& output );
  void format_near_branch( const Instruction& instruction, uint32_t operand, FormatterOutput& output );
  void format_far_branch( const Instruction& instruction, uint32_t operand, FormatterOutput& output );
  void format_memory( const Instruction& instruction, uint32_t operand, FormatterOutput& output );
  void format_evex_decorators( const Instruction& instruction, uint32_t operand, FormatterOutput& output );

  void format_number( uint64_t value, FormatterOutput& output );
  void format_signed_number( int64_t value, FormatterOutput& output );
  void write_symbol( const Instruction& instruction, FormatterOutput& output,
                     uint64_t address, const SymbolResult& symbol, bool write_minus_if_signed = true );

  std::string_view get_mnemonic( Mnemonic mnemonic ) const;
  std::string_view get_memory_size_string( const Instruction& instruction ) const;

  FormatterOptions options_;
  SymbolResolver* symbol_resolver_ = nullptr;
  std::string number_buffer_;  // Reusable buffer for number formatting
};

// ============================================================================
// Implementation
// ============================================================================


















} // namespace iced_x86

#endif // ICED_X86_INTEL_FORMATTER_HPP
