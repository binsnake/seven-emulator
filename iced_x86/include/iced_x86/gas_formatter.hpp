// SPDX-License-Identifier: MIT
// Copyright (C) 2018-present iced project and contributors

#pragma once
#ifndef ICED_X86_GAS_FORMATTER_HPP
#define ICED_X86_GAS_FORMATTER_HPP

#include "formatter_options.hpp"
#include "formatter_output.hpp"
#include "formatter_text_kind.hpp"
#include "symbol_resolver.hpp"
#include "register.hpp"
#include "op_kind.hpp"
#include "mnemonic.hpp"
#include "memory_size.hpp"
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

/// @brief GAS (GNU Assembler) formatter - AT&T syntax
/// 
/// Formats instructions using AT&T/GAS syntax:
/// - Operands are reversed (source, destination order)
/// - Registers have % prefix (%eax, %rbx)
/// - Immediates have $ prefix ($10, $0x1234)
/// - Memory uses parenthetical syntax: disp(%base,%index,scale)
/// Example: @c movl $10, %eax  or  movl 0x10(%ebx,%ecx,4), %eax
class GasFormatter {
public:
  /// @brief Creates a new GAS formatter with default options
  GasFormatter() = default;

  /// @brief Creates a new GAS formatter with the specified options
  /// @param options Formatter options
  explicit GasFormatter( const FormatterOptions& options ) : options_( options ) {}

  /// @brief Creates a new GAS formatter with a symbol resolver
  /// @param symbol_resolver Symbol resolver (can be nullptr)
  explicit GasFormatter( SymbolResolver* symbol_resolver )
      : symbol_resolver_( symbol_resolver ) {}

  /// @brief Creates a new GAS formatter with options and symbol resolver
  /// @param options Formatter options
  /// @param symbol_resolver Symbol resolver (can be nullptr)
  GasFormatter( const FormatterOptions& options, SymbolResolver* symbol_resolver )
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

  /// @brief Formats a register with % prefix
  /// @param reg Register
  /// @return Register name with % prefix
  std::string format_register( Register reg ) const;

  /// @brief Gets if naked registers are used (no % prefix)
  /// @return True if naked registers enabled
  bool naked_registers() const noexcept { return naked_registers_; }

  /// @brief Sets if naked registers are used (no % prefix)
  /// @param value True to enable naked registers
  void set_naked_registers( bool value ) noexcept { naked_registers_ = value; }

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
  char get_size_suffix( const Instruction& instruction ) const;

  FormatterOptions options_;
  SymbolResolver* symbol_resolver_ = nullptr;
  std::string number_buffer_;  // Reusable buffer for number formatting
  std::string register_buffer_;  // Reusable buffer for register formatting
  bool naked_registers_ = false;  // If true, don't use % prefix on registers
};

// ============================================================================
// Implementation
// ============================================================================


















} // namespace iced_x86

#endif // ICED_X86_GAS_FORMATTER_HPP
