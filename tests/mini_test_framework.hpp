#pragma once

#include <gtest/gtest.h>

// Compatibility wrapper: keep legacy bool-returning test bodies but run them as gtest cases.
#ifdef TEST
#undef TEST
#endif
#define TEST(Suite, Name)                                                           \
  static bool Suite##_##Name##_impl();                                              \
  GTEST_TEST(Suite, Name) { EXPECT_TRUE(Suite##_##Name##_impl()); }                \
  static bool Suite##_##Name##_impl()
