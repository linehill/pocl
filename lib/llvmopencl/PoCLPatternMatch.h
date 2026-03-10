// Matchers for PoCL specific patterns to use with LLVM's PatternMatch utility
//
// Copyright (c) 2026 Henry Linjamäki / Tampere University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

// NOTE: This file follows a name convention used in llvm/IR/PatternMatch.h.

#ifndef POCL_PATTERN_MATCH_H
#define POCL_PATTERN_MATCH_H

#include "KernelCompilerUtils.h"

#include <llvm/IR/PatternMatch.h>

template <typename MatchTy> struct PoCLBuiltin_match {
  MatchTy ArgMatcher;
  const std::string BuiltinName;

  PoCLBuiltin_match(const MatchTy &TheArgMatcher,
                    llvm::StringRef TheBuiltinName)
      : ArgMatcher(TheArgMatcher), BuiltinName(TheBuiltinName.str()) {}

#if LLVM_MAJOR >= 21
  template <typename ITy> bool match(ITy *V) const {
#else
  template <typename ITy> bool match(ITy *V) {
#endif
    if (llvm::isa<llvm::IntrinsicInst>(V))
      return false;

    const auto *CI = llvm::dyn_cast<llvm::CallInst>(V);
    if (!CI)
      return false;

    const auto *F = CI->getCalledFunction();
    if (!F)
      return false;

    if (F->getName() != BuiltinName)
      return false;

    return ArgMatcher.match(CI->getArgOperand(0));
  }
};

/// Commutable variant of llvm::PatternMatch::m_Add().
template <typename OP0, typename OP1>
inline llvm::PatternMatch::BinaryOp_match<OP0, OP1, llvm::Instruction::Add,
                                          /*Commutable = */ true>
m_CommutativeAdd(const OP0 &Op0, const OP1 &Op1) {
  return llvm::PatternMatch::BinaryOp_match<OP0, OP1, llvm::Instruction::Add,
                                            /*Commutable = */ true>(Op0, Op1);
}

/// Commutable variant of llvm::PatternMatch::m_Mul().
template <typename OP0, typename OP1>
inline llvm::PatternMatch::BinaryOp_match<OP0, OP1, llvm::Instruction::Mul,
                                          /*Commutable = */ true>
m_CommutativeMul(const OP0 &Op0, const OP1 &Op1) {
  return llvm::PatternMatch::BinaryOp_match<OP0, OP1, llvm::Instruction::Mul,
                                            /*Commutable = */ true>(Op0, Op1);
}

/// Matches get_group_id(SomeValue).
template <typename MatchTy>
PoCLBuiltin_match<MatchTy> m_GroupID(const MatchTy &Match) {
  return PoCLBuiltin_match<MatchTy>(Match, GROUP_ID_BUILTIN_NAME);
}

/// Matches get_local_size(SomeValue).
template <typename MatchTy>
PoCLBuiltin_match<MatchTy> m_LocalSize(const MatchTy &Match) {
  return PoCLBuiltin_match<MatchTy>(Match, LS_BUILTIN_NAME);
}

/// Matches get_local_id(SomeValue).
template <typename MatchTy>
PoCLBuiltin_match<MatchTy> m_LocalID(const MatchTy &Match) {
  return PoCLBuiltin_match<MatchTy>(Match, LID_BUILTIN_NAME);
}

/// Matches get_global_id() call.
template <typename MatchTy>
PoCLBuiltin_match<MatchTy> m_GlobalID(const MatchTy &Match) {
  return PoCLBuiltin_match<MatchTy>(Match, GID_BUILTIN_NAME);
}

#endif
