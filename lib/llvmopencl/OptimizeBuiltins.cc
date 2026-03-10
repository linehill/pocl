// Optimize PoCL kernel builtins.
//
// Copyright (c) 2026 Henry Linjamäki / Tampere University
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>

#include "DebugHelpers.h"
#include "LLVMUtils.h"
#include "OptimizeBuiltins.h"
#include "PoCLPatternMatch.h"
#include "pocl_llvm_api.h"
POP_COMPILER_DIAGS

#define PASS_NAME "opt-builtins"
#define PASS_CLASS pocl::OptimizeBuiltins
#define PASS_DESC "Optimize PoCL builtins."

namespace pocl {

using namespace llvm;
using namespace llvm::PatternMatch;

bool OptimizeBuiltins::applyRangeMDToBuiltin(Instruction *I) {

  // TODO: if the 'I' already has RangeMD, intersect it with ones applied here
  //       rather than override it.

  if (match(I, m_GlobalID(m_Value())) && MaxGridWidthOpt) {
    setRangeMetadata(I, 0, *MaxGridWidthOpt - 1);
    return true;
  }

  uint64_t Dim;
  if (match(I, m_LocalID(m_ConstantInt(Dim))) && Dim < 3 &&
      LocalSizeOpts[Dim]) {
    setRangeMetadata(I, 0, *LocalSizeOpts[Dim] - 1);
    return true;
  }
  return false;
}

bool OptimizeBuiltins::applyRangeMDToBuiltins(Function &F) {
  bool Changed = false;
  for (auto &BB : F)
    for (auto &I : BB)
      Changed |= applyRangeMDToBuiltin(&I);
  return Changed;
}

bool OptimizeBuiltins::combineBuiltins(Instruction &I) {
  auto *M = I.getParent()->getParent()->getParent();

  // Combine patterns that are like get_global_id(x).

  Value *Dim = nullptr;
  // Be sure to have m_Value(Dim) matched before m_Deferred(Dim) in each
  // complete pattern.
  auto GroupID = m_GroupID(m_Value(Dim));
  auto LocalSz = m_LocalSize(m_Deferred(Dim));
  auto LocalID = m_LocalID(m_Deferred(Dim));

  // get_global_id(x) * get_local_size(x) + get_local_id(x).
  auto GlobalIDVariant0 =
      m_CommutativeAdd(m_CommutativeMul(GroupID, LocalSz), LocalID);

  // (ty)get_global_id(x) * (ty)get_local_size(x) + (ty)get_local_id(x), where
  // 'ty's bitwidth is smaller that the builtins' bitwidth.
  auto GlobalIDVariant1 = m_CommutativeAdd(
      m_CommutativeMul(m_Trunc(GroupID), m_Trunc(LocalSz)), m_Trunc(LocalID));

  if ((match(&I, GlobalIDVariant0) || match(&I, GlobalIDVariant1)) &&
      HasNoGlobalOffset) {

    Value *NewGID = createGlobalID(M, Dim, BasicBlock::iterator(I));
    applyRangeMDToBuiltin(cast<Instruction>(NewGID));

    IRBuilder<> B(&I);
    NewGID = B.CreateZExtOrTrunc(NewGID, I.getType());

    I.replaceAllUsesWith(NewGID);
    // TODO: DCE the I (outside the combineBuiltins loop).
  }

  return false;
}

bool OptimizeBuiltins::combineBuiltins(Function &F) {
  bool Changed = false;
  for (auto &BB : F)
    for (auto &I : BB)
      Changed |= combineBuiltins(I);
  return Changed;
}

static std::optional<uint64_t> getMaxGridWidth(Function &F) {
  uint64_t MaxGridWidth;
  if (getModuleIntMetadata(*F.getParent(), "WGMaxGridDimWidth", MaxGridWidth))
    if (MaxGridWidth != 0) // Zero means unknown width.
      return MaxGridWidth;
  return std::nullopt;
}

static std::array<std::optional<size_t>, 3> getLocalSizes(Function &F) {
  std::array<std::optional<size_t>, 3> Result{};

  bool HasDynamicLocalSizes = true;
  getModuleBoolMetadata(*F.getParent(), "WGDynamicLocalSize",
                        HasDynamicLocalSizes);
  if (HasDynamicLocalSizes)
    return Result;

  const char *MDNames[] = {"WGLocalSizeX", "WGLocalSizeY", "WGLocalSizeZ"};
  uint64_t MDValue;
  for (auto [Index, MDName] : enumerate(MDNames))
    if (getModuleIntMetadata(*F.getParent(), MDName, MDValue))
      Result[Index] = MDValue;

  return Result;
}

llvm::PreservedAnalyses OptimizeBuiltins::run(Function &F,
                                              FunctionAnalysisManager &AM) {

  MaxGridWidthOpt = getMaxGridWidth(F);
  LocalSizeOpts = getLocalSizes(F);
  HasNoGlobalOffset = true;
  getModuleBoolMetadata(*F.getParent(), "WGAssumeZeroGlobalOffset",
                        HasNoGlobalOffset);

  bool Changed = false;
  Changed |= applyRangeMDToBuiltins(F);
  Changed |= combineBuiltins(F);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
