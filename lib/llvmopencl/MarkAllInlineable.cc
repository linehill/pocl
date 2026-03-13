// LLVM pass to recursively inline kernels which are called by other kernels
//
// Copyright (c) 2020 Michal Babej / Tampere University
//               2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "MarkAllInlineable.hh"
#include "LLVMUtils.h"
POP_COMPILER_DIAGS

//#define DEBUG_MARK_ALL_INLINEABLE

#include "pocl_llvm_api.h"

#include <iostream>
#include <string>

#define PASS_NAME "mark-all-inlineable"
#define PASS_CLASS pocl::MarkAllInlineable
#define PASS_DESC "mark all functions inlineable"

namespace pocl {

using namespace llvm;

static bool markAllInlineable(Module &M) {
  bool changed = false;

  // removes noinline & optnone, but doesn't add alwaysinline
  for (auto &F : M.functions()) {
    changed = true;
    F.removeFnAttr(Attribute::NoInline);
    F.removeFnAttr(Attribute::OptimizeNone);
    // remove noInline from the callsite. otherwise it could cause alwaysInline
    // pass to skip the inlining
    for (auto U: F.users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI) continue;
      CI->removeFnAttr(Attribute::NoInline);
      CI->removeFnAttr(Attribute::NoBuiltin);
      CI->removeFnAttr(Attribute::OptimizeNone);
    }
  }

  return changed;
}

llvm::PreservedAnalyses MarkAllInlineable::run(llvm::Module &M,
                                              llvm::ModuleAnalysisManager &AM) {
  // PreservedAnalyses PAChanged = PreservedAnalyses::none();
  //return markAllInlineable(M) ? PAChanged : PreservedAnalyses::all();
  markAllInlineable(M);
  return PreservedAnalyses::all();
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
