// LLVM function pass that adds implicit barriers to loops if it sees
// beneficial.
//
// Copyright (c) 2012-2013 Pekka Jääskeläinen / Tampere University of Tech.
//               2023-2024 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>

#include "Barrier.h"
#include "ImplicitLoopBarriers.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include "pocl_runtime_config.h"

#include <iostream>

//#define DEBUG_ILOOP_BARRIERS

#define PASS_NAME "implicit-loop-barriers"
#define PASS_CLASS pocl::ImplicitLoopBarriers
#define PASS_DESC "Adds implicit barriers to loops"

namespace pocl {

using namespace llvm;

/// Adds barriers to uniform loops without barriers to force horizontal
/// vectorization across work-items.
///
/// Currently adds the barriers whenever analyzed legal without considering
/// the vectorization benefits.
bool ImplicitLoopBarriers::addImplicitLoopBarriers(Loop &L) {

  if (Barrier::isLoopWithBarrier(L) || !VUA->isUniformLoop(*F, L))
    return false;

  // Only add barriers to the innermost loops.
  if (L.getSubLoops().size() > 0)
    return false;

  llvm::BasicBlock *ExitingBlock = L.getExitingBlock();
  llvm::BasicBlock *HeaderBlock = L.getHeader();

  // Isolate the loop body to a parallel region with two barriers.
  Barrier::create(ExitingBlock->getTerminator());
  Barrier::create(HeaderBlock->getFirstNonPHI());

#ifdef DEBUG_ILOOP_BARRIERS
  std::cerr << "### added inner-loop barriers to loop " << L.getName().str()
            << std::endl
            << std::endl;
  ExitingBlock->dump();
  HeaderBlock->dump();
#endif

  return false;
}

llvm::PreservedAnalyses
ImplicitLoopBarriers::run(llvm::Loop &L, llvm::LoopAnalysisManager &AM,
                          llvm::LoopStandardAnalysisResults &AR,
                          llvm::LPMUpdater &U) {

  F = L.getHeader()->getParent();

  auto &FAMP = AM.getResult<FunctionAnalysisManagerLoopProxy>(L, AR);

  if (!isKernelToProcess(*F))
    return PreservedAnalyses::all();

#ifdef DEBUG_COND_BARRIERS
  std::cerr << "### Before ImplicitLoopBarriers " << std::endl;
  F.dump();
#endif

  if (FAMP.cachedResultExists<WorkitemHandlerChooser>(*F)) {
    auto Res = FAMP.getCachedResult<WorkitemHandlerChooser>(*F);
    if (Res->WIH == WorkitemHandlerType::CBS)
      return PreservedAnalyses::all();
  } else {
    assert(0 && "missing cached result WIH for ImplicitLoopBarriers");
  }

  if (!pocl_get_bool_option("POCL_FORCE_PARALLEL_OUTER_LOOP", 1) &&
      !hasWorkgroupBarriers(*F)) {
#ifdef DEBUG_ILOOP_BARRIERS
    std::cerr
        << "### ILB: The kernel has no barriers, let's not add implicit ones "
        << "either to avoid WI context switch overheads" << std::endl;
#endif
    return PreservedAnalyses::all();
  }

  VUA = nullptr;
  if (FAMP.cachedResultExists<VariableUniformityAnalysis>(*F)) {
    VUA = FAMP.getCachedResult<VariableUniformityAnalysis>(*F);
  } else {
    assert(0 && "Missing cached VUA results for ImplicitLoopBarriers");
  }

  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  PAChanged.preserve<VariableUniformityAnalysis>();
  bool Changed = addImplicitLoopBarriers(L);

#ifdef DEBUG_COND_BARRIERS
  if (Changed) {
    std::cerr << "### After ImplicitLoopBarriers' changes " << std::endl;
    F.dump();
  }
#endif

  return Changed ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_LPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
