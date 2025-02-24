// LLVM function pass that adds implicit barriers to loops if it sees
// beneficial.
//
// Copyright (c) 2012-2013 Pekka Jääskeläinen / Tampere University of Tech.
//               2023-2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include "CanonicalizeBarriers.h"
#include "DebugHelpers.h"
#include "ImplicitLoopBarriers.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkgroupBarrier.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include "pocl_runtime_config.h"

#include <iostream>

// #define DEBUG_ILOOP_BARRIERS

namespace pocl {

using namespace llvm;

/// Recursively counts the number of times work-item IDs terms appear in the
/// address expression.
///
/// Currently considers only the X dimension.
static size_t countWorkitemIDTerms(Value *Term, int RecursionDepth) {
#ifdef DEBUG_ILOOP_BARRIERS
  std::cerr << "#### Pointer term:\n";
  Term->dump();
#endif
  Instruction *Inst = dyn_cast_or_null<Instruction>(Term);

  if (isa<ZExtInst>(Term) || isa<TruncInst>(Term))
    return countWorkitemIDTerms(Inst->getOperand(0), RecursionDepth);

  if (RecursionDepth > 10)
    return 0;

  if (CallInst *Call = dyn_cast_or_null<CallInst>(Term))
    if (Call->getCalledFunction() != nullptr &&
        (Call->getCalledFunction()->getName() == GID_BUILTIN_NAME ||
         Call->getCalledFunction()->getName() == LID_BUILTIN_NAME) &&
        isa<llvm::ConstantInt>(Call->getArgOperand(0)) &&
        cast<llvm::ConstantInt>(Call->getArgOperand(0))->getZExtValue() == 0)
      return 1;

  if (Inst == nullptr)
    return 0;

  // Recurse into addition's operands in case of non-GEP address arithmetic.
  // If it's a mul, assume non-unit stride.
  if (GetElementPtrInst *GEP = dyn_cast_or_null<GetElementPtrInst>(Inst)) {
    size_t IDTerms = 0;
    for (auto &AddrTerm : GEP->operands()) {
      IDTerms += countWorkitemIDTerms(AddrTerm, RecursionDepth + 1);
    }
    return IDTerms;
  }
  if (Inst->isBinaryOp() && Inst->getOpcode() == Instruction::Add) {
    return countWorkitemIDTerms(Inst->getOperand(0), RecursionDepth + 1) +
           countWorkitemIDTerms(Inst->getOperand(1), RecursionDepth + 1);
  }
  if (LoadInst *Load = dyn_cast_or_null<LoadInst>(Inst)) {
    // In unoptimized non-SSA input we might load terms of the address from
    // alloca'd temporary variables.
    return countWorkitemIDTerms(Inst->getOperand(0), RecursionDepth + 1);
  }
  if (AllocaInst *Alloca = dyn_cast_or_null<AllocaInst>(Inst)) {
    // If the load is from an alloca, let's traverse all stores to it.
    size_t IDTerms = 0;
    for (Instruction::use_iterator UI = Alloca->use_begin(),
                                   UE = Alloca->use_end();
         UI != UE; ++UI) {
      llvm::StoreInst *Store = dyn_cast_or_null<StoreInst>(UI->getUser());
      if (Store == nullptr)
        continue;
      IDTerms +=
          countWorkitemIDTerms(Store->getValueOperand(), RecursionDepth + 1);
    }
    return IDTerms;
  }
  return 0;
}

/// Analyzes the Loop and returns true if outer-loop vectorization is
/// likely more efficient than inner-loop vectorization, thus it is sensible
/// to convert the kernel inner-loop to a B-loop.
///
/// From the other perspective: Tries to to leave such loops intact which are
/// more likely to get more efficient inner-loop vectorization than outer.
/// This should be preferably decided in LLVM's loop interchange, but it might
/// be difficult to add the context data etc. needed for producing the outer
/// loop case there.
static bool outerLoopIsLikelyBeneficial(Loop &L) {
  /// Check the memory accessess of the loop. If the loop more often has the
  /// work item id as a term, than not, we assume it's more efficient to
  /// vectorize over the WI loop. Note that LLVM loopvec can unroll loops to
  /// handle non-unit strides so we should not limit only to single WI-id
  /// stepping accesses.
  size_t WIAddresses = 0;
  size_t NonWIAddresses = 0;
  for (auto &BB : L.getBlocksVector()) {
    for (auto &Inst : *BB) {
      Value *PtrOpr = nullptr;
      if (LoadInst *Load = dyn_cast_or_null<LoadInst>(&Inst))
        PtrOpr = Load->getPointerOperand();
      else if (StoreInst *Store = dyn_cast_or_null<StoreInst>(&Inst))
        PtrOpr = Store->getPointerOperand();
      else
        continue;
#ifdef DEBUG_ILOOP_BARRIERS
      std::cerr << "### Analyzing ptr: ";
      PtrOpr->dump();
#endif
      if (countWorkitemIDTerms(PtrOpr, 0) > 0)
        WIAddresses++;
      else
        NonWIAddresses++;
    }
  }
#ifdef DEBUG_ILOOP_BARRIERS
  std::cerr << "### WIAddresses: " << WIAddresses
            << " NonWIAddresses: " << NonWIAddresses << std::endl;
#endif
  // TODO: Consider the iteration count. If it's low or unknown,
  // it's likely best to outer-loop parallelize anyhow.
  return WIAddresses > 0;
}

/// Adds implicit barriers to the given loop such that its body
/// will contain a parallel work-item loop after parallel region
/// formation.
///
/// If a vectorizer is applied on the result, it can produce "outer loop
/// vectorization" where the "outer loop" is considered the work-item loop
/// that by default would be the outer loop surrounding the kernel
/// "inner loop".
static bool convertToLoopWithBarriers(Loop &L) {

  llvm::BasicBlock *HeaderBlock = L.getHeader();
  llvm::Function *F = L.getHeader()->getParent();

  dumpCFG(*F, F->getName().str() + "_before_impl_loopbbarriers_on_loop_" +
                  L.getName().str() + ".dot");

  std::set<llvm::BasicBlock *> Highlights;

  // Isolate the loop body to a parallel region with two barriers.
  SmallVector<BasicBlock *> ExitingBlocks;
  L.getExitingBlocks(ExitingBlocks);
  for (BasicBlock *ExitingBlock : ExitingBlocks) {
    WorkgroupBarrier::create(ExitingBlock->getTerminator());
    Highlights.insert(ExitingBlock);
  }

  WorkgroupBarrier::create(HeaderBlock->getFirstNonPHI());

#ifdef DEBUG_ILOOP_BARRIERS
  std::cerr << "### added inner-loop barriers to loop " << L.getName().str()
            << std::endl;
  HeaderBlock->dump();
  Highlights.insert(HeaderBlock);
  dumpCFG(*F,
          F->getName().str() + "_after_impl_loopbbarriers_on_loop_" +
              L.getName().str() + ".dot",
          nullptr, nullptr, &Highlights);
#endif

  return true;
}

bool enforceOuterLoopParIfBeneficial(llvm::Function &F, llvm::LoopInfo &LI,
                                     VariableUniformityAnalysisResult &VUA) {

  if (!isKernelToProcess(F))
    return false;

#ifdef DEBUG_ILOOP_BARRIERS
  std::cerr << "### Before ImplicitLoopBarriers:\n";
  F.dump();
#endif

  bool Changed = false;
  for (llvm::Loop *OuterLoop : LI) {
    auto Loops = OuterLoop->getLoopsInPreorder();
    for (llvm::Loop *L : Loops) {

      // Only add barriers to the innermost loops.
      if (L->getSubLoops().size() > 0)
        continue;

      if (Barrier::isLoopWithBarrier(*L)) {
#ifdef DEBUG_ILOOP_BARRIERS
        std::cerr << "#### loop with barrier\n";
#endif
        continue;
      }

      if (!VUA.isUniformLoop(F, *L)) {
#ifdef DEBUG_ILOOP_BARRIERS
        std::cerr << "#### not a uniform loop\n";
#endif
        continue;
      }

      if (!pocl_get_bool_option("POCL_FORCE_PARALLEL_OUTER_LOOP", 0) &&
          !outerLoopIsLikelyBeneficial(*L)) {
#ifdef DEBUG_ILOOP_BARRIERS
        std::cerr << "#### likely better inner-loop vectorized\n";
#endif
        continue;
      }
      Changed = convertToLoopWithBarriers(*L) || Changed;
    }
  }

  if (Changed) {
#ifdef DEBUG_ILOOP_BARRIERS
    std::cerr << "### After ImplicitLoopBarriers\n";
    F.dump();
#endif
  }
  return Changed;
}

} // namespace pocl
