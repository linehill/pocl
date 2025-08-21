// Adds implicit barriers to branches where required and seen beneficial.
//
// Copyright (c) 2013 Pekka Jääskeläinen / TUT
//               2024-2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"
#include "CanonicalizeBarriers.h"
#include "DebugHelpers.h"
#include "ImplicitConditionalBarriers.h"
#include "ImplicitLoopBarriers.h"
#include "LLVMUtils.h"
#include "SubgroupBarrier.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkgroupBarrier.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>

#include "pocl.h"

#define DEBUG_TYPE "DeSPMD-ICB"

#ifdef ENABLE_DEBUG
#undef ENABLE_DEBUG
#endif

#ifdef DEBUG_COND_BARRIERS
#define ENABLE_DEBUG
#endif

#include "TemporaryLLVMDebugMacros.hh"

namespace pocl {

using namespace llvm;

/// Finds a predecessor basic block for \p BB that does not originate from
/// a back edge.
///
/// This is used to include loops in the conditional parallel region.
static BasicBlock *firstNonBackedgePredecessor(llvm::BasicBlock *BB,
                                               DominatorTree &DT) {

  pred_iterator I = pred_begin(BB), E = pred_end(BB);
  while (I != E && DT.dominates(BB, *I))
    ++I;
  if (I == E)
    return NULL;
  return *I;
}

bool addImplicitBranchBarriers(llvm::Function &F, llvm::LoopInfo &LI,
                               pocl::VariableUniformityAnalysisResult &VUA,
                               llvm::PostDominatorTree &PDT,
                               llvm::DominatorTree &DT) {

  if (!isKernelToProcess(F))
    return false;

  if (!hasWorkgroupBarriers(F))
    return false;

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_before_implicit_cond_barriers.dot", nullptr,
          nullptr);
#endif

  typedef std::vector<BasicBlock*> BarrierBlockIndex;
  BarrierBlockIndex ConditionalBarriers;

  bool Changed = false;

  for (BasicBlock &BB : F) {
    if (!Barrier::hasBarrier(&BB))
      continue;

    // Unconditional barrier postdominates the entry node.
    if (PDT.dominates(&BB, &F.getEntryBlock())) {
      LLVM_DEBUG(dbgs() << "BB postdominates the entry block\n");
      LLVM_DEBUG(BB.dump());
      continue;
    }
    ConditionalBarriers.push_back(&BB);
  }

  for (BasicBlock *BB : ConditionalBarriers) {
    LLVM_DEBUG(dbgs() << "Handling a conditional barrier in basic block:\n");
    LLVM_DEBUG(BB->dump());

    // Trace upwards from the barrier until one encounters another
    // barrier or the split point that makes the barrier conditional.
    // In case of the latter, add a new barrier to both branches of the split
    // point.

    // BB before which to inject the barrier.
    BasicBlock *Pos = BB;
    if (pred_begin(BB) == pred_end(BB)) {
      LLVM_DEBUG(dbgs() << "BB before which to inject the barrier:\n");
      LLVM_DEBUG(BB->dump());

      assert (pred_begin(BB) == pred_end(BB));
    }
    BasicBlock *Pred = firstNonBackedgePredecessor(BB, DT);

    while (!Barrier::hasOnlyBarrier(Pred) && PDT.dominates(BB, Pred)) {

      LLVM_DEBUG(dbgs() << "Looking at BB " << Pred->getName().str() << "\n");

      Pos = Pred;
      // If our BB post dominates the given block, we know it is not the
      // branching block that makes the barrier conditional.
      Pred = firstNonBackedgePredecessor(Pred, DT);

      if (Pred == BB) break; // Traced across a loop edge, skip this case.
    }

    if (Barrier::hasOnlyBarrier(Pos)) continue;

    Changed = true;

    if (BasicBlock *Source = Pos->getSinglePredecessor()) {

      if (!Barrier::hasBarrier(Source)) {
        if (WorkgroupBarrier::hasWGBarrier(BB))
          WorkgroupBarrier::createAtEnd(Source);
        else
          SubgroupBarrier::createAtEnd(Source);
      }

      // Add implicit barrier to merge block as well.
      BasicBlock *MergeBlock = PDT.getNode(Source)->getIDom()->getBlock();
      if (MergeBlock) {

        if (!Barrier::hasBarrier(MergeBlock)) {

          if (WorkgroupBarrier::hasWGBarrier(BB))
            WorkgroupBarrier::createAtStart(MergeBlock);
          else
            SubgroupBarrier::createAtStart(MergeBlock);

          // If the conditional barrier is an SG-barrier, but the merge block
          // already contains a WG-barrier, change it to an SG-barrier.
        } else if (SubgroupBarrier::hasSGBarrier(BB) &&
                   WorkgroupBarrier::hasWGBarrier(MergeBlock)) {

          switchBarrierGranularity(MergeBlock);
        }
      }
    }
  }

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_after_implicit_cond_barriers.dot", nullptr,
          nullptr);
#endif
  if (Changed) {
    LLVM_DEBUG(dbgs() << "After ImplicitConditionalBarriers\n");
    LLVM_DEBUG(F.dump());
    LLVM_DEBUG(dumpCFG(F, F.getName().str() + "_after_cond_barriers.dot",
                       nullptr, nullptr));
  }
  return Changed;
}

} // namespace pocl
