// LLVM function pass to canonicalize barriers.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2014 Pekka Jääskeläinen / Tampere University of Technology
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

#include <iostream>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"
#include "CanonicalizeBarriers.h"
#include "DebugHelpers.h"
#include "LLVMUtils.h"
#include "SubgroupBarrier.h"
#include "Workgroup.h"
#include "WorkgroupBarrier.h"

POP_COMPILER_DIAGS

#include <set>

#define DEBUG_TYPE "DeSPMD-CanonBAR"
// Use the LLVM_DEBUG-style macros to gradually convert to LLVM-upstreamable
// code.
#ifdef LLVM_DEBUG
#undef LLVM_DEBUG
#endif

#ifdef DEBUG_CANON_BARRIERS
#define LLVM_DEBUG(X) X
#define dbgs() std::cerr << DEBUG_TYPE << ": "
#else
#define LLVM_DEBUG(X)
#endif

namespace pocl {

using namespace llvm;

static bool isolateBarrierBlocks(Function &F);

using InstructionSet = std::set<llvm::Instruction *>;

bool canonicalizeBarriers(Function &F, llvm::LoopInfo &LI,
                          llvm::DominatorTree &DT) {

  LLVM_DEBUG(dbgs() << "Before CanonicalizeBarriers:\n");
  LLVM_DEBUG(F.dump());
  LLVM_DEBUG(llvm::verifyFunction(F));

  dumpCFG(F, "_before_canon.dot");

  bool Changed = false;

  BasicBlock *Entry = &F.getEntryBlock();
  // The function entry node should be a pure barrier at this point.
  // It should start the first parallel region.

  // Ensure the basic block before the first barrier is a forced uniform basic
  // block/ where we can push context array allocas and other code that needs to
  // be run only once per WG function.
  if (!isPureUniformBlock(Entry)) {
    SplitBlock(Entry, &(Entry->front()));
    Entry = &F.getEntryBlock();
    LLVM_DEBUG(dbgs() << "Marking function entry block as pure uniform.\n");
    markAsPureUniformBlock(Entry, "wg-function entry");
    Entry->setName("wg-func-entry");
    Changed = true;
  }

  BasicBlock *FirstPRStart = F.getEntryBlock().getSingleSuccessor();
  if (!Barrier::hasOnlyBarrier(FirstPRStart)) {
    BasicBlock *EffectiveEntry =
        SplitBlock(FirstPRStart, &(FirstPRStart->front()));

    EffectiveEntry->takeName(FirstPRStart);
    FirstPRStart->setName("entry.barrier");
    WorkgroupBarrier::createAtEnd(FirstPRStart);
    Changed = true;
    LLVM_DEBUG(
        dbgs() << "Inserted implicit entry barrier (WG) in basic block [ "
               << FirstPRStart->getName().str() << " ]\n");
  }

  // Function exits should have WG barriers.
  // Note: Even if exit has SG barrier, create new WG barrier exit.
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *BB = &*I;
    auto *T = BB->getTerminator();
    const bool IsExitNode = (T->getNumSuccessors() == 0) &&
                            (!WorkgroupBarrier::hasOnlyWGBarrier(BB));

    if (IsExitNode && !WorkgroupBarrier::hasOnlyWGBarrier(BB)) {
      // In case the bb is already terminated with a WG barrier,
      // split before the barrier so we don't create an empty
      // parallel region.
      //
      // This is because the assumptions of the other passes in the
      // compilation that are
      // a) exit node is a barrier block1
      // b) there are no empty parallel regions (which would be formed
      // between the explicit barrier and the added one). */
      /// TO CLEAN: The splitting should not be needed any more.
      BasicBlock *Exit;
      if (WorkgroupBarrier::endsWithWGBarrier(BB))
        Exit = SplitBlock(BB, T->getPrevNode());
      else
        Exit = SplitBlock(BB, T);
      Exit->setName("exit.barrier");
      WorkgroupBarrier::createAtEnd(Exit);

      LLVM_DEBUG(
          dbgs() << "Inserted implicit exit barrier (WG) in basic block [ "
                 << Exit->getName().str() << " ]\n");
      Changed = true;
    }
  }

  bool MoreChanges = false;
  do {
    MoreChanges = isolateBarrierBlocks(F);
    Changed |= MoreChanges;
  } while (MoreChanges);

  // Handling isolation barriers on uniform block relies on loop analysis.
  DT.recalculate(F);
  LI.releaseMemory();
  LI.analyze(DT);

  // Ensure regions of forced uniform blocks are isolated with a barrier
  // so they start/end parallel regions cleanly.
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *BB = &*I;
    if (isPureUniformBlock(BB)) {

      // Determine whether isolation barrier should be WG barrier or SG barrier.
      bool UseSGBarr = false;

      // Insert isolating SG barrier IF basic block is within a loop that has
      // sg-barrier(s). Otherwise, insert isolating WG barrier.
      if (llvm::Loop *L = LI.getLoopFor(BB)) {
        if (SubgroupBarrier::isLoopWithSGBarrier(*L))
          UseSGBarr = true;
      }

      for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I) {
        BasicBlock *PredBB = *I;
        if (!isPureUniformBlock(PredBB) && !Barrier::endsWithBarrier(PredBB)) {

          // Create the barrier to the beginning of the uniform block so
          // all predecessors can branch to it in case it's a join point.
          if (UseSGBarr)
            SubgroupBarrier::createAtStart(BB);
          else
            WorkgroupBarrier::createAtStart(BB);

          Changed = true;
          LLVM_DEBUG(dbgs()
                     << "Inserted implicit uniform block isolation barrier "
                     << (UseSGBarr ? "(SG)" : "(WG)")
                     << " at the start of basic block [ " << BB->getName().str()
                     << " ]\n");
          continue;
        }
      }

      for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
        BasicBlock *SuccBB = *I;
        if (!isPureUniformBlock(SuccBB) &&
            !Barrier::startsWithBarrier(SuccBB)) {
          // Create a barrier at the end of the uniform block which can then
          // potentially start multiple parallel regions.
          if (UseSGBarr)
            SubgroupBarrier::createAtEnd(BB);
          else
            WorkgroupBarrier::createAtEnd(BB);

          Changed = true;
          LLVM_DEBUG(dbgs()
                     << "Inserted implicit uniform block isolation barrier "
                     << (UseSGBarr ? "(SG)" : "(WG)")
                     << " at the end of basic block [ " << BB->getName().str()
                     << " ]\n");
          continue;
        }
      }
    }
  }

  // Prune empty regions: If there are two successive pure barrier blocks
  // without side branches, remove the other one (unless one of the blocks
  // contains a subgroup-barrier).
  bool EmptyRegionDeleted = false;
  do {
    EmptyRegionDeleted = false;
    for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
      BasicBlock *BB = &*I;
      auto *Term = BB->getTerminator();
      if (!Barrier::hasOnlyBarrier(BB) || Term->getNumSuccessors() != 1)
        continue;

      BasicBlock *Successor = Term->getSuccessor(0);

      // Skip cases where there is an SG-barrier - WG-barrier pair
      if (WorkgroupBarrier::hasWGBarrier(BB) &&
          SubgroupBarrier::hasSGBarrier(Successor))
        continue;
      if (SubgroupBarrier::hasSGBarrier(BB) &&
          WorkgroupBarrier::hasWGBarrier(Successor))
        continue;

      if (Barrier::hasOnlyBarrier(Successor) &&
          Successor->getSinglePredecessor() == BB) {
        LLVM_DEBUG(dbgs() << "Removing redundant barrier block [ "
                          << BB->getName().str() << " ]\n");
        BB->replaceAllUsesWith(Successor);
        BB->eraseFromParent();
        EmptyRegionDeleted = true;
        Changed = true;
        break;
      }
    }
  } while (EmptyRegionDeleted);

  if (Changed) {
    LLVM_DEBUG(dbgs() << "After CanonicalizeBarriers:\n");
    LLVM_DEBUG(F.dump(););
    LLVM_DEBUG(llvm::verifyFunction(F););
    dumpCFG(F, "_after_canon.dot", nullptr, nullptr);
  }

  return Changed;
}

/// Ensures all barrier calls are in their own basic blocks without any other
/// instructions than the barrier call and a branch.
///
/// \returns True in case of any changes to the function were done.
static bool isolateBarrierBlocks(Function &F) {

  bool Changed = false;

  InstructionSet Barriers;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *BB = &*I;
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      if (isa<Barrier>(I))
        Barriers.insert(&*I);
    }
  }

  for (InstructionSet::iterator I = Barriers.begin(), E = Barriers.end();
       I != E; ++I) {
    BasicBlock *BB = (*I)->getParent();

    // Split post barrier first because it does not make the barrier go to
    // another basic block.
    Instruction *Term = BB->getTerminator();

    const bool HasNonBranchInstructionsAfterBarrier = Term->getPrevNode() != *I;

    BasicBlock *PostBB = nullptr;
    if (HasNonBranchInstructionsAfterBarrier) {
      BasicBlock *NewBB = SplitBlock(BB, (*I)->getNextNode());
      NewBB->setName(BB->getName() + ".postbarrier");
      PostBB = NewBB;
      copyPureUniformMD(NewBB, BB);
      Changed = true;
    }

    BasicBlock *Predecessor = BB->getSinglePredecessor();
    if (Predecessor != NULL) {
      auto *PT = Predecessor->getTerminator();
      if ((PT->getNumSuccessors() == 1) && (&BB->front() == (*I))) {
        // Barrier is at the beginning of the BB, which has a single
        // predecessor with just one successor (the barrier itself), thus
        // no need to split before barrier.
        continue;
      }
      // This is the case where there are multiple predecessors.
    } else {
      // Skip if barrier is the first instruction of the block.
      if (&BB->front() == (*I))
        continue;
    }

    if ((BB == &(BB->getParent()->getEntryBlock())) && (&BB->front() == (*I)))
      continue;

    BasicBlock *NewBB = SplitBlock(BB, *I);
    NewBB->takeName(BB);
    BB->setName(NewBB->getName() + ".prebarrier");
    Changed = true;

    // Retain the pure uniform MD in the original basic block as the barrier
    // block splitting doesn't change that property. No need to remove the MD
    // from the barrier block as it can be treated as a pure uniform BB as well.
    copyPureUniformMD(PostBB != nullptr ? PostBB : NewBB, BB);
  }
  return Changed;
}

} // namespace pocl
