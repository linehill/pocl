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
#include "Workgroup.h"

POP_COMPILER_DIAGS

#include <set>

// #define DEBUG_CANON_BARRIERS

namespace pocl {

using namespace llvm;

static bool isolateBarrierBlocks(Function &F);

using InstructionSet = std::set<llvm::Instruction *>;

bool canonicalizeBarriers(Function &F) {

#ifdef DEBUG_CANON_BARRIERS
  std::cerr << "Before CanonicalizeBarriers:\n";
  F.dump();
  llvm::verifyFunction(F);
#endif

  dumpCFG(F, F.getName().str() + "_before_canon.dot");

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
    Barrier::createAtEnd(FirstPRStart);
    Changed = true;
  }

  // Function exits should have barriers.
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *BB = &*I;
    auto *T = BB->getTerminator();
    const bool IsExitNode =
        (T->getNumSuccessors() == 0) && (!Barrier::hasOnlyBarrier(BB));

    if (IsExitNode && !Barrier::hasOnlyBarrier(BB)) {
      // In case the bb is already terminated with a barrier,
      // split before the barrier so we don't create an empty
      // parallel region.
      //
      // This is because the assumptions of the other passes in the
      // compilation that are
      // a) exit node is a barrier block
      // b) there are no empty parallel regions (which would be formed
      // between the explicit barrier and the added one). */
      /// TO CLEAN: The splitting should not be needed any more.
#ifdef DEBUG_CANON_BARRIERS
      std::cerr << "CanonBar: isExitNode & !hasOnlyBarrier\n";
#endif
      BasicBlock *Exit;
      if (Barrier::endsWithBarrier(BB))
        Exit = SplitBlock(BB, T->getPrevNode());
      else
        Exit = SplitBlock(BB, T);
      Exit->setName("exit.barrier");
      Barrier::create(Inst2InsertPt(t));
      Changed = true;
    }
  }

  bool MoreChanges = false;
  do {
    MoreChanges = isolateBarrierBlocks(F);
    Changed |= MoreChanges;
  } while (MoreChanges);

  // Ensure regions of forced uniform blocks are isolated with a barrier
  // so they start/end parallel regions cleanly.
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *BB = &*I;
    if (isPureUniformBlock(BB)) {
      for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I) {
        BasicBlock *PredBB = *I;
        if (!isPureUniformBlock(PredBB) && !Barrier::endsWithBarrier(PredBB)) {
          // Create the barrier to the beginning of the uniform block so
          // all predecessors can branch to it in case it's a join point.
          Barrier::create(BB->getFirstNonPHI());
          Changed = true;
          continue;
        }
      }

      for (succ_iterator I = succ_begin(BB), E = succ_end(BB); I != E; ++I) {
        BasicBlock *SuccBB = *I;
        if (!isPureUniformBlock(SuccBB) &&
            !Barrier::startsWithBarrier(SuccBB)) {
          // Create a barrier at the end of the uniform block which can then
          // potentially start multiple parallel regions.
          Barrier::create(BB->getTerminator());
          Changed = true;
          continue;
        }
      }
    }
  }

  // Prune empty regions: If there are two successive pure barrier blocks
  // without side branches, remove the other one.
  bool EmptyRegionDeleted = false;
  do {
    EmptyRegionDeleted = false;
    for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
      BasicBlock *BB = &*I;
      auto *Term = BB->getTerminator();
      if (!Barrier::hasOnlyBarrier(BB) || Term->getNumSuccessors() != 1)
        continue;

      BasicBlock *Successor = Term->getSuccessor(0);

      if (Barrier::hasOnlyBarrier(Successor) &&
          Successor->getSinglePredecessor() == BB) {
        BB->replaceAllUsesWith(Successor);
        BB->eraseFromParent();
        EmptyRegionDeleted = true;
        Changed = true;
        break;
      }
    }
  } while (EmptyRegionDeleted);

  if (Changed) {
#ifdef DEBUG_CANON_BARRIERS
    std::cerr << "After CanonicalizeBarriers:\n";
    F.dump();
    llvm::verifyFunction(F);
#endif
    dumpCFG(F, F.getName().str() + "_after_canon.dot", nullptr, nullptr);
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
