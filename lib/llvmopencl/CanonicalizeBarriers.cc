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
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"
#include "CanonicalizeBarriers.h"
#include "DebugHelpers.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"

POP_COMPILER_DIAGS

#include <set>

#define PASS_NAME "canon-barriers"
#define PASS_CLASS pocl::CanonicalizeBarriers
#define PASS_DESC "Barrier canonicalization pass"

// #define DEBUG_CANON_BARRIERS

namespace pocl {

using namespace llvm;

bool isolateBarrierBlocks(Function &F, WorkitemHandlerType Handler);

using InstructionSet = std::set<llvm::Instruction *>;

/// Ensures all barrier calls are in their own basic blocks without any other
/// instructions than the barrier call and a branch.
///
/// Also isolates the entry and exit nodes of the functions as well
/// as regions of pure uniform basic blocks.
bool canonicalizeBarriers(Function &F, WorkitemHandlerType Handler) {

#ifdef DEBUG_CANON_BARRIERS
  std::cerr << "Before CanonicalizeBarriers:\n";
  F.dump();
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
  }

  BasicBlock *FirstPRStart = F.getEntryBlock().getSingleSuccessor();
  if (!Barrier::hasOnlyBarrier(FirstPRStart)) {
    BasicBlock *EffectiveEntry =
        SplitBlock(FirstPRStart, &(FirstPRStart->front()));

    EffectiveEntry->takeName(Entry);
    Entry->setName("entry.barrier");
    Barrier::createAtEnd(Entry);
    changed |= true;
  }

  // Function exits should have barriers.
  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
    BasicBlock *BB = &*i;
    auto t = BB->getTerminator();
    const bool isExitNode =
      (t->getNumSuccessors() == 0) && (!Barrier::hasOnlyBarrier(BB));

    if (isExitNode && !Barrier::hasOnlyBarrier(BB)) {
      /* In case the bb is already terminated with a barrier,
         split before the barrier so we don't create an empty
         parallel region.

         This is because the assumptions of the other passes in the
         compilation that are
         a) exit node is a barrier block
         b) there are no empty parallel regions (which would be formed
         between the explicit barrier and the added one). */
      /// TO CLEAN: The splitting should not be needed any more.
#ifdef DEBUG_CANON_BARRIERS
      std::cerr << "CanonBar: isExitNode & !hasOnlyBarrier\n";
#endif
      BasicBlock *exit;
      if (Barrier::endsWithBarrier(BB))
        exit = SplitBlock(BB, t->getPrevNode());
      else
        exit = SplitBlock(BB, t);
      exit->setName("exit.barrier");
      Barrier::create(Inst2InsertPt(t));
      Changed |= true;
    }
  }

  bool MoreChanges = false;
  do {
    MoreChanges = isolateBarrierBlocks(F, Handler);
    Changed |= MoreChanges;
  } while (MoreChanges);

  // Ensure regions of forced uniform blocks are isolated with a barrier
  // so they start/end parallel regions cleanly.
  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
    BasicBlock *BB = &*i;
    if (isPureUniformBlock(BB)) {
      for (pred_iterator i = pred_begin(BB), e = pred_end(BB); i != e; ++i) {
        BasicBlock *PredBB = *i;
        if (!isPureUniformBlock(PredBB) && !Barrier::endsWithBarrier(PredBB)) {
          // Create the barrier to the beginning of the uniform block so
          // all predecessors can branch to it in case it's a join point.
          Barrier::create(BB->getFirstNonPHI());
          continue;
        }
      }

      for (succ_iterator i = succ_begin(BB), e = succ_end(BB); i != e; ++i) {
        BasicBlock *SuccBB = *i;
        if (!isPureUniformBlock(SuccBB) &&
            !Barrier::startsWithBarrier(SuccBB)) {
          // Create a barrier at the end of the uniform block which can then
          // potentially start multiple parallel regions.
          Barrier::create(BB->getTerminator());
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
    for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
      BasicBlock *BB = &*i;
      auto Term = BB->getTerminator();
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
#endif
    dumpCFG(F, F.getName().str() + "_after_canon.dot", nullptr, nullptr);
  }

  return Changed;
}

/// Ensures all barrier calls are in their own basic blocks without any other
/// instructions than the barrier call and a branch.
///
/// \returns True in case of any changes to the function were done.
bool isolateBarrierBlocks(Function &F, WorkitemHandlerType Handler) {

  bool Changed = false;

  InstructionSet Barriers;
  for (Function::iterator i = F.begin(), e = F.end();
       i != e; ++i) {
    BasicBlock *BB = &*i;
    for (BasicBlock::iterator i = BB->begin(), e = BB->end(); i != e; ++i) {
      if (isa<Barrier>(i))
        Barriers.insert(&*i);
    }
  }

  for (InstructionSet::iterator i = Barriers.begin(), e = Barriers.end();
       i != e; ++i) {
    BasicBlock *BB = (*i)->getParent();

    // Split post barrier first because it does not make the barrier go to
    // another basic block.
    Instruction *Term = BB->getTerminator();

    const bool HasNonBranchInstructionsAfterBarrier =
        Term->getPrevNode() != *i ||
        (Handler == WorkitemHandlerType::CBS && Term->getNumSuccessors() > 1);

    BasicBlock *PostBB = nullptr;
    if (HasNonBranchInstructionsAfterBarrier) {
      BasicBlock *NewBB = SplitBlock(BB, (*i)->getNextNode());
      NewBB->setName(BB->getName() + ".postbarrier");
      Changed = true;
      PostBB = NewBB;
      copyPureUniformMD(NewBB, BB);
    }

    BasicBlock *Predecessor = BB->getSinglePredecessor();
    if (Predecessor != NULL) {
      auto PT = Predecessor->getTerminator();
      if ((PT->getNumSuccessors() == 1) && (&BB->front() == (*i))) {
        // Barrier is at the beginning of the BB, which has a single
        // predecessor with just one successor (the barrier itself), thus
        // no need to split before barrier.
        continue;
      }
    }
    if ((BB == &(BB->getParent()->getEntryBlock())) && (&BB->front() == (*i)))
      continue;

    BasicBlock *NewBB = SplitBlock(BB, *i);
    NewBB->takeName(BB);
    BB->setName(NewBB->getName() + ".prebarrier");
    Changed = true;

    // Retain the pure uniform MD in the original basic block as the barrier
    // block splitting doesn't change that property. No need to remove the MD
    // from the barrier block as it can be treated as a pure uniform BB as well.
    dumpCFG(F, F.getName().str() + "_before_copy.dot", nullptr, nullptr);
    copyPureUniformMD(PostBB != nullptr ? PostBB : NewBB, BB);

#if 0
    // Tässä logiikassa jotain vikaa. Meidän pitäisi hanskata tämä tapaus
    // aiemmin. Jos barrier ei aloita basic blockia, se pitää splitata. Nyt
    // splitataan vain jos se ei lopeta basic blockia?
    if (!Barrier::startsWithBarrier(BB)) {
      // Create a barrier at the end of the uniform block which can be a join
      // point.

      // TODO: JOS tässä PHInode alussa, tulee epäkanonikaalinen barrier, eli
      // joudutaan ajamaan uusiksi.
      Barrier::create(BB->getFirstNonPHI());
    }
#endif

    dumpCFG(F, F.getName().str() + "_after_copy.dot", nullptr, nullptr);
  }
  return Changed;
}

llvm::PreservedAnalyses
CanonicalizeBarriers::run(llvm::Function &F,
                          llvm::FunctionAnalysisManager &AM) {
  if (!pocl::isKernelToProcess(F))
    return PreservedAnalyses::all();
  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return canonicalizeBarriers(F, WIH) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
