// Addition of implicit barriers to isolate loops for clean and correct
// parallel regions.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
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
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>

#include "Barrier.h"
#include "DebugHelpers.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "LoopBarriers.h"
#include "SubgroupBarrier.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkgroupBarrier.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>

#define DEBUG_TYPE "DeSPMD-LBAR"


// Use the LLVM_DEBUG-style macros to gradually convert to LLVM-upstreamable
// code.
#ifdef LLVM_DEBUG
#undef LLVM_DEBUG
#endif

#ifdef DEBUG_LOOP_BARRIERS
#define LLVM_DEBUG(X) X
#define dbgs() std::cerr << DEBUG_TYPE << ": "
#else
#define LLVM_DEBUG(X)
#endif

namespace pocl {

using namespace llvm;

/// Finds the basic block in a loop that contains the loop condition check.
/// \return nullptr if unable to analyze the loop.
static BasicBlock *getConditionCheckBlock(Loop &L,
                                          ICmpInst **CondCmpI = nullptr) {

  BasicBlock *CondComp = nullptr;
  if (BasicBlock *Exit = L.getExitingBlock()) {
    CondComp =
        Barrier::hasOnlyBarrier(Exit) ? Exit->getSinglePredecessor() : Exit;
  } else if (BasicBlock *Header = L.getHeader()) {
    CondComp = Header;

    // Due to the barrier transformation, the implicit barriers might have
    // pushed the condition check block forward. Find it.
    while (Barrier::hasOnlyBarrier(CondComp) ||
           isa<BranchInst>(CondComp->begin())) {
      CondComp = CondComp->getSingleSuccessor();
    }
  } else {
    return nullptr;
  }

  if (CondComp != nullptr) {

    ICmpInst *Temp;
    if (CondCmpI == nullptr)
      CondCmpI = &Temp;

#if LLVM_MAJOR < 22
    auto *Instr = CondComp->getTerminator()->getPrevNonDebugInstruction();
#else
    auto *Instr = CondComp->getTerminator()->getPrevNode();
#endif

    if (Instr == nullptr) {
      // A basic block with only a branch in the end.
      return nullptr;
    }

    *CondCmpI = dyn_cast_or_null<ICmpInst>(Instr);
    if (*CondCmpI != nullptr)
      return CondComp;

    // We might have added implicit barriers to the block. They should be fine
    // in the condition check block of a b-loop. Just skip them.
    if (isa<Barrier>(Instr)) {
#if LLVM_MAJOR < 22
      Instr = Instr->getPrevNonDebugInstruction();
#else
      Instr = Instr->getPrevNode();
#endif
      *CondCmpI = dyn_cast_or_null<ICmpInst>(Instr);
      if (*CondCmpI != nullptr)
        return CondComp;
    }
  }
  return nullptr;
}

/// Splits uniform loop count instructions after possible diverging instructions
/// in the given \p Latch. \return True in case the Latch was modified.
static bool isolateUniformLatch(BasicBlock *Latch,
                                VariableUniformityAnalysisResult &VUA) {

  // The first instruction in the new Latch basic block.
  auto *SplitPoint = Latch->getTerminator();
  do {
#if LLVM_MAJOR < 22
    Instruction *Prev = SplitPoint->getPrevNonDebugInstruction(true);
#else
    Instruction *Prev = SplitPoint->getPrevNode();
#endif
    if (Prev == nullptr || !VUA.isUniform(Latch->getParent(), Prev))
      break;
    SplitPoint = Prev;
  } while (true);

  // Do not create basic blocks with only the branch.
  if (SplitPoint == Latch->getTerminator() ||
#if LLVM_MAJOR < 22
      SplitPoint->getNextNonDebugInstruction(true) == Latch->getTerminator())
#else
      SplitPoint->getNextNode() == Latch->getTerminator())
#endif
    return false;

  BasicBlock *NewLatch = SplitBlock(Latch, SplitPoint);
  NewLatch->setName(Latch->getName() + ".uniform");

  return true;
}

/// Returns true in case \p L is an ideal/canonical loop that only contains
/// iteration variable increment/comparison instructions in its loop structure
/// basic blocks, thus is suitable for cross-WI b-loop structure sharing.
static bool
isSuitableForBLoopStructureSharing(Loop &L,
                                   VariableUniformityAnalysisResult &VUA) {

  // TODO: Expand the coverage of loop cases incrementally.
  // Currently assumes unoptimized non-SSA (no PHIs) input.

  ICmpInst *CondCmpI = nullptr;
  BasicBlock *CondComp = getConditionCheckBlock(L, &CondCmpI);

  BasicBlock *Latch = L.getLoopLatch();
  BasicBlock *Header = L.getHeader();

  if (CondComp == Latch || CondComp == nullptr || Latch == nullptr ||
      VUA.hasDivergingInstructions(*CondComp) ||
      VUA.hasDivergingInstructions(*Latch)) {
    LLVM_DEBUG(dbgs() << "Not suitable for loop structure sharing:\n");
    LLVM_DEBUG(
        dbgs() << "CondComp: "
               << (CondComp != nullptr ? CondComp->getName().str() : "null")
               << "\n");
    LLVM_DEBUG(dbgs() << "Latch: "
                      << (Latch != nullptr ? Latch->getName().str() : "null")
                      << "\n");
    LLVM_DEBUG(dbgs() << "Header: "
                      << (Header != nullptr ? Header->getName().str() : "null")
                      << "\n");
    LLVM_DEBUG(
        if (CondComp != nullptr && VUA.hasDivergingInstructions(*CondComp)) {
          dbgs() << "CondComp has diverging instructions:\n";
          CondComp->dump();
        });
    LLVM_DEBUG(if (Latch != nullptr && VUA.hasDivergingInstructions(*Latch)) {
      dbgs() << "Latch has diverging instructions:\n";
      Latch->dump();
    });
    return false;
  }

  if (CondCmpI == nullptr)
    return false;

  Value *CCLeft = CondCmpI->getOperand(0);
  if (LoadInst *Load = dyn_cast_or_null<LoadInst>(CCLeft))
    CCLeft = Load->getPointerOperand();

  Value *CCRight = CondCmpI->getOperand(0);
  if (LoadInst *Load = dyn_cast_or_null<LoadInst>(CCRight))
    CCRight = Load->getPointerOperand();

  Value *Iterator = nullptr;
  if (CCLeft->isUsedInBasicBlock(Latch)) {
    // The latch should only increment the iterator in our case.
    Iterator = CCLeft;
  } else {
    Iterator = CCRight;
  }

  // The for-loop case with only the iteration variable increment in the
  // latch and a condition check in another.
  if (CondComp != nullptr && Latch != nullptr) {
    size_t InstructionsToAnalyze = std::distance(Latch->begin(), Latch->end());
    LLVM_DEBUG(dbgs() << "Analyzing latch for loop construct sharing:\n");
    LLVM_DEBUG(Latch->dump());

    auto I = Latch->begin();
    LoadInst *IteratorLoad = nullptr;

    if (InstructionsToAnalyze == 4) {
      // Iterator load in the increment block?
      if (!(IteratorLoad = dyn_cast<LoadInst>(I++)))
        return false;
      InstructionsToAnalyze--;
    }

    if (InstructionsToAnalyze != 3)
      return false;

    BinaryOperator *Modify = nullptr;
    if ((Modify = dyn_cast_or_null<BinaryOperator>(I++))) {

      if (IteratorLoad == nullptr)
        IteratorLoad = dyn_cast<LoadInst>(Modify->getOperand(0));

      if (IteratorLoad == nullptr || Modify->getOperand(0) != IteratorLoad ||
          !isa<Constant>(Modify->getOperand(1)) ||
          IteratorLoad->getPointerOperand() != Iterator)
        return false;
    } else
      return false;

    StoreInst *Store = nullptr;
    if ((Store = dyn_cast_or_null<StoreInst>(I++))) {
      if (Store->getOperand(0) != Modify || Store->getOperand(1) != Iterator)
        return false;
    } else
      return false;

    return isa<BranchInst>(I);
  }
  return false;
}

static bool processLoopWithBarriers(Loop &L, llvm::DominatorTree &DT,
                                    VariableUniformityAnalysisResult &VUA) {

  llvm::Function *K = L.getLoopPreheader()->getParent();

  std::set<llvm::BasicBlock *> Highlights;
  /*   dumpCFG(*K, "_before_loopbbarriers_on_bloop_" +
                    L.getName().str() + ".dot");

    LLVM_DEBUG(dbgs() << "Loop: " << L.getName().str() << "\n"); */

  // TO clean: This loop construct is not necessary here anymore,
  // as the b-loop property is detected earlier.
  for (Loop::block_iterator I = L.block_begin(), E = L.block_end(); I != E;
       ++I) {
    for (BasicBlock::iterator J = (*I)->begin(), E = (*I)->end(); J != E; ++J) {
      if (isa<Barrier>(J)) {

        // Found a barrier in this loop:
        // 1) add a barrier in the loop header.
        // 2) add a barrier in the latches

        // Add a barrier on the preheader to ensure all WIs reach
        // the loop header with all the previous code already
        // executed.
        BasicBlock *Preheader = L.getLoopPreheader();
        assert((Preheader != NULL) && "Non-canonicalized loop found!\n");

        LLVM_DEBUG(dbgs() << "adding to preheader BB\n");
        LLVM_DEBUG(Preheader->dump());
        LLVM_DEBUG(dbgs() << "before instr\n");
        LLVM_DEBUG(Preheader->getTerminator()->dump());

        // Add appropriate barrier to the loop preheader.
        // TODO: optimization opportunity by changing sg-barriers to wg-barriers
        // in uniform loop.
        if (SubgroupBarrier::isLoopWithSGBarrier(L))
          // Switch the barrier type if there is WG barrier already.
          if (WorkgroupBarrier::hasWGBarrier(Preheader))
            switchBarrierGranularity(Preheader);
          else
            SubgroupBarrier::createAtEnd(Preheader);
        else
          WorkgroupBarrier::createAtEnd(Preheader);

        Preheader->setName(Preheader->getName() + ".loopbarrier");
        Highlights.insert(Preheader);

        // Add a barrier after the PHI nodes on the header (the replicated
        // headers will be merged afterwards).
        BasicBlock *Header = L.getHeader();
#if LLVM_MAJOR < 20
        if (Header->getFirstNonPHI() != &Header->front()) {
#else
        if (Header->getFirstNonPHIIt() != Header->begin()) {
#endif
          if (SubgroupBarrier::isLoopWithSGBarrier(L))
            SubgroupBarrier::createAtStart(Header);
          else
            WorkgroupBarrier::createAtStart(Header);

          Header->setName(Header->getName() + ".phibarrier");
          Highlights.insert(Header);
        }

        BasicBlock *CondBlock = getConditionCheckBlock(L);

        // Add barriers on the exiting block and the latches,
        // which might not always be the same if there is computation
        // after the exit decision.
        BasicBlock *BrExit = L.getExitingBlock();
        if (BrExit != NULL) {

          if (SubgroupBarrier::isLoopWithSGBarrier(L))
            SubgroupBarrier::createAtEnd(BrExit);
          else
            WorkgroupBarrier::createAtEnd(BrExit);

          BrExit->setName(BrExit->getName() + ".brexitbarrier");
          Highlights.insert(BrExit);
        }

        BasicBlock *Latch = L.getLoopLatch();
        // Check if we can share the loop construct (the iteration
        // variable and the code that manages it) across the work-items,
        // like is usually the case with loops containing barrier calls.
        // If we can share the iteration variable without storing it in the
        // context, it helps the loop vectorizer a lot when analyzing the
        // memory access patterns.
        bool UniformLoopConstruct = isSuitableForBLoopStructureSharing(L, VUA);

        if (Latch != NULL && BrExit != Latch) {
          if (SubgroupBarrier::isLoopWithSGBarrier(L))
            SubgroupBarrier::createAtEnd(Latch);
          else
            WorkgroupBarrier::createAtEnd(Latch);

          Latch->setName(Latch->getName() + ".latchbarrier");
        }

        if (UniformLoopConstruct) {
          markAsPureUniformBlock(CondBlock, "b-loop condition check");
          LLVM_DEBUG(dbgs()
                     << "Marked [" << CondBlock->getName().str()
                     << "] as pure uniform block: b-loop condition check\n");
          Highlights.insert(CondBlock);
          if (Latch != nullptr) {
            // Only a single latch.
            markAsPureUniformBlock(Latch, "b-loop latch");
            LLVM_DEBUG(dbgs() << "Marked [" << Latch->getName().str()
                              << "] as pure uniform block: b-loop latch\n");
            Highlights.insert(Latch);
          }
        }

        if (Latch != nullptr) {
          // Single latch case.
          /*  dumpCFG(*K,
                   "_after_loopbbarriers_on_bloop_" +
                       L.getName().str() + ".dot",
                   nullptr, nullptr, &Highlights); */
          return true;
        }

        // Go through all the latches ('continues').
        BasicBlock *Header2 = L.getHeader();
        typedef GraphTraits<Inverse<BasicBlock *> > InvBlockTraits;
        InvBlockTraits::ChildIteratorType PI =
          InvBlockTraits::child_begin(Header2);
        InvBlockTraits::ChildIteratorType PE =
          InvBlockTraits::child_end(Header2);

        BasicBlock *Latch2 = nullptr;
        for (; PI != PE; ++PI) {
          BasicBlock *N = *PI;
          if (L.contains(N)) {
            Latch2 = N;
            // Latch found in the loop, see if the barrier dominates it
            // (otherwise if might not even belong to this "tail", see
            // forifbarrier1 graph test).
            if (DT.dominates(J->getParent(), Latch2)) {

              if (SubgroupBarrier::isLoopWithSGBarrier(L))
                SubgroupBarrier::createAtEnd(Latch);
              else
                WorkgroupBarrier::createAtEnd(Latch);

              if (UniformLoopConstruct) {
                markAsPureUniformBlock(Latch2, "b-loop latch");
                LLVM_DEBUG(dbgs() << "Marked [" << Latch2->getName().str()
                                  << "] as pure uniform block: b-loop latch\n");
              }
              Highlights.insert(Latch2);
            }
          }
        }
        return true;
      }
    }
  }
  return false;
}

static bool processLoop(Loop &L, llvm::DominatorTree &DT,
                        VariableUniformityAnalysisResult &VUA) {

  if (Barrier::isLoopWithBarrier(L)) {
    LLVM_DEBUG(dbgs() << "loopbarriers: loop with barrier\n" );
    return processLoopWithBarriers(L, DT, VUA);
  }

  // This is a loop without a barrier. Ensure we have a non-barrier
  // block as a preheader so we can capture the loop as a whole
  // to the parallel region.
  //
  // If the block has proper instructions after the barrier, it
  // will be split in CanonicalizeBarriers.
  //
  // Also attempt to isolate the loop with barriers to create the
  // WI loop around it in order to produce a nicely well-formed
  // WI-loop + K-loop hierarchy for the loop-interchange and loop
  // vectorizer to optimize.

  // Include all the layers in a multi-level loop without barriers
  // in the region. Thus, do the handling only for the outermost
  // loop without barriers. Otherwise we end up creating a b-loop
  // to the innermost loop if we process that first, ruining the
  // idea for multi-level loops.
  Loop *ParentLoop = L.getParentLoop();
  if (!(ParentLoop == nullptr || Barrier::isLoopWithBarrier(*ParentLoop)))
    return false;

  BasicBlock *Preheader = L.getLoopPreheader();
  assert((Preheader != NULL) && "Non-canonicalized loop found!\n");

  Instruction *Inst = Preheader->getTerminator();
  Instruction *PrevInst = NULL;
  if (&Preheader->front() != Inst)
    PrevInst = Inst->getPrevNode();
  if (isa_and_nonnull<Barrier>(PrevInst)) {
    BasicBlock *NewBB = SplitBlock(Preheader, Inst);
    NewBB->setName(Preheader->getName() + ".postbarrier_dummy");
    return true;
  }

  return false;
}

bool addLoopConstructIsolationBarriers(llvm::Function &F, llvm::LoopInfo &LI,
                                       VariableUniformityAnalysisResult &VUA,
                                       llvm::DominatorTree &DT) {

  if (!isKernelToProcess(F))
    return false;

  if (!hasWorkgroupBarriers(F))
    return false;

  LLVM_DEBUG(dbgs() << "Before LoopBarriers\n");
  LLVM_DEBUG(F.dump());

  // Prettify the loop structure blocks to make them suitable for loop
  // construct sharing etc.
  bool Changed = false;

  for (llvm::Loop *OuterLoop : LI) {
    auto Loops = OuterLoop->getLoopsInPreorder();
    for (llvm::Loop *L : Loops) {
      BasicBlock *Latch = L->getLoopLatch();
      if (Latch != nullptr)
        Changed = isolateUniformLatch(Latch, VUA) || Changed;
    }
  }

  if (Changed) {
    DT.recalculate(F);
    LI.releaseMemory();
    LI.analyze(DT);
    LI.verify(DT);
  }

  for (llvm::Loop *OuterLoop : LI) {
    auto Loops = OuterLoop->getLoopsInPreorder();
    for (llvm::Loop *L : Loops)
      Changed = processLoop(*L, DT, VUA) || Changed;
  }

  LLVM_DEBUG(dbgs() << "After LoopBarriers:\n");
  LLVM_DEBUG(F.dump());
  return Changed;
}

} // namespace pocl
