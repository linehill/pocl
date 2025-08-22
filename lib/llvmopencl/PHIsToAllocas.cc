// LLVM function to convert all PHIs to alloca reads.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
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
#include <llvm/ADT/Twine.h>
#include <llvm/Analysis/PostDominators.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/IRBuilder.h>

#include "Barrier.h"
#include "LLVMUtils.h"
#include "PHIsToAllocas.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemLoops.h"
POP_COMPILER_DIAGS

#include <iostream>

//#define DEBUG_PHIS_TO_ALLOCAS

// Use the LLVM_DEBUG-style macros to gradually convert to LLVM-upstreamable
// code.
#ifdef LLVM_DEBUG
#undef LLVM_DEBUG
#endif

#define DEBUG_TYPE "DeSPMD-PTA"

#ifdef DEBUG_PHIS_TO_ALLOCAS
#define LLVM_DEBUG(X) X
#define dbgs() std::cerr << DEBUG_TYPE << ": "
#else
#define LLVM_DEBUG(X)
#endif

namespace pocl {

using namespace llvm;


/// An in-depth check to evaluate whether sinking can be done.
///
/// We have to evaluate the domination relationship of the Users
/// and inserted loads.
/// \param Phi the PHI instruction to consider.
/// \param Users the users of the PHI node.
bool isFeasibleToSink(llvm::BasicBlock *PhiBlock, std::vector<llvm::Value *> Users, llvm::DominatorTree &DT) {

  bool ShouldSink = true;

  // Check that every basic block containing user instruction is dominated by
  // some potential load basic block.
  for (auto &Usr : Users) {

    llvm::Instruction *UserInstr = dyn_cast<Instruction>(Usr);

    bool FoundDominatingBlock = false;

    for (BasicBlock *PhiSucc : successors(PhiBlock)) {
      // For a PHI user, check the dominance in the context of incoming
      // blocks instead.
      if (llvm::isa<llvm::PHINode>(UserInstr)) {
        llvm::PHINode *PhiUser = dyn_cast<PHINode>(UserInstr);

        for (int I = 0; I < PhiUser->getNumIncomingValues(); ++I) {
          BasicBlock *IncomingBB = PhiUser->getIncomingBlock(I);

          if (DT.dominates(PhiSucc, IncomingBB))
            FoundDominatingBlock = true;
        }
        // For a Non-phi user:
      } else {
        if (DT.dominates(PhiSucc, UserInstr->getParent()))
          FoundDominatingBlock = true;
      }
    }
    if(!FoundDominatingBlock)
      ShouldSink = false;
  }
  return ShouldSink;
}

bool convertPHIsToAllocaAccesses(llvm::Function &F, llvm::DominatorTree &DT) {

  LLVM_DEBUG(dbgs() << "Before PHIsToAllocas\n");
  LLVM_DEBUG(F.dump());

  std::vector<PHINode *> PHIs;
  for (auto &BB : F) {
    for (auto &I : BB) {
      PHINode *PHI = dyn_cast_or_null<PHINode>(&I);
      if (PHI != nullptr)
        PHIs.push_back(PHI);
    }
  }

  for (PHINode *Phi : PHIs) {

    std::string AllocaName = std::string(Phi->getName().str()) + ".ex_phi";

    llvm::Function *Function = Phi->getParent()->getParent();

    IRBuilder<> Builder(&*(Function->getEntryBlock().getFirstInsertionPt()));

    llvm::Instruction *AllocaI =
        Builder.CreateAlloca(Phi->getType(), 0, AllocaName);

    // If any of the PHI "update stores" are in the same basic block as
    // the PHI itself.
    bool StoreInPhiBlock = false;

    for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues();
         ++Incoming) {
      Value *Val = Phi->getIncomingValue(Incoming);
      BasicBlock *IncomingBB = Phi->getIncomingBlock(Incoming);

      if (IncomingBB == Phi->getParent())
        StoreInPhiBlock = true;

      // Push the value update higher up in the basic block, just after the
      // producer (or the beginning of the BB) to avoid fuzzying loop structure
      // instructions. We want to leave the increment (and its store) of the
      // loop counter as the last instructions in the basic block to enable
      // splitting the basic block for divergent and uniform parts for barrier
      // loop construct sharing (see LoopBarriers.cc).
      Instruction *Pos = IncomingBB->getTerminator();
      do {
        Instruction *Prev = Pos->getPrevNonDebugInstruction(true);
        if (Prev == nullptr || Prev == Val || Prev == AllocaI)
          break;

        if (StoreInst *Store = dyn_cast<StoreInst>(Prev))
          if (Store->getPointerOperand() == AllocaI)
            break;

        if (LoadInst *Load = dyn_cast<LoadInst>(Prev))
          if (Load->getPointerOperand() == AllocaI)
            break;
        Pos = Prev;
      } while (true);

      Builder.SetInsertPoint(Pos);
      Builder.CreateStore(Val, AllocaI);
    }

    // Convert the Phi to loads from the created alloca, but sink the load
    // to the successor basic blocks, if possible. This is to reduce non-uniform
    // code in loop structure blocks with potentially non-uniform variables,
    // which makes sharing of barrier loop structures between work-items not
    // feasible.
    std::vector<llvm::Value *> Users(Phi->user_begin(), Phi->user_end());
    llvm::BasicBlock *PhiBB = Phi->getParent();

    // If the value is used in the original PHI basic block, just add the
    // load in its place. Not much we can do here as we cannot sink it down.
    bool IsSinkable = !(Phi->isUsedInBasicBlock(PhiBB) || StoreInPhiBlock);

    if (IsSinkable) {
      // Check that the destination basic blocks are not likely loop entries,
      // thus do not have other branches into them expect the branch from
      // the PHI block.
      for (BasicBlock *Succ : successors(PhiBB)) {
        if (Succ->getSinglePredecessor() != PhiBB) {
          IsSinkable = false;
          break;
        }

        // Check if we can find dominating load.
        if (!isFeasibleToSink(PhiBB, Users, DT)) {
          IsSinkable = false;
          break;
        }
      }
    }

    if (!IsSinkable) {
      // Add the load in the original Phis place. We cannot sink this one.
      Builder.SetInsertPoint(Phi);

      llvm::Instruction *LoadedValue =
          Builder.CreateLoad(Phi->getType(), AllocaI);
      Phi->replaceAllUsesWith(LoadedValue);
    } else {
      // Add loads to each of the successor basic blocks of the PHI instead.
      std::vector<llvm::LoadInst *> SunkLoads;
      for (BasicBlock *Succ : successors(PhiBB)) {
#if LLVM_MAJOR < 20
        Builder.SetInsertPoint(Succ->getFirstNonPHI());
#else
        Builder.SetInsertPoint(Succ->getFirstNonPHIIt());
#endif
        llvm::LoadInst *Load = Builder.CreateLoad(Phi->getType(), AllocaI);
        SunkLoads.push_back(Load);
      }
      // Find out which user can reach which load using dominator analysis.
      for (auto &U : Users) {
        llvm::Instruction *Instr = dyn_cast<Instruction>(U);
        llvm::LoadInst *DominatingLoad = nullptr;
        for (llvm::LoadInst *Load : SunkLoads) {
          LLVM_DEBUG(std::cerr << "SunkLoad:\n");
          LLVM_DEBUG(Load->dump());
          // If user itself is a phi node, the basic block isn't necessarily
          // dominated by any of the loads.
          if (llvm::isa<llvm::PHINode>(Instr)) {

            llvm::PHINode *PhiUser = dyn_cast<PHINode>(Instr);
            // Check dominance in the context of incoming blocks instead.
            for (unsigned I = 0; I < PhiUser->getNumIncomingValues(); ++I) {
              BasicBlock *IncomingBB = PhiUser->getIncomingBlock(I);

              if (DT.dominates(Load->getParent(), IncomingBB)) {
                DominatingLoad = Load;
                break;
              }
            }
            // For a Non-phi user:
          } else {
            if (DT.dominates(Load->getParent(), Instr->getParent())) {
              DominatingLoad = Load;
              break;
            }
          }
        }
        assert(DominatingLoad != nullptr && "No dominating load created?");
        Instr->replaceUsesOfWith(Phi, DominatingLoad);
      }
    }
    Phi->eraseFromParent();
  }

  LLVM_DEBUG(dbgs() << "After PHIsToAllocas\n");
  LLVM_DEBUG(F.dump());

  return PHIs.size() > 0;
}

} // namespace pocl
