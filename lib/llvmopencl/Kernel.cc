// Class for kernels, llvm::Functions that represent OpenCL C kernels.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012-2019 Pekka Jääskeläinen
//               2024 Pekka Jääskeläinen / Intel Finland Oy
//               2025 Tapio Nevalainen / Tampere University
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
#include <stack>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS

IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"
#include "DebugHelpers.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "SubgroupBarrier.h"
#include "WorkgroupBarrier.h"

#include "pocl.h"
#include "pocl_llvm_api.h"

POP_COMPILER_DIAGS

#define DEBUG_TYPE "ParallelRegion"

#ifdef LLVM_DEBUG
#undef LLVM_DEBUG
#endif

#ifdef DEBUG_PR_CREATION
#define LLVM_DEBUG(X) X
#define dbgs() std::cerr << DEBUG_TYPE << ": "
#else
#define LLVM_DEBUG(X)
#endif

using namespace llvm;
using namespace pocl;

static void addPredecessors(SmallVectorImpl<BasicBlock *> &V, BasicBlock *BB);

void Kernel::getRegionBarrierMapping(
    std::map<llvm::BasicBlock *, std::vector<llvm::BasicBlock *>>
        &BarrierMapping) {

  // Collect barrier blocks here.
  SmallVector<llvm::BasicBlock *, 4> BarrierBlocks;

  for (iterator i = begin(), e = end(); i != e; ++i) {
    llvm::BasicBlock *BB = cast<BasicBlock>(i);
    // Exclude the exit barrier as it won't act as an entry barrier.
    if (Barrier::hasBarrier(BB) &&
        BB->getTerminator()->getNumSuccessors() > 0) {
      BarrierBlocks.push_back(BB);
    }
    // Pure uniform blocks also behave similarly to barriers in the context of
    // parallel region formation.
    if (isPureUniformBlock(BB)) {
      BarrierBlocks.push_back(BB);
    }
  }

  // Create a barrier mapping.
  for (llvm::BasicBlock *BarrEntry : BarrierBlocks) {

    // Collect exit barriers here.
    std::vector<llvm::BasicBlock *> BarrierExits;

    // Iterate all successors up until barriers.
    std::vector<llvm::BasicBlock *> worklist;
    std::vector<llvm::BasicBlock *> visited;
    worklist.push_back(BarrEntry);

    while (!worklist.empty()) {
      llvm::BasicBlock *current = worklist.back();
      worklist.pop_back();
      visited.push_back(current);
      // Dont proceed with paths with non start-barrier.
      if (current != BarrEntry && Barrier::hasBarrier(current)) {
        BarrierExits.push_back(current);
        continue;
      }

      // These will serve as exits:
      if (isPureUniformBlock(current) && current != BarrEntry) {
        BarrierExits.push_back(current);
        continue;
      }

      for (llvm::BasicBlock *Succ : successors(current)) {
        if (std::find(visited.begin(), visited.end(), Succ) == visited.end()) {
          worklist.push_back(Succ);
        }
      }
    }

    BarrierMapping[BarrEntry] = BarrierExits;
  }
}

/// Finds the exit blocks of parallel regions.
///
/// Exit blocks are function exit nodes, barriers or forced uniform blocks.
void Kernel::getRegionExitBlocks(SmallVectorImpl<llvm::BasicBlock *> &B) {
  for (iterator i = begin(), e = end(); i != e; ++i) {
    auto t = i->getTerminator();
    llvm::BasicBlock *BB = cast<BasicBlock>(i);
    if (t->getNumSuccessors() == 0) {
      // All exits must be barrier blocks.
      // TO CLEAN: This should not be needed any more since CanonicalizeBarriers
      // adds them.
      if (!Barrier::hasBarrier(BB))
        WorkgroupBarrier::createAtEnd(BB);
      B.push_back(BB);
    } else if (Barrier::hasBarrier(BB)) {
      B.push_back(BB);
    } else if (isPureUniformBlock(BB) && BB != &getEntryBlock()) {
      B.push_back(BB);
    }
  }
}

ParallelRegion *Kernel::CreateParallelRegionBetween(
    llvm::BasicBlock *EntryBarr, llvm::BasicBlock *ExitBarr,
    pocl::ParallelRegion::ParallelRegionVector *regions,
    VariableUniformityAnalysisResult &VUA) {

  LLVM_DEBUG(dbgs() << "# createParallelRegionBetween "
                    << EntryBarr->getName().str() << " and "
                    << ExitBarr->getName().str() << "\n");

  BasicBlock *EntryBlock = nullptr;

  bool isSGRegion = false;

  // Subgroup region is a PR bounded by sg-barriers. Set the flag in case we
  // encounter one.
  if (SubgroupBarrier::hasSGBarrier(EntryBarr) &&
      SubgroupBarrier::hasSGBarrier(ExitBarr)) {
    isSGRegion = true;
  }

  bool emptyPath = false;

  // Set the parallel region entry block and check that it's not empty region.
  for (llvm::BasicBlock *Succ : successors(EntryBarr)) {

    // Empty region.
    if (Succ == ExitBarr) {
      // return nullptr;
      emptyPath = true;
    } else {
      EntryBlock = Succ;
    }
  }

  // If there is single empty path from entry barrier to exit barrier, do not
  // create parallel region.
  if (emptyPath && EntryBlock == nullptr) {
    return nullptr;
  }

  // We have to handle cases where entry barrier has two branches, both of which
  // go to the same parallel region. Extract the condition out of the barrier
  // and create single entry block.
  int successorCount = EntryBarr->getTerminator()->getNumSuccessors();
  int exitBarrCountered = 0;
  bool createNewEntryBlock = false;

  // We have
  if (successorCount > 1) {
    createNewEntryBlock = true;
    // Traverse all successors until barrier is encountered. If barrier is same
    // for all paths, create single entry block. Unless there is a barrier
    // directly after entry.
    for (int i = 0; i < successorCount; i++) {
      llvm::BasicBlock *Successor = EntryBarr->getTerminator()->getSuccessor(i);

      std::vector<llvm::BasicBlock *> worklist;
      std::vector<llvm::BasicBlock *> visited;

      if (Barrier::hasBarrier(Successor)) {
        createNewEntryBlock = false;
        continue;
      }

      worklist.push_back(Successor);

      while (!worklist.empty()) {

        llvm::BasicBlock *Current = worklist.back();
        visited.push_back(Current);
        worklist.pop_back();

        if (Barrier::hasBarrier(Current) && Current != ExitBarr) {
          createNewEntryBlock = false;
          break;
        }
        if (Current == ExitBarr) {
          continue;
        }

        for (llvm::BasicBlock *SuccBlock : successors(Current)) {
          if (std::find(visited.begin(), visited.end(), SuccBlock) ==
              visited.end()) {
            worklist.push_back(SuccBlock);
          }
        }
      }
    }
  }

  // Create new entry block, when there is a conditional branch from entry
  // barrier and both branches should be in the same parallel region. Create a
  // new barrier block before the old entry barrier and then remove the barrier
  // from the old barrier block.
  //  (B)     (B)
  //  /|   ->  |
  // ()()     ()
  //          /|
  //         ()()
  if (createNewEntryBlock) {

    ValueToValueMapTy VMap;
    BasicBlock *ClonedEntryBarr = CloneBasicBlock(
        EntryBarr, VMap, "_singular_entry", EntryBarr->getParent());

    // Update instructions inside cloned block to use mapped values
    for (Instruction &I : *ClonedEntryBarr) {
      RemapInstruction(&I, VMap, RF_IgnoreMissingLocals);
    }

    // Remove barrier from the cloned block.
    llvm::Instruction *BarrierToRemove;
    for (llvm::Instruction &Instr : *ClonedEntryBarr) {
      if (isa<Barrier>(&Instr))
        BarrierToRemove = &Instr;
    }

    BarrierToRemove->eraseFromParent();

    // Erase branching from the barrier and make a new branch to singular entry
    // block.
    llvm::Instruction *BarrTerminator = EntryBarr->getTerminator();
    BarrTerminator->eraseFromParent();

    llvm::IRBuilder<> Builder(EntryBarr);
    Builder.CreateBr(ClonedEntryBarr);
  }

  SmallPtrSet<BasicBlock *, 8> BlocksInRegion;

  BasicBlock *ExitBlock = nullptr;

  bool done = false;

  // Traverse the paths from entry barrier to exit barrier and collect the basic
  // blocks. Single barrier can branch to multiple parallel regions so we need
  // to check all paths to find the correct exit barrier.
  for (llvm::BasicBlock *EntryBarrSucc : successors(EntryBarr)) {

    // Skip immediate successive barriers after the entry barrier.
    if (Barrier::hasBarrier(EntryBarrSucc)) {
      continue;
    }

    // Found the correct path to exit barrier, no need to check others.
    if (done) {
      break;
    }

    // Entry block candidate.
    EntryBlock = EntryBarrSucc;

    std::vector<llvm::BasicBlock *> worklist;
    SmallVector<llvm::BasicBlock *, 4> visited;

    worklist.push_back(EntryBarrSucc);

    // Keep track of the previous block, the exit block will be stored here once
    // we find the correct exit barrier.
    llvm::BasicBlock *PreviousBlock;

    while (!worklist.empty()) {
      llvm::BasicBlock *CurrentBlock = worklist.back();
      worklist.pop_back();

      visited.push_back(CurrentBlock);

      if (!Barrier::hasBarrier(CurrentBlock) &&
          !isPureUniformBlock(CurrentBlock)) {
        BlocksInRegion.insert(CurrentBlock);
      } else {
        // Ended up traversing the incorrect parallel region.
        // Clear the collected blocks and continue with next successor of
        // entry barrier.
        if (CurrentBlock != ExitBarr) {
          BlocksInRegion.clear();
          break;
        } else {
          ExitBlock = PreviousBlock;
          done = true;
          continue;
        }
      }

      for (llvm::BasicBlock *SuccBlock : successors(CurrentBlock)) {
        if (find(visited, SuccBlock) == visited.end())
          worklist.push_back(SuccBlock);
      }
      PreviousBlock = CurrentBlock;
    }
  }

  // It can be the case that Exit barrier is the successor of Entry barrier AND
  // there is no other unbarriered path from entry barrier to exit barrier. In
  // this case, parallel region is not created.
  if (BlocksInRegion.size() == 0) {
    return nullptr;
  }

  // Single parallel region can have two different entry barriers.
  // Check that this region has not been created already with different barrier
  // pair.
  for (ParallelRegion::ParallelRegionVector::iterator PRI = regions->begin(),
                                                      PRE = regions->end();
       PRI != PRE; ++PRI) {

    ParallelRegion *PRegion = (*PRI);

    if (PRegion->exitBB() == ExitBlock)
      return nullptr;
  }

  llvm::BasicBlock *MaybeExitBlock = ExitBarr->getSinglePredecessor();

  // Single exit block.
  if (MaybeExitBlock != nullptr) {
    ExitBlock = MaybeExitBlock;
  } else {
    // More than one exit block leading to exit barrier.
    // Need to create singular exit block, and update references.

    std::vector<llvm::BasicBlock *> exitBlocks;
    // Collect blocks that precede the exit barrier AND belong to this region.
    for (llvm::BasicBlock *ExitBarrPred : predecessors(ExitBarr)) {
      if (BlocksInRegion.count(ExitBarrPred) == 1)
        exitBlocks.push_back(ExitBarrPred);
    }

    if (exitBlocks.size() > 1) {
      // Create new singular exit block for the region.
      llvm::BasicBlock *NewExitBlock =
          llvm::BasicBlock::Create(EntryBarr->getContext(), "singular_exit",
                                   EntryBarr->getParent(), ExitBarr);
      BlocksInRegion.insert(NewExitBlock);
      llvm::IRBuilder<> Builder(EntryBarr->getContext());
      Builder.SetInsertPoint(NewExitBlock);

      Builder.CreateBr(ExitBarr);

      // Update references to new exit blocks.
      for (llvm::BasicBlock *ExitingBlock : exitBlocks) {
        for (unsigned succ_idx = 0;
             succ_idx < ExitingBlock->getTerminator()->getNumSuccessors();
             ++succ_idx) {
          if (ExitingBlock->getTerminator()->getSuccessor(succ_idx) == ExitBarr)
            ExitingBlock->getTerminator()->setSuccessor(succ_idx, NewExitBlock);
        }
      }
      ExitBlock = NewExitBlock;
    } else {
      ExitBlock = exitBlocks[0];
    }
  }

  BasicBlock *PREntry = BasicBlock::Create(
      EntryBlock->getContext(),
      "parallel_region_" + std::to_string(ParallelRegion::getNextID()) +
          "_entry",
      EntryBlock->getParent(), EntryBlock);

  IRBuilder<> Builder(PREntry);
  Builder.CreateBr(EntryBlock);

  // Does it have a jump to the next block?
  BlocksInRegion.insert(PREntry);

  std::set<BasicBlock *> Preds;
  for (pred_iterator PI = pred_begin(EntryBlock), PE = pred_end(EntryBlock);
       PI != PE; ++PI) {
    Preds.insert(*PI);
  }
  // There can be many predecessor basic blocks to the region,
  // fix all the predecessor blocks from other regions to jump to the region
  // entry. Note: a special case is the intra-PR-loop case where the header node
  // is the loop header. In that case get incoming branches to the
  // entry from inside the PR.
  for (BasicBlock *PredBB : Preds) {
    if (BlocksInRegion.count(PredBB) > 0)
      continue; // Must be an intra-PR loop backedge source.

    BranchInst *BR = cast<BranchInst>(PredBB->getTerminator());

    for (unsigned Suc = 0; Suc < BR->getNumSuccessors(); ++Suc)
      if (BR->getSuccessor(Suc) == EntryBlock)
        BR->setSuccessor(Suc, PREntry);
    EntryBlock->replacePhiUsesWith(PredBB, PREntry);
  }

  // Corner-case exists where there is a branch from entry barrier to exit
  // barrier skipping the parallel region altogether. This may be intended if it
  // is a 'work-group level' branch. However, sometimes they are supposed to be
  // 'WI level' branches. We have to check if this is the case and fix those.
  // TODO: This is not the ideal solution, and depends on how things are
  // implemented currently. Maybe we don't need this if PR formation logic is
  // rewritten. This was Gromacs 'PME-Gather/Solve'-kernel issue.
  //  (B)
  //  / |
  //  | [PR]
  //  |/
  //  (B)

  // In all relevant cases, there are two branches from the entry barrier.
  if (EntryBarr->getTerminator()->getNumSuccessors() > 1) {

    for (int i = 0; i < EntryBarr->getTerminator()->getNumSuccessors(); ++i) {
      // If we have a branch from entry barrier to exit barrier.
      if (EntryBarr->getTerminator()->getSuccessor(i) == ExitBarr) {

        // Check if loop-variable is uniform.
        llvm::Function *F = EntryBarr->getParent();
        llvm::BranchInst *Br =
            dyn_cast<llvm::BranchInst>(EntryBarr->getTerminator());

        if (!VUA.isUniform(F, Br->getCondition())) {

          // Modify entry barrier to have a single branch, and move the
          // conditional branch inside the PR. Note we cant simply redirect the
          // 'passing' branch to exit block of the PR. So we have to create new
          // exit block as well.

          // The new entry block.
          llvm::BasicBlock *NewEntryBB =
              BasicBlock::Create(EntryBarr->getContext(), "handled_entry",
                                 EntryBarr->getParent(), EntryBarr);
          IRBuilder<> Builder(NewEntryBB);

          // Branch to 'old' entry block.
          Builder.CreateBr(PREntry);
          BlocksInRegion.insert(NewEntryBB);

          // The new exit block.
          llvm::BasicBlock *NewExitBB =
              BasicBlock::Create(EntryBarr->getContext(), "handled_exit",
                                 EntryBarr->getParent(), ExitBarr);

          Builder.SetInsertPoint(ExitBlock);

          // Insert the new exit block between the old exit block and the exit
          // barrier.
          llvm::Instruction *Term = ExitBlock->getTerminator();
          Builder.CreateBr(NewExitBB);
          Term->eraseFromParent();
          Builder.SetInsertPoint(NewExitBB);
          Builder.CreateBr(ExitBarr);
          BlocksInRegion.insert(NewExitBB);

          // Finally, move the conditional branch from the barrier to the new
          // entry block.
          llvm::Instruction *BranchTerminator = EntryBarr->getTerminator();
          BranchTerminator->removeFromParent();
          Builder.SetInsertPoint(EntryBarr);
          Builder.CreateBr(NewEntryBB);
          Builder.SetInsertPoint(NewEntryBB);
          llvm::Instruction *Terminator = NewEntryBB->getTerminator();

          // This is deprecated on LLVM20.
          BranchTerminator->insertBefore(Terminator);

          Terminator->eraseFromParent();

          // Redirect the 'exit barrier' branch to the new exit block.
          BranchTerminator->replaceUsesOfWith(ExitBarr, NewExitBB);

          // Update new entry/exit block.
          PREntry = NewEntryBB;
          ExitBlock = NewExitBB;
        }
      }
    }
  }

  return ParallelRegion::Create(BlocksInRegion, PREntry, ExitBlock, isSGRegion);
}

static void addPredecessors(SmallVectorImpl<BasicBlock *> &V, BasicBlock *BB) {
  for (pred_iterator i = pred_begin(BB), e = pred_end(BB); i != e; ++i) {
    V.push_back(*i);
  }
}

/// The main entry to the "parallel region formation" which searches for regions
/// of basic blocks between barriers that can be freely parallelized across
/// work-items in the work-group.
void Kernel::getParallelRegions(
    llvm::LoopInfo &LI, ParallelRegion::ParallelRegionVector *ParallelRegions,
    VariableUniformityAnalysisResult &VUA) {

  SmallVector<BasicBlock *, 4> RegionExitBlocks;

  std::map<llvm::BasicBlock *, std::vector<llvm::BasicBlock *>> BarrierMap;

  // Get barrier mapping.
  getRegionBarrierMapping(BarrierMap);

  for (const auto &region : BarrierMap) {

    llvm::BasicBlock *entryBarrier = region.first;

    const std::vector<llvm::BasicBlock *> &exitBarriers = region.second;

    for (llvm::BasicBlock *exitBarrier : exitBarriers) {

      ParallelRegion *PR = CreateParallelRegionBetween(
          entryBarrier, exitBarrier, ParallelRegions, VUA);

      // We might have empty region.
      if (PR == nullptr) {
        continue;
      }

      // Check that parallel region does not exist already.
      // This can happen if there are two different entry barriers to the
      // parallel region.
      bool PRexists = false;

      for (ParallelRegion::ParallelRegionVector::iterator
               PRI = ParallelRegions->begin(),
               PRE = ParallelRegions->end();
           PRI != PRE; ++PRI) {

        ParallelRegion *PRegion = (*PRI);

        if (PRegion->exitBB() == PR->exitBB()) {
          PRexists = true;
          break;
        }
      }
      if (!PRexists) {
        ParallelRegions->push_back(PR);
      }
    }
  }

#ifdef DEBUG_PR_CREATION
  dumpCFG(*this, this->getName().str() + ".pregions.dot", nullptr,
          ParallelRegions);
#endif
}

void Kernel::addLocalSizeInitCode(size_t LocalSizeX, size_t LocalSizeY,
                                  size_t LocalSizeZ) {

  CreateBuilder(Builder, getEntryBlock());

  GlobalVariable *GV;

  llvm::Module* M = getParent();

  unsigned long AddressBits;
  getModuleIntMetadata(*M, "device_address_bits", AddressBits);

  llvm::Type *SizeT = IntegerType::get(M->getContext(), AddressBits);

  GV = M->getGlobalVariable("_local_size_x");
  if (GV != nullptr) {
    Builder.CreateStore(ConstantInt::get(SizeT, LocalSizeX), GV);
  }

  GV = M->getGlobalVariable("_local_size_y");
  if (GV != nullptr)
    Builder.CreateStore(ConstantInt::get(SizeT, LocalSizeY), GV);

  GV = M->getGlobalVariable("_local_size_z");
  if (GV != nullptr)
    Builder.CreateStore(ConstantInt::get(SizeT, LocalSizeZ), GV);
}
