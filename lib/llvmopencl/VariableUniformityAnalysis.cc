// Implementation for VariableUniformityAnalysis function pass.
//
// Copyright (c) 2013-2019 Pekka Jääskeläinen
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
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/CommandLine.h"
POP_COMPILER_DIAGS

#include "Barrier.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandler.h"

#include <iostream>
#include <map>
#include <sstream>

#include "pocl_llvm_api.h"

// #define DEBUG_UNIFORMITY_ANALYSIS

#ifdef DEBUG_UNIFORMITY_ANALYSIS
#include "DebugHelpers.h"
#endif

#define PASS_NAME "pocl-vua"
#define PASS_CLASS pocl::VariableUniformityAnalysis
#define PASS_DESC                                                              \
  "Analyses the variables of the function for uniformity (same value across "  \
  "WIs)."

namespace pocl {

using namespace llvm;

/// Analyzes the loops in the function.
///
/// Loops without barriers are analyzed for divergence in order to
/// enable forced horizontal parallelization via implicit barriers.
/// For that to be done safely, we must prove that the loop iterations
/// are executed the same number of times for all work-items.
///
/// This holds if
/// a) The loop entry is not divergent when not considering the back edges:
/// All work-items either *encounter* the loop or not (but might not enter it),
/// and
/// b) the loop branch is not divergent. Currently we detect this
/// condition by checking if the loop condition values are uniform and
/// fall-back to treating them as divergent for safety in unhandled
/// cases such as updating them inside the loop body.
///
/// Due to performing these checks on unoptimized input, we might
/// not detect all safely horizontally parallelizable cases. The
/// analysis will be improved gradually.
///
/// Before this analysis, loop headers have been analyzed with a
/// top-down acyclic pass, which marks loop headers as uniform in case
/// the are encountered by all work-items.
///
/// After this analysis step, loop headers are considered uniform
/// only if the loop executes the same number of times for all
/// work-items.
void VariableUniformityAnalysisResult::analyzeLoop(
    Function &F, llvm::Loop &L, llvm::PostDominatorTree &PDT) {

  LoopUniformityIndex &Cache = LoopUniformityCache_[&F];
  LoopUniformityIndex::const_iterator I = Cache.find(&L);
  if (I != Cache.end())
    return;

  llvm::Loop *ParentLoop = L.getParentLoop();
  if (ParentLoop != nullptr) {
    // Ensure we have analyzed the parent loop(s) first because their
    // uniformity affects the child loops' uniformity.
    analyzeLoop(F, *ParentLoop, PDT);
  }

  llvm::BasicBlock *HeaderBlock = L.getHeader();
  llvm::BasicBlock *PredecessorBlock = L.getLoopPredecessor();
  llvm::BasicBlock *LatchBlock = L.getLoopLatch();
  llvm::BranchInst *LoopEntryBranch =
      PredecessorBlock != nullptr
          ? dyn_cast<llvm::BranchInst>(PredecessorBlock->getTerminator())
          : nullptr;

  const bool ParentLoopIsDivergent =
      ParentLoop != nullptr && !isUniformLoop(F, *ParentLoop);

  // Utilize the precalculated acyclic analysis information: If the header is
  // not uniform according to it, the loop might not be reached at all by some
  // of the WIs.
  const bool LoopNotReachedByAllWIs =
      LoopEntryBranch == nullptr || !isUniform(&F, PredecessorBlock) ||
      (LoopEntryBranch->isConditional() &&
       !isUniform(&F, LoopEntryBranch->getCondition()));

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "#### analyzing loop " << L.getName().str();
  std::cerr << ": ";
  if (ParentLoopIsDivergent)
    std::cerr << "parent loop is divergent ";
  if (LoopNotReachedByAllWIs)
    std::cerr << "loop is not reached by all WIs ";
  std::cerr << "\n";
#endif

  if (ParentLoopIsDivergent || LoopNotReachedByAllWIs) {
    LoopUniformityCache_[&F][&L] = false;
    return;
  }

  bool LoopStructureIsUniform = true;
  SmallVector<BasicBlock *> ExitingBlocks;
  L.getExitingBlocks(ExitingBlocks);
  // Check that all the exiting blocks have a uniform branch condition.
  for (BasicBlock *ExitingBlock : ExitingBlocks) {
    llvm::BranchInst *LoopBranch =
        ExitingBlock == nullptr
            ? nullptr
            : dyn_cast<llvm::BranchInst>(ExitingBlock->getTerminator());

    llvm::Value *LoopCondition = LoopBranch->getCondition();
    // Now the uniformity data can treat the condition as divergent since it is
    // written in the loop check basic block, which is treated as divergent at
    // this point, even if the values written to it were uniform. Let's treat
    // the increment block as uniform and run the check. If the check fails,
    // there was some other reason for the divergent result.

    setUniform(&F, ExitingBlock);
    setUniform(&F, LatchBlock);

    removeUniformityData(*LoopCondition, 10);

    LoopStructureIsUniform =
        LoopStructureIsUniform &&
        (LoopUniformityCache_[&F][&L] = isUniform(&F, LoopCondition));
  }

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "#### loop detected as "
            << (LoopStructureIsUniform ? "uniform" : "divergent") << "\n";
#endif

  if (LoopStructureIsUniform) {
    // Recompute the uniformity of the loop body's basic blocks.
    llvm::BasicBlock *BodyStart = nullptr;

    // Find the block that should have a branch to the first body BB:
    llvm::BasicBlock *EnteringBlock = L.getHeader();
    if (EnteringBlock == nullptr)
      EnteringBlock = L.getExitingBlock();

    // Todo: do...whiles with breaks might not get caught here?
    assert(EnteringBlock != nullptr);

    // One of the header branches should point to the start of the loop.
    if (L.contains(EnteringBlock->getTerminator()->getSuccessor(0)))
      BodyStart = EnteringBlock->getTerminator()->getSuccessor(0);
    else if (L.contains(EnteringBlock->getTerminator()->getSuccessor(1)))
      BodyStart = EnteringBlock->getTerminator()->getSuccessor(1);

    for (auto &BB : L.getBlocksVector()) {
      removeUniformityDatum(F, *BB);
    }
    assert(BodyStart != nullptr);

    for (BasicBlock *ExitingBlock : ExitingBlocks)
      setUniform(&F, ExitingBlock);

    setUniform(&F, LatchBlock);
    setUniform(&F, BodyStart);

    // Propagate the uniformity info to the remaining blocks of the loop body.
    analyzeBBDivergence(&F, BodyStart, BodyStart, PDT);
  } else {
    // Mark all basic blocks in the loop divergent because the loop structure
    // is divergent.
    for (auto &BB : L.getBlocksVector()) {
      setUniform(&F, BB, false);
    }
  }
}

bool VariableUniformityAnalysisResult::isUniformLoop(llvm::Function &F,
                                                     llvm::Loop &L) {
  LoopUniformityIndex &Cache = LoopUniformityCache_[&F];
  LoopUniformityIndex::const_iterator I = Cache.find(&L);
  if (I == Cache.end()) {
    // Assume non-uniform by default.
    return false;
  }
  return Cache[&L];
}

bool VariableUniformityAnalysisResult::runOnFunction(
    Function &F, llvm::LoopInfo &LI, llvm::PostDominatorTree &PDT) {

  if (!isKernelToProcess(F))
    return false;

  // Do the actual analysis on-demand except for the basic block
  // divergence analysis.
  uniformityCache_[&F].clear();

  setUniform(&F, &F.getEntryBlock());

  // Analyze the divergence first with an acyclic downwards pass.
  analyzeBBDivergence(&F, &F.getEntryBlock(), &F.getEntryBlock(), PDT);

  // Then correct loop divergence information.
  for (llvm::LoopInfo::iterator i = LI.begin(), e = LI.end(); i != e; ++i) {
    llvm::Loop *L = *i;
    analyzeLoop(F, *L, PDT);
  }

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### refreshed VUA" << std::endl;
  // Highlight the uniform basic blocks in the graph dump.
  std::set<llvm::BasicBlock *> UniformBBs;
  for (auto &BB : F) {
    if (isUniform(&F, &BB))
      UniformBBs.insert(&BB);
  }
  dumpCFG(F, F.getName().str() + "_vua.dot", nullptr, nullptr, &UniformBBs);
  F.dump();
#endif

  return false;
}

/// Returns true in case the value should be privatized, e.g., a copy
/// should be created for each parallel work-item.
///
/// This is not the same as !isUniform() because of some of the allocas.
/// Specifically, the loop iteration variables are in some cases uniform,
/// that is, each work item sees the same induction variable value at every
/// iteration, but the variables should be still replicated to avoid multiple
/// increments of the same induction variable by each work-item in a b-loop
/// of which iterator cannot be merged cross WIs.
bool VariableUniformityAnalysisResult::shouldBePrivatized(llvm::Function *F,
                                                          llvm::Value *Val) {
  if (!isUniform(F, Val)) return true;

  // Check if the value is stored in stack (is an alloca or writes to an alloca).
  // It should be enough to context save the initial alloca and the stores to
  // make sure each work-item gets their own stack slot and they are updated.
  // How the value (based on which of those allocas) is computed does not matter as
  // we are deadling with uniform computation.

  if (isa<AllocaInst>(Val)) return true;

  if (isa<StoreInst>(Val) &&
      isa<AllocaInst>(dyn_cast<StoreInst>(Val)->getPointerOperand())) return true;
  return false;
}

/// Perform basic block divergence analysis from the given basic block
/// downwards.
///
/// Definitions:
/// Uniform BB: A basic block which is known to be executed by all
/// or none of the work-items, that is, a BB where it's known safe to add a
/// barrier.
///
/// Divergent/varying BB: A basic block where work-items *might* diverge.
/// That is, it cannot be proven that all work-items execute the BB.
///
/// The function propagates the information from the entry downwards (breadth
/// first). This avoids infinite recursion with loop back edges and enables
/// book keeping of the "last seen" uniform BB.
///
/// It uses the following conditions to mark a BB 'uniform':
///
/// a) the function entry, or
/// b) BBs that post-dominate at least one uniform BB (try the previously
///    found one), or
/// c) BBs that are branched to directly from a uniform BB using a uniform
/// branch. Note: This assumes the CFG is well-formed in a way that there cannot
/// be a divergent branch to the same BB in that case.
///
/// Otherwise, we assume divergent for safety (it might not be *proven* to be one
/// though!).
void VariableUniformityAnalysisResult::analyzeBBDivergence(
    llvm::Function *F, llvm::BasicBlock *StartingBB,
    llvm::BasicBlock *PreviousUniformBB, llvm::PostDominatorTree &PDT) {

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### Analyzing BB divergence (BB=" << StartingBB->getName().str()
            << ", prevUniform=" << PreviousUniformBB->getName().str() << ")"
            << std::endl;
#endif

  auto Term = PreviousUniformBB->getTerminator();
  if (Term == NULL) {
    // this is most likely a function with a single basic block, the entry
    // node, which ends with a ret
    return;
  }

  llvm::BranchInst *BrInst = dyn_cast<llvm::BranchInst>(Term);
  llvm::SwitchInst *SwInst = dyn_cast<llvm::SwitchInst>(Term);

  if (BrInst == nullptr && SwInst == nullptr) {
    // Can only handle branches and switches for now.
    return;
  }

  // The BBs that were found uniform.
  std::vector<llvm::BasicBlock *> FoundUniforms;

  // Condition c)
  if ((BrInst && (!BrInst->isConditional() ||
                  isUniform(F, BrInst->getCondition()))) ||
      (SwInst && isUniform(F, SwInst->getCondition()))) {
    // This is a branch with a uniform condition, propagate the uniformity
    // to the BB of interest.
    for (unsigned suc = 0, end = Term->getNumSuccessors(); suc < end; ++suc) {
      llvm::BasicBlock *Successor = Term->getSuccessor(suc);
      // TODO: should we check that there are no divergent entries to this
      // BB even though if the currently checked condition is uniform?
      setUniform(F, Successor, true);
      FoundUniforms.push_back(Successor);
    }
  }

  // Condition b)
  if (FoundUniforms.size() == 0) {
    if (PDT.dominates(StartingBB, PreviousUniformBB)) {
      setUniform(F, StartingBB, true);
      FoundUniforms.push_back(StartingBB);
    }
  }

  // Assume diverging.
  if (!isUniformityAnalyzed(F, StartingBB))
    setUniform(F, StartingBB, false);

  for (auto UniformBB : FoundUniforms) {

    // Propagate the Uniform BB data downwards.
    auto NextTerm = UniformBB->getTerminator();

    for (unsigned Suc = 0, End = NextTerm->getNumSuccessors(); Suc < End;
         ++Suc) {
      llvm::BasicBlock *NextBB = NextTerm->getSuccessor(Suc);
      if (!isUniformityAnalyzed(F, NextBB)) {
        analyzeBBDivergence(F, NextBB, UniformBB, PDT);
      }
    }
  }
}

/// Removes the uniformity datum for the given value, if found.
///
/// \p V can be a basic block.
void VariableUniformityAnalysisResult::removeUniformityDatum(llvm::Function &F,
                                                             llvm::Value &V) {
  UniformityIndex &Cache = uniformityCache_[&F];
  UniformityIndex::const_iterator I = Cache.find(&V);
  if (I != Cache.end()) {
    Cache.erase(I);
  }
}

/// Clears the uniformity result cache for the given value and its producers so
/// the data gets recomputed the next time it's requested.
///
/// \p V is the value to start from. Must not be a basic block.
/// \p Depth the maximum recursion depth.
void VariableUniformityAnalysisResult::removeUniformityData(llvm::Value &V,
                                                            int Depth) {

  llvm::Instruction *Instr = dyn_cast<llvm::Instruction>(&V);
  if (Instr == nullptr)
    return;

  llvm::Function *F = Instr->getParent()->getParent();
  removeUniformityDatum(*F, V);

  if (Depth == 0)
    return;

  for (unsigned OprI = 0; OprI < Instr->getNumOperands(); ++OprI) {
    llvm::Value &Operand = *Instr->getOperand(OprI);
    removeUniformityData(Operand, Depth - 1);
  }
}

bool VariableUniformityAnalysisResult::isUniformityAnalyzed(
    llvm::Function *F, llvm::Value *V) const {
  UniformityIndex &Cache = uniformityCache_[F];
  UniformityIndex::const_iterator I = Cache.find(V);
  if (I != Cache.end()) {
    return true;
  }
  return false;
}

/// Recursively analyses all the operands affecting the value to find out
/// the uniformity.
///
/// Known uniform Values that act as "leafs" in the recursive uniformity
/// check logic:
/// a) kernel arguments
/// b) constants
/// c) OpenCL C identifiers that are constant for all work-items in a work-group
bool VariableUniformityAnalysisResult::isUniform(llvm::Function *F,
                                                 llvm::Value *V) {

  UniformityIndex &Cache = uniformityCache_[F];
  UniformityIndex::const_iterator I = Cache.find(V);
  if (I != Cache.end()) {
    return (*I).second;
  }

  if (llvm::BasicBlock *BB = dyn_cast<llvm::BasicBlock>(V)) {
    if (BB == &F->getEntryBlock() || isPureUniformBlock(BB)) {
      setUniform(F, V, true);
      return true;
    }
  }

  if (isa<llvm::Argument>(V)) {
    setUniform(F, V, true);
    return true;
  }

  llvm::Module *M = F->getParent();
  if (isa<llvm::Constant>(V) && !(V == M->getGlobalVariable(LID_G_NAME(0)) ||
                                  V == M->getGlobalVariable(LID_G_NAME(1)) ||
                                  V == M->getGlobalVariable(LID_G_NAME(2)) ||
                                  V == M->getGlobalVariable(GID_G_NAME(0)) ||
                                  V == M->getGlobalVariable(GID_G_NAME(1)) ||
                                  V == M->getGlobalVariable(GID_G_NAME(2)))) {
    setUniform(F, V, true);
    return true;
  }

  if (isa<llvm::AllocaInst>(V)) {
    llvm::AllocaInst *Alloca = dyn_cast<llvm::AllocaInst>(V);

    /* Allocas might or might not be divergent. These are produced
       from work-item private arrays or the PHIsToAllocas. It depends
       what is written to them whether they are really divergent.

       We need to figure out if any of the stores to the alloca contain
       work-item id dependent data. We take a white-listing approach that
       detects the ex-phi allocas of loop iteration variables of non-diverging
       loops.

       Currently the following case is considered uniform:
       a) contains a scalar type and
       b) are accessed only with load and stores (e.g. address not taken) from
          uniform basic blocks, and
       c) the stored data is uniform OR
       d) the alloca is written to from inside a forced uniform block (B-loop
          constructs currently)

       Because alloca data can be modified in loops and thus be dependent on
       itself, we need a bit involved mechanism to handle it. First create
       a copy of the uniformity cache, then assume the alloca itself is uniform,
       then check if all the stores to the alloca contain uniform data. If
       our initial assumption was wrong, restore the cache from the backup.
    */

    // Check the case d) first.
    for (Instruction::use_iterator UI = Alloca->use_begin(),
                                   UE = Alloca->use_end();
         UI != UE; ++UI) {
      llvm::StoreInst *Store = dyn_cast<llvm::StoreInst>(UI->getUser());
      if (Store == nullptr)
        continue;
      bool ForcedUniformStoreFound = isPureUniformBlock(Store->getParent());
      if (ForcedUniformStoreFound) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
        std::cerr << "### alloca was written in a forced-uniform BB"
                  << std::endl;
#endif
        setUniform(F, V);
        return true;
      } else {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
        std::cerr << "### alloca written in "
                  << Store->getParent()->getName().str() << std::endl;
#endif
      }
    }

    UniformityCache backupCache(uniformityCache_);
    setUniform(F, V);

    bool isUniformAlloca = true;
    for (Instruction::use_iterator ui = Alloca->use_begin(),
                                   ue = Alloca->use_end();
         ui != ue; ++ui) {
      llvm::Instruction *user = cast<Instruction>(ui->getUser());
      if (user == NULL) continue;
      
      llvm::StoreInst *store = dyn_cast<llvm::StoreInst>(user);
      if (store) {
        if (!isUniform(F, store->getValueOperand()) ||
            !isUniform(F, store->getParent())) {
          if (!isUniform(F, store->getParent())) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
            std::cerr << "### alloca was written in a non-uniform BB" << std::endl;
            store->getParent()->dump();
            /* TODO: This is a problematic chicken-egg situation because the 
               BB uniformity check ends up analyzing allocas in phi-removed code:
               the loop constructs refer to these allocas and at that point we
               do not yet know if the BB itself is uniform. This leads to not
               being able to detect loop iteration variables as uniform. */
#endif
          }
          isUniformAlloca = false;
          break;
        }
      } else if (isa<llvm::LoadInst>(user) || isa<llvm::BitCastInst>(user)) {
      } else if (isa<llvm::CallInst>(user)) {
        CallInst *CallInstr = dyn_cast<CallInst>(user);
        Function *Callee = CallInstr->getCalledFunction();
        if (Callee != nullptr &&
            (Callee->getName().starts_with("llvm.lifetime.end") ||
             Callee->getName().starts_with("llvm.lifetime.start"))) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
          std::cerr << "### alloca is used by llvm.lifetime" << std::endl;
          user->dump();
#endif
        } else {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
          std::cerr << "### alloca has a suspicious user" << std::endl;
          user->dump();
#endif
          isUniformAlloca = false;
          break;
        }
      } else {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
        std::cerr << "### alloca has a suspicious user" << std::endl;
        user->dump();
#endif
        isUniformAlloca = false;
        break;
      }
    }

    if (!isUniformAlloca) {
      // restore the old uniform data as our guess was wrong
      uniformityCache_ = backupCache;
    }
    setUniform(F, V, isUniformAlloca);
    
    return isUniformAlloca;
  }

  /* TODO: global memory loads are uniform in case they are accessing
     the higher scope ids (group_id_?). */
  if (isa<llvm::LoadInst>(V)) {
    llvm::LoadInst *load = dyn_cast<llvm::LoadInst>(V);
    llvm::Value *pointer = load->getPointerOperand();
    llvm::Module *M = load->getParent()->getParent()->getParent();

    if (pointer == M->getGlobalVariable("_group_id_x") ||
        pointer == M->getGlobalVariable("_group_id_y") ||
        pointer == M->getGlobalVariable("_group_id_z") ||
        pointer == M->getGlobalVariable("_work_dim") ||
        pointer == M->getGlobalVariable("_num_groups_x") ||
        pointer == M->getGlobalVariable("_num_groups_y") ||
        pointer == M->getGlobalVariable("_num_groups_z") ||
        pointer == M->getGlobalVariable("_global_offset_x") ||
        pointer == M->getGlobalVariable("_global_offset_y") ||
        pointer == M->getGlobalVariable("_global_offset_z") ||
        pointer == M->getGlobalVariable("_local_size_x") ||
        pointer == M->getGlobalVariable("_local_size_y") ||
        pointer == M->getGlobalVariable("_local_size_z") ||
        // Since we support only uniform SG sizes for now:
        pointer == M->getGlobalVariable("_pocl_sub_group_size") ||
        pointer == M->getGlobalVariable(PoclGVarBufferName)) {

      setUniform(F, V, true);
      return true;
    }
  } else if (llvm::CallInst *Call = dyn_cast<llvm::CallInst>(V)) {
    auto Callee = Call->getCalledFunction();
    if (Callee == nullptr) {
      // Likely inline asm. Cannot analyze uniformity.
      setUniform(F, V, false);
      return false;
    }
    auto CalleeName = Callee->getName();
    bool IsUniformBuiltin = CalleeName == GROUP_ID_BUILTIN_NAME ||
                            CalleeName == GS_BUILTIN_NAME ||
                            CalleeName == LS_BUILTIN_NAME;
#ifdef DEBUG_UNIFORMITY_ANALYSIS
    std::cerr << "### VUA: call to " << CalleeName.str() << " is "
              << (IsUniformBuiltin ? "" : "not ") << "uniform\n";
#endif
    setUniform(F, V, IsUniformBuiltin);
    return IsUniformBuiltin;
  }

  if (llvm::PHINode *PHI = dyn_cast<llvm::PHINode>(V)) {
    // Do not try to prove PHIs uniform for now due to recursivity.
    setUniform(F, V, false);
    return false;
  }

  llvm::Instruction *instr = dyn_cast<llvm::Instruction>(V);
  if (instr == NULL) {
    setUniform(F, V, false);
    return false;
  }

  // Atomic operations might look like uniform if only considering the operands
  // (access a global memory location of which ordering by default is not
  // constrained), but their semantics have ordering: Each work-item should get
  // their own value from that memory location.
  if (instr->isAtomic()) {
      setUniform(F, V, false);
      return false;
  }

  // Not computed previously, scan all operands of the instruction
  // and figure out their uniformity recursively.
  for (unsigned opr = 0; opr < instr->getNumOperands(); ++opr) {
      llvm::Value *operand = instr->getOperand(opr);
      if (!isUniform(F, operand)) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
      std::cerr << "### operand not uniform" << std::endl;
      operand->dump();
#endif
      setUniform(F, V, false);
      return false;
    }
  }
  setUniform(F, V, true);
  return true;
}

/// Returns true in case the \p BB contains diverging non-branch instructions.
bool VariableUniformityAnalysisResult::hasDivergingInstructions(
    llvm::BasicBlock &BB) {
  for (llvm::Instruction &I : BB) {
    if (isa<BranchInst>(I))
      continue;
    if (!isUniform(I.getParent()->getParent(), &I)) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
      std::cerr << "### not uniform:\n";
      I.dump();
#endif
      return true;
    }
  }
  return false;
}

bool VariableUniformityAnalysisResult::isPureUniformAlloca(
    llvm::AllocaInst *Alloca) {
  bool PureUniformAccessesFound = false;
  size_t NonPUWriteCount = 0;
  size_t NonUniformWriteCount = 0;
  for (Instruction::use_iterator UI = Alloca->use_begin(),
                                 UE = Alloca->use_end();
       UI != UE; ++UI) {
    llvm::StoreInst *Store = dyn_cast<llvm::StoreInst>(UI->getUser());
    llvm::LoadInst *Load = dyn_cast<llvm::LoadInst>(UI->getUser());

    if (Store == nullptr && Load == nullptr)
      continue;
    llvm::Instruction *MemAccess = dyn_cast<llvm::Instruction>(UI->getUser());

    bool PureUniformBlock = isPureUniformBlock(MemAccess->getParent());

    PureUniformAccessesFound |= PureUniformBlock;
    if (Store != nullptr &&
        !isUniform(Store->getParent()->getParent(), Store->getValueOperand()))
      NonUniformWriteCount++;
  }

  // The variable could be updated outside with non-uniform values.
  // We should not mark the block uniform in that case.
  assert(NonPUWriteCount == 0 || NonUniformWriteCount == 0);

  return PureUniformAccessesFound > 0;
}

void VariableUniformityAnalysisResult::setUniform(llvm::Function *F,
                                                  llvm::Value *V,
                                                  bool isUniform) {

  UniformityIndex &Cache = uniformityCache_[F];
  Cache[V] = isUniform;

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### ";
  if (isUniform)
    std::cerr << "uniform ";
  else
    std::cerr << "varying ";

  if (isa<llvm::BasicBlock>(V)) {
    std::cerr << "BB: " << V->getName().str() << std::endl;
  } else {
    V->dump();
  }
#endif
}

bool VariableUniformityAnalysisResult::doFinalization(llvm::Module & /*M*/) {
  uniformityCache_.clear();
  return true;
}


llvm::AnalysisKey VariableUniformityAnalysis::Key;

VariableUniformityAnalysis::Result
VariableUniformityAnalysis::run(llvm::Function &F,
                                llvm::FunctionAnalysisManager &AM) {
  llvm::LoopInfo &LI = AM.getResult<llvm::LoopAnalysis>(F);
  llvm::PostDominatorTree &PDT =
      AM.getResult<llvm::PostDominatorTreeAnalysis>(F);

  VariableUniformityAnalysisResult Res;
  Res.runOnFunction(F, LI, PDT);
  return Res;
}

bool VariableUniformityAnalysisResult::invalidate(
    llvm::Function &F, const llvm::PreservedAnalyses PA,
    llvm::AnalysisManager<llvm::Function>::Invalidator &Inv) {
  // TODO: this is required by the LoopPasses that use this analysis; however,
  // it's most likely incorrect. We should convert LoopPasses to FunctionPasses
  // and properly invalidate VUA
  return false;
#if 0
  auto PAC = PA.getChecker<VariableUniformityAnalysis>();
  bool Preserved = (PAC.preserved() ||
    PAC.preservedSet<AllAnalysesOn<Function>>());
  if (!Preserved) {
    uniformityCache_.erase(&F);
  }
  return !Preserved;
#endif
}

REGISTER_NEW_FANALYSIS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
