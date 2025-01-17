// LLVM function pass to replicate barrier tails (successors to barriers).
//
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012-2019 Pekka Jääskeläinen
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
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>

#include "Barrier.h"
#include "BarrierTailReplication.h"
#include "CanonicalizeBarriers.h"
#include "DebugHelpers.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"

POP_COMPILER_DIAGS

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#define PASS_NAME "barriertails"
#define PASS_CLASS pocl::BarrierTailReplication
#define PASS_DESC "Barrier tail replication pass"

namespace pocl {

using namespace llvm;

class BarrierTailReplicationImpl {

public:
  bool runOnFunction(llvm::Function &F);
  BarrierTailReplicationImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI)
      : DT(DT), LI(LI){};

private:
  typedef std::set<llvm::BasicBlock *> BasicBlockSet;
  typedef std::vector<llvm::BasicBlock *> BasicBlockVector;
  typedef std::map<llvm::Value *, llvm::Value *> ValueValueMap;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::Function *F;

  bool ProcessFunction(llvm::Function &F);
  bool processBarriersDFS(llvm::BasicBlock *BB, BasicBlockSet &ProcessedBBs);
  bool ReplicateJoinedSubgraphs(llvm::BasicBlock *Dominator,
                                llvm::BasicBlock *SubgraphEntry,
                                BasicBlockSet &ProcessedBBs);

  llvm::BasicBlock *replicateTail(llvm::BasicBlock *Entry, llvm::Function *F);
  void findTailBlocks(BasicBlockVector &Subgraph, llvm::BasicBlock *Entry);
  void ReplicateBasicBlocks(BasicBlockVector &NewGraph,
                            llvm::ValueToValueMapTy &ReferenceMap,
                            BasicBlockVector &Graph, llvm::Function *F);
  void UpdateReferences(const BasicBlockVector &Graph,
                        llvm::ValueToValueMapTy &ReferenceMap);

  bool CleanupPHIs(llvm::BasicBlock *BB);
};

bool BarrierTailReplicationImpl::runOnFunction(Function &Func) {

  F = &Func;

  bool Changed = pocl::canonicalizeBarriers(Func);

  if (Changed) {
    DT.recalculate(*F);
    LI.releaseMemory();
    LI.analyze(DT);
  }

#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### Before barrier tail replication:\n";
  Func.dump();
#endif

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(Func, Func.getName().str() + "_before_btr.dot", nullptr, nullptr);
#endif

  Changed = ProcessFunction(Func) || Changed;

  LI.verify(DT);
  // The created tails might contain PHI nodes with operands
  // referring to the non-predecessor (split point) BB.
  // These must be cleaned to avoid breakage later on.
  for (Function::iterator I = Func.begin(), E = Func.end(); I != E; ++I)
    Changed |= CleanupPHIs(&*I);

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(Func, Func.getName().str() + "_after_btr.dot", nullptr, nullptr);
#endif

  if (Changed) {
    pocl::canonicalizeBarriers(Func);
#ifdef DEBUG_BARRIER_REPL
    std::cerr << "### After barrier tail replication:\n";
    Func.dump();
#endif
  }

  return Changed;
}

bool BarrierTailReplicationImpl::ProcessFunction(Function &F) {
  BasicBlockSet ProcessedBBs;
  return processBarriersDFS(&F.getEntryBlock(), ProcessedBBs);
}

/// Recursively (depht-first) look for barriers in all possible
/// execution paths starting on entry, replicating the barrier
/// successors to ensure there is a separate function exit BB
/// for each combination of traversed barriers.
///
/// \p ProcessedBBs stores the already traversed barriers.
bool BarrierTailReplicationImpl::processBarriersDFS(
    BasicBlock *BB, BasicBlockSet &ProcessedBBs) {
  bool Changed = false;

  // Check if we already visited this BB to avoid infinite recursion in
  // case of unbarriered loops.
  if (ProcessedBBs.count(BB) != 0)
    return Changed;

  ProcessedBBs.insert(BB);

  if (Barrier::hasBarrier(BB)) {
#ifdef DEBUG_BARRIER_REPL
    std::cerr << "#### BB " << BB->getName().str()
              << " has a barrier, replicate the tail" << std::endl;
#endif
    BasicBlockSet ProcessedBBsRJS;
    Changed |= ReplicateJoinedSubgraphs(BB, BB, ProcessedBBsRJS);
  }

  auto t = BB->getTerminator();

  // Find barriers in the successors (depth first).
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i)
    Changed |= processBarriersDFS(t->getSuccessor(i), ProcessedBBs);

  return Changed;
}

/// Only replicate those parts of the subgraph that are not dominated by
/// a (barrier) basic block, to avoid excessive (and confusing) code
/// duplication.
bool BarrierTailReplicationImpl::ReplicateJoinedSubgraphs(
    BasicBlock *Dominator, BasicBlock *SubgraphEntry,
    BasicBlockSet &ProcessedBBs) {
  bool Changed = false;

  assert(DT.dominates(Dominator, SubgraphEntry));

  Function *F = Dominator->getParent();

  auto Term = SubgraphEntry->getTerminator();
  for (int i = 0, e = Term->getNumSuccessors(); i != e; ++i) {
    BasicBlock *BB = Term->getSuccessor(i);
#ifdef DEBUG_BARRIER_REPL
    std::cerr << "### traversing from " << SubgraphEntry->getName().str()
              << " to " << BB->getName().str() << std::endl;
#endif

    // Check if we already handled this BB and all its branches.
    if (ProcessedBBs.count(BB) != 0) {
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "### already processed " << std::endl;
#endif
      continue;
    }

    const bool isBackedge = DT.dominates(BB, SubgraphEntry);
    if (isBackedge) {
      // This is a loop backedge. Do not traverse.
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "### a loop backedge, skipping" << std::endl;
#endif
      continue;
    }
    if (DT.dominates(Dominator, BB)) {
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "### " << Dominator->getName().str() << " dominates "
                << BB->getName().str() << std::endl;
#endif
      Changed |= ReplicateJoinedSubgraphs(Dominator, BB, ProcessedBBs);
    } else {
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "#### " << Dominator->getName().str()
                << " does not dominate " << BB->getName().str()
                << " replicating " << std::endl;
#endif
      BasicBlock *OrigTailEntry = BB;
      BasicBlock *NewTailEntry = replicateTail(OrigTailEntry, F);

      Term->setSuccessor(i, NewTailEntry);
      Changed = true;

      // Next we'll choose whether the other basic blocks that branched to
      // the old tail block should keep branching to the original, or
      // should be switched to branch to the new one.

      // In case of blocks that are dominated by the same barrier, we should
      // include them in the same parallel region as the new joined one,
      // otherwise control flow breakage might occur: We might cause the other
      // side of a diverging branch to go to a separate parallel region.
      // See test_id_dependent_computation.cpp which reproduces such a case due
      // to the diverging branch after the barrier both of which share the same
      // exit barrier with the early exiting branch due to control flow
      // merging.
      std::vector<BasicBlock *> IncludedPredecessors;
      for (pred_iterator PI = pred_begin(OrigTailEntry),
                         PE = pred_end(OrigTailEntry);
           PI != PE; ++PI) {
        llvm::BasicBlock *PredBB = *PI;

        if (!DT.dominates(Dominator, PredBB))
          continue;

#ifdef DEBUG_BARRIER_REPL
        std::cerr << "#### " << Dominator->getName().str() << " dominates "
                  << OrigTailEntry->getName().str() << "\n";
        std::cerr << "#### fixing it to branch to the new tail\n";
#endif
        IncludedPredecessors.push_back(PredBB);
      }

      for (auto &PredBB : IncludedPredecessors) {
        auto PredTerm = PredBB->getTerminator();
        for (int i = 0, e = PredTerm->getNumSuccessors(); i != e; ++i) {
          BasicBlock *OrigSucc = PredTerm->getSuccessor(i);
          if (OrigSucc == OrigTailEntry) {
            PredTerm->setSuccessor(i, NewTailEntry);
            break;
          }
        }
      }
    }

    if (Changed) {
#ifdef DEBUG_BARRIER_REPL
      static int SubgraphCase = 0;
      std::cerr << "#### Replicated a subgraph #" << SubgraphCase << "\n";
      dumpCFG(*F,
              F->getName().str() + "_btr_repl_case_" +
                  std::to_string(SubgraphCase) + ".dot",
              nullptr, nullptr);
      ++SubgraphCase;
#endif
      // We modified the function. Possibly created new loops and possibly
      // now some barriers do have new dominating barriers.
      // Update analysis passes.
      DT.reset();
      DT.recalculate(*F);
      LI.releaseMemory();
      LI.analyze(DT);
    }
  }
  ProcessedBBs.insert(SubgraphEntry);
  return Changed;
}

/// Removes phi elements for which there are no successors anymore due
/// to replication removing a join point.
bool BarrierTailReplicationImpl::CleanupPHIs(llvm::BasicBlock *BB) {

  bool Changed = false;
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### CleanupPHIs for BB:" << std::endl;
  BB->dump();
#endif

  for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
    PHINode *PN = dyn_cast<PHINode>(BI);
    if (PN == NULL)
      break;

    bool PHIRemoved = false;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i) {
      bool isSuccessor = false;
      // find if the predecessor branches to this one (anymore)
      for (unsigned
               S = 0,
               SE =
                   PN->getIncomingBlock(i)->getTerminator()->getNumSuccessors();
           S < SE; ++S) {
        if (PN->getIncomingBlock(i)->getTerminator()->getSuccessor(S) == BB) {
          isSuccessor = true;
          break;
        }
      }
      if (!isSuccessor) {
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "removing incoming value " << i
                  << " from PHINode:" << std::endl;
        PN->dump();
#endif
        PN->removeIncomingValue(i, true);
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "now:" << std::endl;
        PN->dump();
#endif
        Changed = true;
        e--;
        if (e == 0) {
          PHIRemoved = true;
          break;
        }
        i = 0;
        continue;
      }
    }
    if (PHIRemoved)
      BI = BB->begin();
    else
      BI++;
  }
  return Changed;
}

BasicBlock *BarrierTailReplicationImpl::replicateTail(BasicBlock *Entry,
                                                      Function *F) {
  BasicBlockVector Tail;
  findTailBlocks(Tail, Entry);

  // Replicate subgraph maintaining control flow.
  BasicBlockVector V;

  ValueToValueMapTy VVM;
  ReplicateBasicBlocks(V, VVM, Tail, F);
  UpdateReferences(V, VVM);

  // Return entry block of replicated subgraph.
  return cast<BasicBlock>(VVM[Entry]);
}

/// Finds basic blocks to tail replicate from a given \p Entry point.
///
/// Traverses from the given replication point down to the exit.
/// TODO: Reduce duplication by traversing until the next shared barrier.
void BarrierTailReplicationImpl::findTailBlocks(BasicBlockVector &Subgraph,
                                                BasicBlock *Entry) {
  // The subgraph can have internal branches (join points) avoid replicating
  // these parts multiple times within the same tail.
  if (std::count(Subgraph.begin(), Subgraph.end(), Entry) > 0)
    return;

  Subgraph.push_back(Entry);

  auto Terminator = Entry->getTerminator();
  for (unsigned I = 0, E = Terminator->getNumSuccessors(); I != E; ++I) {
    BasicBlock *Successor = Terminator->getSuccessor(I);
    const bool isBackedge = DT.dominates(Successor, Entry);
    if (isBackedge) continue;
    findTailBlocks(Subgraph, Successor);
  }
}

void BarrierTailReplicationImpl::ReplicateBasicBlocks(
    BasicBlockVector &NewGraph, ValueToValueMapTy &ReferenceMap,
    BasicBlockVector &Graph, Function *F) {
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### ReplicateBasicBlocks: " << std::endl;
#endif
  for (BasicBlockVector::const_iterator I = Graph.begin(),
         E = Graph.end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    BasicBlock *NewBB = BasicBlock::Create(BB->getContext(),
             BB->getName() + ".btr",
             F);
    ReferenceMap.insert(std::make_pair(BB, NewBB));
    NewGraph.push_back(NewBB);

#ifdef DEBUG_BARRIER_REPL
    std::cerr << "Replicated BB: " << NewBB->getName().str() << std::endl;
#endif

    for (BasicBlock::iterator I2 = BB->begin(), E2 = BB->end();
         I2 != E2; ++I2) {
      Instruction *Inst = I2->clone();
      ReferenceMap.insert(std::make_pair(&*I2, Inst));
      Inst->insertInto(NewBB, NewBB->end());
    }

    // Add predicates to PHINodes of basic blocks the replicated block jumps
    // to (backedges).
    auto Terminator = NewBB->getTerminator();
    for (unsigned I = 0, E = Terminator->getNumSuccessors(); I != E; ++I) {
      BasicBlock *Successor = Terminator->getSuccessor(I);
      if (std::count(Graph.begin(), Graph.end(), Successor) == 0) {
        // Successor is not in the graph, possible backedge.
        for (BasicBlock::iterator BBI = Successor->begin(),
                                  BBE = Successor->end();
             BBI != BBE; ++BBI) {
          PHINode *Phi = dyn_cast<PHINode>(BBI);
          if (Phi == NULL)
            break; // All PHINodes already checked.

          // Get value for original incoming edge and add new predicate.
          Value *OldV = Phi->getIncomingValueForBlock(BB);
          Value *NewV = ReferenceMap.find(OldV) == ReferenceMap.end() ?
            NULL : ReferenceMap[OldV];

          if (NewV == NULL) {
            // This case can happen at least when replicating a latch block
            // in a b-loop. The value produced might be from a common path
            // before the replicated part. Then just use the original value.
            NewV = OldV;
          }
          Phi->addIncoming(NewV, NewBB);
        }
      }
    }
  }
}

void BarrierTailReplicationImpl::UpdateReferences(
    const BasicBlockVector &Graph, ValueToValueMapTy &ReferenceMap) {
  for (BasicBlockVector::const_iterator BBVI = Graph.begin(),
                                        BBVE = Graph.end();
       BBVI != BBVE; ++BBVI) {
    BasicBlock *BB = *BBVI;
    for (BasicBlock::iterator BBI = BB->begin(), BBE = BB->end();
         BBI != BBE; ++BBI) {
      Instruction *Inst = &*BBI;
      RemapInstruction(Inst, ReferenceMap,
                       RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
    }
  }
}

llvm::PreservedAnalyses
BarrierTailReplication::run(llvm::Function &F,
                            llvm::FunctionAnalysisManager &FAM) {
  if (!isKernelToProcess(F))
    return PreservedAnalyses::all();

  WorkitemHandlerType WIH = FAM.getResult<WorkitemHandlerChooser>(F).WIH;
  if (WIH == WorkitemHandlerType::CBS)
    return PreservedAnalyses::all();

  llvm::DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  llvm::LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);

  BarrierTailReplicationImpl BTRI(DT, LI);

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_before_btr.dot", nullptr, nullptr);
#endif

  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();
  PAChanged.preserve<LoopAnalysis>();
  PAChanged.preserve<DominatorTreeAnalysis>();

  return BTRI.runOnFunction(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
