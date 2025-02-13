// Header for work-item looping functionality.
//
// Copyright (c) 2012 Pekka Jääskeläinen / TUT
//               2022-2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"

// To be moved to DeSPMD:
#include "BarrierTailReplication.h"
#include "CanonicalizeBarriers.h"
#include "ImplicitConditionalBarriers.h"
#include "ImplicitLoopBarriers.h"
#include "LoopBarriers.h"

#include "DebugHelpers.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
#include "pocl_runtime_config.h"

POP_COMPILER_DIAGS

#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "WIL"

#define PASS_NAME "workitemloops"
#define PASS_CLASS pocl::WorkitemLoops
#define PASS_DESC "Workitem loop generation pass"

//#define DEBUG_WORK_ITEM_LOOPS
//#define POCL_KERNEL_COMPILER_DUMP_CFGS

// Use the LLVM_DEBUG-style macros to gradually convert to LLVM-upstreamable
// code.
#ifdef LLVM_DEBUG
#undef LLVM_DEBUG
#endif

#ifdef DEBUG_WORK_ITEM_LOOPS
#define LLVM_DEBUG(X) X
#define dbgs() std::cerr << DEBUG_TYPE << ": "
#else
#define LLVM_DEBUG(X)
#endif

namespace pocl {

using namespace llvm;

class WorkitemLoopsImpl : public pocl::WorkitemHandler {
public:
  WorkitemLoopsImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                    llvm::PostDominatorTree &PDT,
                    VariableUniformityAnalysisResult &VUA)
      : WorkitemHandler(), DT(DT), LI(LI), PDT(PDT), VUA(VUA) {}
  virtual bool runOnFunction(llvm::Function &F);

protected:
  llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr) override;
  llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr,
                                        size_t Dim) override;

private:
  using BasicBlockVector = std::vector<llvm::BasicBlock *>;
  using InstructionIndex = std::set<llvm::Instruction *>;
  using InstructionVec = std::vector<llvm::Instruction *>;
  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;
  llvm::Module *M;
  llvm::Function *F;

  VariableUniformityAnalysisResult &VUA;

  ParallelRegion::ParallelRegionVector OriginalParallelRegions;

  StrInstructionMap ContextArrays;

  // Temporary global_id_* iteration variables updated by the work-item
  // loops.
  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;

  bool processFunction(llvm::Function &F);

  bool foldTrivialAllocas();
  bool localizePrivateVariables();
  bool fixMultiRegionVariables();
  void addContextSaveRestore(llvm::Instruction *Instruction);
  void releaseParallelRegions();

  // Returns an instruction in the entry block which computes the
  // total size of work-items in the work-group. If it doesn't
  // exist, creates it to the end of the entry block.
  llvm::Instruction *getWorkGroupSizeInstr(llvm::Function &F);

  llvm::Value *getLinearWiIndex(llvm::IRBuilder<> &Builder, llvm::Module *M,
                                ParallelRegion *Region);
  llvm::Instruction *addContextSave(llvm::Instruction *Def,
                                    llvm::AllocaInst *AllocaI);

  llvm::Instruction *
  addContextRestore(llvm::Value *Val, llvm::AllocaInst *AllocaI,
                    llvm::Type *LoadInstType, bool PaddingWasAdded,
                    llvm::Instruction *Before = nullptr, bool IsAlloca = false);
  llvm::AllocaInst *getContextArray(llvm::Instruction *Inst,
                                    bool &PoclWrapperStructAdded);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
  createLoopAround(ParallelRegion &Region, llvm::BasicBlock *EntryBB,
                   llvm::BasicBlock *ExitBB, int Dim,
                   llvm::Value *DynamicLocalSize = nullptr);

  llvm::BasicBlock *appendIncBlock(llvm::BasicBlock *After, int Dim,
                                   llvm::BasicBlock *Before = nullptr,
                                   const std::string &BBName = "");

  ParallelRegion *regionOfBlock(llvm::BasicBlock *BB);

  bool shouldNotBeContextSaved(llvm::Instruction *Instr);

  llvm::Type *recursivelyAlignArrayType(llvm::Type *ArrayType,
                                        llvm::Type *ElementType,
                                        size_t Alignment,
                                        const llvm::DataLayout &Layout);

  std::map<llvm::Instruction *, unsigned> TempInstructionIds;
  size_t TempInstructionIndex;
};

bool WorkitemLoopsImpl::runOnFunction(Function &Func) {

  M = Func.getParent();
  F = &Func;

  Initialize(cast<Kernel>(&Func));

  LLVM_DEBUG(dbgs() << "Before WILoops:\n");
  LLVM_DEBUG(Func.dump());

  GlobalIdIterators = {
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(0), ST)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(1), ST)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(2), ST))};

  TempInstructionIndex = 0;

  bool Changed = processFunction(Func);

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(*F, F->getName().str() + "_after_wiloops.dot", nullptr,
          &OriginalParallelRegions);
#endif

  ContextArrays.clear();
  TempInstructionIds.clear();

  releaseParallelRegions();
  LLVM_DEBUG(dbgs() << "After WILoops:\n");
  LLVM_DEBUG(Func.dump());
  eraseInvalidLifetimeMarkers(&Func);
  return Changed;
}

std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoopsImpl::createLoopAround(ParallelRegion &Region,
                                    llvm::BasicBlock *EntryBB,
                                    llvm::BasicBlock *ExitBB, int Dim,
                                    llvm::Value *DynamicLocalSize) {
  Value *LocalIdVar = LocalIdGlobals[Dim];

  size_t LocalSizes[] = {WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ};
  size_t LocalSizeForDim = LocalSizes[Dim];
  Instruction *GlobalIdOrigin = getGlobalIdOrigin(Dim);

  llvm::BasicBlock *LoopBodyEntryBB = EntryBB;
  llvm::LLVMContext &C = LoopBodyEntryBB->getContext();
  llvm::Function *F = LoopBodyEntryBB->getParent();
  LoopBodyEntryBB->setName(std::string("pregion_for_entry.") + EntryBB->getName().str());

  assert (ExitBB->getTerminator()->getNumSuccessors() == 1);

  llvm::BasicBlock *OldExit = ExitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *ForInitBB =
      BasicBlock::Create(C, "pregion_for_init", F, LoopBodyEntryBB);

  llvm::BasicBlock *LoopEndBB =
      BasicBlock::Create(C, "pregion_for_end", F, ExitBB);

  llvm::BasicBlock *ForCondBB =
      BasicBlock::Create(C, "pregion_for_cond", F, ExitBB);

  DT.reset();
  DT.recalculate(*F);

  // Collect the basic blocks in the parallel region that dominate the
  // exit. These are used in determining whether load instructions may
  // be executed unconditionally in the parallel loop (see below).
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> DominatesExitBB;
  for (auto *BB : Region) {
    if (DT.dominates(BB, ExitBB)) {
      DominatesExitBB.insert(BB);
    }
  }

  // For fixing the old edges jumping to the region to jump to the basic block
  // that starts the created loop. Back edges should still point to the old
  // basic block so we preserve the old loops. TODO: is this still needed with
  // the forced PR entry block?
  BasicBlockVector Preds;
  llvm::pred_iterator PI = llvm::pred_begin(EntryBB),
                      E = llvm::pred_end(EntryBB);

  for (; PI != E; ++PI)
    Preds.push_back(*PI);

  for (BasicBlockVector::iterator I = Preds.begin(); I != Preds.end(); ++I) {
    llvm::BasicBlock *BB = *I;
    // Do not fix loop edges inside the region. The loop is replicated as
    // a whole to the body of the WI-loop.
    if (DT.dominates(LoopBodyEntryBB, BB))
      continue;
    BB->getTerminator()->replaceUsesOfWith(LoopBodyEntryBB, ForInitBB);
  }

  IRBuilder<> Builder(ForInitBB);

  Builder.CreateStore(ConstantInt::get(ST, 0), LocalIdVar);

  // Initialize the global id counter with the base.
  GlobalVariable *GlobalId = GlobalIdIterators[Dim];
  Builder.CreateStore(GlobalIdOrigin, GlobalId);

  Builder.CreateBr(LoopBodyEntryBB);

  ExitBB->getTerminator()->replaceUsesOfWith(OldExit, ForCondBB);
  appendIncBlock(ExitBB, Dim);

  Builder.SetInsertPoint(ForCondBB);

  llvm::Value *CmpResult;
  if (!WGDynamicLocalSize) {
    CmpResult = Builder.CreateICmpULT(Builder.CreateLoad(ST, LocalIdVar),
                                      ConstantInt::get(ST, LocalSizeForDim));
  } else {
    GlobalVariable *LocalSizeGlobal = M->getGlobalVariable(LS_G_NAME(Dim));
    if (LocalSizeGlobal == NULL)
      LocalSizeGlobal = new GlobalVariable(
          *M, ST, true, GlobalValue::CommonLinkage, NULL, LS_G_NAME(Dim),
          NULL, GlobalValue::ThreadLocalMode::NotThreadLocal, 0, true);
    CmpResult = Builder.CreateICmpULT(Builder.CreateLoad(ST, LocalIdVar),
                                      Builder.CreateLoad(ST, LocalSizeGlobal));
  }

  Instruction *LoopBranch =
      Builder.CreateCondBr(CmpResult, LoopBodyEntryBB, LoopEndBB);

  if (canAnnotateParallelLoops() && !Region.shouldBeSerialized()) {
    // Add the metadata to mark a parallel loop. The metadata refers to
    // a loop-unique dummy metadata that is not merged automatically.
    // TODO: Merge with the similar code in SubCFGFormation.

    // This creation of the identifier metadata is copied from
    // LLVM's MDBuilder::createAnonymousTBAARoot().

    MDNode *Dummy = MDNode::getTemporary(C, ArrayRef<Metadata *>()).release();
    MDNode *AccessGroupMD = MDNode::getDistinct(C, {});
    MDNode *ParallelAccessMD = MDNode::get(
        C, {MDString::get(C, "llvm.loop.parallel_accesses"), AccessGroupMD});

    MDNode *Root = MDNode::get(C, {Dummy, ParallelAccessMD});

    // At this point we have
    //   !0 = metadata !{}            <- dummy
    //   !1 = metadata !{metadata !0} <- root
    // Replace the dummy operand with the root node itself and delete the dummy.
    Root->replaceOperandWith(0, Root);
    MDNode::deleteTemporary(Dummy);
    // We now have
    //   !1 = metadata !{metadata !1} <- self-referential root
    LoopBranch->setMetadata("llvm.loop", Root);

    auto IsLoadUnconditionallySafe =
        [&DominatesExitBB](llvm::Instruction *Insn) -> bool {
      assert(Insn->mayReadFromMemory());
      // Checks that the instruction isn't in a conditional block.
      return DominatesExitBB.count(Insn->getParent());
    };

    Region.addParallelLoopMetadata(AccessGroupMD, IsLoadUnconditionallySafe);
  }

  Builder.SetInsertPoint(LoopEndBB);
  Builder.CreateBr(OldExit);

  return std::make_pair(ForInitBB, LoopEndBB);
}

ParallelRegion *WorkitemLoopsImpl::regionOfBlock(llvm::BasicBlock *BB) {
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *PRegion = (*PRI);
    if (PRegion->hasBlock(BB))
      return PRegion;
  }
  return nullptr;
}

void WorkitemLoopsImpl::releaseParallelRegions() {
  for (auto PRI = OriginalParallelRegions.begin(),
            PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *P = *PRI;
    delete P;
  }
  OriginalParallelRegions.clear();
}

bool WorkitemLoopsImpl::processFunction(Function &F) {

  releaseParallelRegions();

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  // Append 'dyn' or 'static' to the dot files to differentiate between the
  // dynamic WG one (produced for the binaries) and the specialized static one.
  std::string DotSuffix = WGDynamicLocalSize ? "_dyn" : "_static";
#endif
  dumpCFG(F, F.getName().str() + "_before_pregions" + DotSuffix + ".dot",
          nullptr, nullptr);

  K->getParallelRegions(LI, &OriginalParallelRegions);

  bool Changed = false;

  Changed = handleLocalMemAllocas() || Changed;
  Changed = handleWorkitemFunctions() || Changed;

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_before_wiloops" + DotSuffix + ".dot", nullptr,
          &OriginalParallelRegions);
#endif

  if (foldTrivialAllocas()) {
    Changed = true;
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "#### after trivial alloca folding:\n";
    F.dump();
#endif
  }

  if (localizePrivateVariables()) {
    Changed = true;
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "#### after private variable localization:\n";
    F.dump();
#endif
  }

  if (fixMultiRegionVariables()) {
    Changed = true;
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "#### after multi-region variable fixing:\n";
    F.dump();
#endif
  }

  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {

    ParallelRegion *PRegion = (*PRI);

    LLVM_DEBUG(dbgs() << "Handling region:\n");
    LLVM_DEBUG(PRegion->dumpNames());
    // The original predecessor nodes of which branches should be fixed
    // later on to jump to the looped region's start.
    BasicBlockVector Preds;
    llvm::pred_iterator PI = llvm::pred_begin(PRegion->entryBB()),
                        E = llvm::pred_end(PRegion->entryBB());
    for (; PI != E; ++PI) {
      llvm::BasicBlock *BB = *PI;
      if (DT.dominates(PRegion->entryBB(), BB) &&
          (regionOfBlock(PRegion->entryBB()) == regionOfBlock(BB)))
        continue;
      Preds.push_back(BB);
    }

    // The parallel WI-loop being constructed.
    std::pair<llvm::BasicBlock *, llvm::BasicBlock *> WILoop =
        std::make_pair(PRegion->entryBB(), PRegion->exitBB());

    for (size_t Dim = 0; Dim < 3; ++Dim) {
      WILoop = createLoopAround(*PRegion, WILoop.first, WILoop.second, Dim);
      // Ensure the global id for is initialized even for a 1-size dimension.
      getGlobalIdOrigin(Dim);
    }

    // Fix the predecessors to jump to the beginning of the new WI loop.
    for (BasicBlockVector::iterator I = Preds.begin(); I != Preds.end(); ++I) {
      llvm::BasicBlock *BB = *I;
      BB->getTerminator()->replaceUsesOfWith(PRegion->entryBB(), WILoop.first);
    }
  }

  if (!WGDynamicLocalSize)
    K->addLocalSizeInitCode(WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ);

  ParallelRegion::insertLocalIdInit(&F.getEntryBlock(), 0, 0, 0);

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(*K, K->getName().str() + "_after_wiloops" + DotSuffix + ".dot", nullptr,
          &OriginalParallelRegions);
#endif

  removeBarrierCalls();
  return true;
}

/// Gets rid of trivial temporary variable usages by replacing alloca uses with
/// the written value in the simple cases.
///
/// Currently we consider the alloca "trivial" if gets a single store with
/// a function argument or a constant value.
bool WorkitemLoopsImpl::foldTrivialAllocas() {

  struct AllocaFolding {
    // The alloca to fold.
    llvm::AllocaInst *Alloca;
    // The initializer store.
    llvm::StoreInst *Initializer;
  };

  std::vector<AllocaFolding> Foldings;
  for (auto &BB : *K) {
    for (auto &I : BB) {
      AllocaInst *Alloca = dyn_cast_or_null<AllocaInst>(&I);
      if (Alloca == nullptr)
        continue;

      llvm::StoreInst *Initializer = nullptr;
      for (Instruction::use_iterator UI = Alloca->use_begin(),
                                     UE = Alloca->use_end();
           UI != UE; ++UI) {
        llvm::StoreInst *Store = dyn_cast_or_null<StoreInst>(UI->getUser());
        llvm::LoadInst *Load = dyn_cast_or_null<LoadInst>(UI->getUser());

        if (Store == nullptr && Load == nullptr) {
          // Can handle only stores and loads.
          Initializer = nullptr;
          break;
        }

        if (Store == nullptr)
          continue; // A load.

        if (Initializer != nullptr) {
          // Multiple stores.
          Initializer = nullptr;
          break;
        }

        if (isa<Constant>(Store->getValueOperand()) ||
            isa<Argument>(Store->getValueOperand())) {
          Initializer = Store;
          // Keep scanning so we make sure there are no more writes.
        } else {
          // Unsupported value written.
          Initializer = nullptr;
          break;
        }
      }
      if (Initializer != nullptr)
        Foldings.push_back({Alloca, Initializer});
    }
  }

  for (auto &M : Foldings) {
    for (Instruction::use_iterator UI = M.Alloca->use_begin(),
                                   UE = M.Alloca->use_end();
         UI != UE;) {
      llvm::Instruction *Inst = dyn_cast_or_null<Instruction>(UI->getUser());
      if (Inst == nullptr || Inst == M.Initializer) {
        ++UI;
        continue;
      }
      assert(isa<LoadInst>(Inst));
      // Replace the uses of the alloca load with the value written to the
      // alloca. Note that it's legal to do a wide load from a vector,
      // e.g. 2 x i8 can be loaded to a i16 type. This is why we have to
      // perform a bitcast here sometimes.
      Value *Replacement = M.Initializer->getValueOperand();
      if (Replacement->getType() != Inst->getType()) {
        llvm::IRBuilder<> Builder(Inst);
        Replacement = Builder.CreateBitCast(Replacement, Inst->getType());
      }
      Inst->replaceAllUsesWith(Replacement);
      Inst->eraseFromParent();
      UI = M.Alloca->use_begin();
      UE = M.Alloca->use_end();
    }
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "#### folded an alloca:\n";
    M.Alloca->dump();
#endif
    M.Initializer->eraseFromParent();
    M.Alloca->eraseFromParent();
  }
  return Foldings.size() > 0;
}

/// If there are allocas that are _actually_ used only inside a single PR, but
/// the actual alloca and a potentially single initialization write is in
/// another PR, this function moves the alloca to the PR where it's actually
/// used.
///
/// This helps the context data analysis to decide not to add the alloca to the
/// context data.
bool WorkitemLoopsImpl::localizePrivateVariables() {

  struct AllocaMotion {
    // The alloca to move.
    llvm::AllocaInst *Alloca;
    // Store the initializer (optional) to move.
    llvm::StoreInst *Initializer;
    // The destination parallel region.
    ParallelRegion *Dest;
  };

  std::vector<AllocaMotion> AllocasToMove;

  for (auto &BB : *K) {
    for (auto &I : BB) {
      AllocaInst *Alloca = dyn_cast_or_null<AllocaInst>(&I);
      if (Alloca == nullptr)
        continue;

      ParallelRegion *AllocaRegion = regionOfBlock(Alloca->getParent());
      if (AllocaRegion == nullptr)
        continue;

      ParallelRegion *UsageRegion = nullptr;
      ParallelRegion *AnotherUsageRegion = nullptr;

      llvm::StoreInst *InitializerCandidate = nullptr;

      for (Instruction::use_iterator UI = Alloca->use_begin(),
                                     UE = Alloca->use_end();
           UI != UE; ++UI) {
        llvm::Instruction *User = dyn_cast<Instruction>(UI->getUser());

        if (User == NULL)
          continue;

        llvm::StoreInst *Store = dyn_cast_or_null<StoreInst>(User);

        if (Store != nullptr && LI.getLoopFor(Store->getParent())) {
          // Cannot localize the alloca as it would break the multiple
          // update semantics.
          UsageRegion = nullptr;
          break;
        }

        BasicBlock *UserBB = User->getParent();
        if (isPureUniformBlock(UserBB)) {
          // Variables used in pure unifrom cannot and should not be localized,
          // since they won't be context saved either.
          UsageRegion = nullptr;
          break;
        }

        ParallelRegion *Region = regionOfBlock(UserBB);

        if (Store != nullptr) {
          if (isa<Constant>(Store->getValueOperand())) {
            // Allow only a constant initialization store in the usage region.
            InitializerCandidate = Store;
            continue;
          }
          UsageRegion = nullptr;
          break;
        }

        assert(Region != nullptr);

        if (Region == AllocaRegion) {
          // Either already private alloca or a multi-region variable.
          UsageRegion = Region;
          break;
        }
        if (UsageRegion != nullptr && UsageRegion != Region) {
          // Multi-region variable.
          AnotherUsageRegion = Region;
          break;
        }
        UsageRegion = Region;
      }
      if (UsageRegion != nullptr && UsageRegion != AllocaRegion &&
          AnotherUsageRegion == nullptr &&
          (InitializerCandidate == nullptr ||
           regionOfBlock(InitializerCandidate->getParent()) == UsageRegion))
        AllocasToMove.push_back({Alloca, InitializerCandidate, UsageRegion});
    }
  }
  for (auto &M : AllocasToMove) {
    M.Alloca->moveBefore(M.Dest->entryBB()->getTerminator());
    if (M.Initializer != nullptr)
      M.Initializer->moveAfter(M.Alloca);

#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "#### localized a private variable:\n";
    M.Alloca->dump();
#endif
  }
  return AllocasToMove.size() > 0;
}

/// Add context save/restore code to variables that are defined in the given
/// region and are used outside the region.
bool WorkitemLoopsImpl::fixMultiRegionVariables() {

  InstructionVec ValuesToContextSave;
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {
    ParallelRegion *PRegion = (*PRI);

    InstructionIndex InstructionsInRegion;

    // Construct an index of the region's instructions so it's fast to figure
    // out if the variable uses are all in the region.
    for (BasicBlockVector::iterator I = PRegion->begin(); I != PRegion->end();
         ++I) {
      for (llvm::BasicBlock::iterator Instr = (*I)->begin();
           Instr != (*I)->end(); ++Instr) {
        InstructionsInRegion.insert(&*Instr);
      }
    }

    // Find all the instructions that define new values and check if they need
    // to be context saved.
    for (BasicBlockVector::iterator R = PRegion->begin(); R != PRegion->end();
         ++R) {
      for (llvm::BasicBlock::iterator I = (*R)->begin(); I != (*R)->end();
           ++I) {

        llvm::Instruction *Instr = &*I;

        if (shouldNotBeContextSaved(&*Instr))
          continue;

        for (Instruction::use_iterator UI = Instr->use_begin(),
                                       UE = Instr->use_end();
             UI != UE; ++UI) {
          llvm::Instruction *User = dyn_cast<Instruction>(UI->getUser());

          if (User == NULL)
            continue;

          if ((InstructionsInRegion.find(User) == InstructionsInRegion.end() &&
               regionOfBlock(User->getParent()) != NULL)) {
            ValuesToContextSave.push_back(Instr);
            break;
          }
        }
      }
    }
  }
  // Finally generate the context save/restore (or rematerialization) code for
  // the instructions requiring it.
  for (auto &I : ValuesToContextSave) {
    LLVM_DEBUG(dbgs() << "#### Adding context/save restore for\n");
    LLVM_DEBUG(I->dump());
    addContextSaveRestore(I);
  }
  return ValuesToContextSave.size() > 0;
}

// TO CLEAN: Refactor into getLinearWIIndexInRegion.
llvm::Value *WorkitemLoopsImpl::getLinearWiIndex(llvm::IRBuilder<> &Builder,
                                                 llvm::Module *M,
                                                 ParallelRegion *Region) {
  GlobalVariable *LocalSizeXPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_x", ST));
  GlobalVariable *LocalSizeYPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_y", ST));

  assert(LocalSizeXPtr != NULL && LocalSizeYPtr != NULL);

  LoadInst *LoadX = Builder.CreateLoad(ST, LocalSizeXPtr, "ls_x");
  LoadInst *LoadY = Builder.CreateLoad(ST, LocalSizeYPtr, "ls_y");

  /* Form linear index from xyz coordinates:
       local_size_x * local_size_y * local_id_z  (z dimension)
     + local_size_x * local_id_y                 (y dimension)
     + local_id_x                                (x dimension)
  */
  Value* LocalSizeXTimesY =
    Builder.CreateBinOp(Instruction::Mul, LoadX, LoadY, "ls_xy");

  Value *ZPart =
      Builder.CreateBinOp(Instruction::Mul, LocalSizeXTimesY,
                          Region->getOrCreateIDLoad(LID_G_NAME(2)), "tmp");

  Value *YPart =
      Builder.CreateBinOp(Instruction::Mul, LoadX,
                          Region->getOrCreateIDLoad(LID_G_NAME(1)), "ls_x_y");

  Value* ZYSum =
    Builder.CreateBinOp(Instruction::Add, ZPart, YPart,
                        "zy_sum");

  return Builder.CreateBinOp(Instruction::Add, ZYSum,
                             Region->getOrCreateIDLoad(LID_G_NAME(0)),
                             "linear_xyz_idx");
}

llvm::Value *
WorkitemLoopsImpl::getLinearWIIndexInRegion(llvm::Instruction *Instr) {
  ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
  assert(ParRegion != nullptr);
  IRBuilder<> Builder(Instr);
  return getLinearWiIndex(Builder, M, ParRegion);
}

llvm::Instruction *
WorkitemLoopsImpl::getLocalIdInRegion(llvm::Instruction *Instr, size_t Dim) {
  ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
  if (ParRegion != nullptr) {
    return ParRegion->getOrCreateIDLoad(LID_G_NAME(Dim));
  }
  IRBuilder<> Builder(Instr);
  return Builder.CreateLoad(ST, LocalIdGlobals[Dim]);
}

/// Adds a value store to the context array after the given defining
/// instruction.
///
/// \param Def The instruction that defines the original value.
/// \param AllocaI The alloca created for for the context array.
llvm::Instruction *
WorkitemLoopsImpl::addContextSave(llvm::Instruction *Def,
                                  llvm::AllocaInst *AllocaI) {

  if (isa<AllocaInst>(Def)) {
    // If the variable to be context saved is itself an alloca, we have created
    // one big alloca that stores the data of all the work-items and return
    // pointers to that array. Thus, we need no initialization code other than
    // the context data alloca itself.
    return NULL;
  }

  /* Save the produced variable to the array. */
  BasicBlock::iterator Definition = (dyn_cast<Instruction>(Def))->getIterator();
  ++Definition;
  while (isa<PHINode>(Definition))
    ++Definition;

  // TO CLEAN: Refactor by calling CreateContextArrayGEP.
  IRBuilder<> Builder(&*Definition);
  std::vector<llvm::Value *> GepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *Region = regionOfBlock(Def->getParent());
  assert("Adding context save outside any region produces illegal code." &&
         Region != NULL);

  if (WGDynamicLocalSize) {
    Module *M = AllocaI->getParent()->getParent()->getParent();
    GepArgs.push_back(getLinearWiIndex(Builder, M, Region));
  } else {
    GepArgs.push_back(ConstantInt::get(ST, 0));
    GepArgs.push_back(Region->getOrCreateIDLoad(LID_G_NAME(2)));
    GepArgs.push_back(Region->getOrCreateIDLoad(LID_G_NAME(1)));
    GepArgs.push_back(Region->getOrCreateIDLoad(LID_G_NAME(0)));
  }

  return Builder.CreateStore(
      Def,
      builder.CreateGEP(AllocaI->getAllocatedType(), AllocaI, gepArgs));
}

llvm::Instruction *WorkitemLoopsImpl::addContextRestore(
    llvm::Value *Val, llvm::AllocaInst *AllocaI, llvm::Type *LoadInstType,
    bool PaddingWasAdded, llvm::Instruction *Before, bool IsAlloca) {

  assert(Before != nullptr);

  llvm::Instruction *GEP =
      createContextArrayGEP(AllocaI, Before, PaddingWasAdded);
  if (IsAlloca) {
    // In case the context saved instruction was an alloca, we created a
    // context array with pointed-to elements, and now want to return a
    // pointer to the elements to emulate the original alloca.
    return GEP;
  }
  IRBuilder<> Builder(Before);
  return Builder.CreateLoad(LoadInstType, GEP);
}

/// Returns the context array (alloca) for the given \param Inst, creates it if
/// not found.
///
/// \param PaddingAdded will be set to true in case a wrapper struct was
/// added for padding in order to enforce proper alignment to the elements of
/// the array. Such padding might be needed to ensure aligned accessed from
/// single work-items accessing aggregates in the context data.
llvm::AllocaInst *WorkitemLoopsImpl::getContextArray(llvm::Instruction *Inst,
                                                     bool &PaddingAdded) {
  PaddingAdded = false;

  std::ostringstream Var;
  Var << ".";

  if (std::string(Inst->getName().str()) != "") {
    Var << Inst->getName().str();
  } else if (TempInstructionIds.find(Inst) != TempInstructionIds.end()) {
    Var << TempInstructionIds[Inst];
  } else {
    // Unnamed temp instructions need a name generated for the context array.
    // Create one using a running integer.
    TempInstructionIds[Inst] = TempInstructionIndex++;
    Var << TempInstructionIds[Inst];
  }

  Var << ".wi_context";
  std::string CArrayName = Var.str();

  if (ContextArrays.find(CArrayName) != ContextArrays.end())
    return ContextArrays[CArrayName];

  BasicBlock &Entry = K->getEntryBlock();
  return ContextArrays[CArrayName] = createAlignedAndPaddedContextAlloca(
             Inst, &*(Entry.getFirstInsertionPt()), CArrayName, PaddingAdded);
}

/// Tries to rematerialize the given value-defining instruction.
///
/// Rematerialization in this context means recomputing the value produced
/// in the use site instead of storing and loading a once-computed variable
/// from the context.
///
/// \param Before the instruction before which the cloned instructions should
/// be added. Can be nullptr if only testing for rematerialization-ability.
/// \param Def is the produced value to attempt to clone recursively.
/// \param NamePrefix a prefix string to add to the name of the cloned
/// instructions.
/// \param CanDoIt can be set to a true-initialized boolean in which case the
/// cloning is not actually done, but only its possibility is investigated.
/// \param Depth the recursion depth. Used to limit rematerialization size.
/// \return The rematerialized instruction if possible and beneficial.
static llvm::Value *tryToRematerialize(llvm::Instruction *Before,
                                       llvm::Value *Def,
                                       const std::string &NamePrefix,
                                       bool *CanDoIt = nullptr,
                                       int *Depth = 0) {

  auto DbgRemat = [=](const std::string &Reason) {
    LLVM_DEBUG(dbgs() << "##### " << Reason << "\n");
    LLVM_DEBUG(Def->dump());
  };

#define UNABLE_TO_REMAT(REASON)                                                \
  do {                                                                         \
    DbgRemat("cannot remat: " REASON);                                         \
    if (CanDoIt != nullptr)                                                    \
      *CanDoIt = false;                                                        \
    return nullptr;                                                            \
  } while (0)

#define ABLE_TO_REMAT()                                                        \
  do {                                                                         \
    if (CanDoIt != nullptr)                                                    \
      return nullptr;                                                          \
  } while (0)

  // A call without arguments: Setup a pre-check before cloning to see if we
  // can succeed.
  if (CanDoIt == nullptr && Depth == nullptr) {
    bool Able = true;
    int Depth = 0;
    tryToRematerialize(Before, Def, NamePrefix, &Able, &Depth);
    if (!Able)
      return nullptr;
    Depth = 0;
    return tryToRematerialize(Before, Def, NamePrefix, nullptr, &Depth);
  }

  // Limit the height of the cloned instruction tree to avoid counter-
  // productive rematerialization.
  if (Depth != nullptr && *Depth > 10)
    UNABLE_TO_REMAT("too deep");

  if (llvm::CallInst *Call = dyn_cast<CallInst>(Def)) {
    auto *Callee = Call->getCalledFunction();
    if (Callee == nullptr || (Callee->getName() != GID_BUILTIN_NAME &&
                              Callee->getName() != GS_BUILTIN_NAME &&
                              Callee->getName() != GROUP_ID_BUILTIN_NAME &&
                              Callee->getName() != LID_BUILTIN_NAME &&
                              Callee->getName() != LS_BUILTIN_NAME)) {
      UNABLE_TO_REMAT("called an unsupported function");
    }
  } else if (isa<Constant>(Def) || isa<Argument>(Def)) {
    ABLE_TO_REMAT();
    // No need to clone a constant or function argument, we can refer to the
    // original directly.
    return Def;
  } else if (isa<AllocaInst>(Def) &&
             !isPureUniformBlock(dyn_cast<AllocaInst>(Def)->getParent())) {
    // The allocas in the pure uniform entry block can be referred to without
    // rematerialization. But other than that we do not yet handle recursive
    // alloca references. Should be an easy and valuable low hanging fruit.
    UNABLE_TO_REMAT("accesses another alloca that we cannot remat");
  }

  llvm::Instruction *Inst = dyn_cast<Instruction>(Def);
  if (Inst == nullptr)
    UNABLE_TO_REMAT("unsupported value type");

  if (Inst->mayWriteToMemory() || Inst->mayHaveSideEffects())
    UNABLE_TO_REMAT("has side-effects");

  if (Depth != nullptr)
    (*Depth)++;

  // If we end up referring to instructions in pure uniform blocks (at
  // least work group allocas are such), let's stop the cloning there
  // and refer to the original.
  if (isPureUniformBlock(Inst->getParent()))
    return Inst;

  llvm::Instruction *Copy = CanDoIt == nullptr ? Inst->clone() : nullptr;
  if (Copy != nullptr) {
    Copy->setName(NamePrefix + ".remat");
    Copy->insertBefore(Inst2InsertPt(Before));
  }
  for (unsigned I = 0; I < Inst->getNumOperands(); ++I) {
    llvm::Value *ClonedArg = tryToRematerialize(Copy, Inst->getOperand(I),
                                                NamePrefix, CanDoIt, Depth);
    if (CanDoIt == nullptr)
      Copy->setOperand(I, ClonedArg);
    else if (!CanDoIt)
      return nullptr;
  }
  return Copy;
}

/// Adds context save/restore code for the value produced by the given
/// instruction.
///
/// First try to rematerialize the value instead of storing it to memory.
/// \todo Refactor to subfunctions one of which handles AllocaInsts and another
/// for the rest.
void WorkitemLoopsImpl::addContextSaveRestore(llvm::Instruction *Def) {

  InstructionVec Uses;
  // Restore the produced variable before each use to ensure the correct
  // context copy is used.

  bool RematCandidate = pocl_get_bool_option("POCL_PREGION_VALUE_REMAT", true);

  // In case of a rematerialized alloca with only a single store, this will have
  // the store that initializes it.
  StoreInst *InitializerStore = nullptr;
  size_t Stores = 0;
  ParallelRegion *PrevStoreRegion = nullptr;

  // Find out the uses to fix first as fixing them invalidates the iterator.
  for (Instruction::use_iterator UI = Def->use_begin(), UE = Def->use_end();
       UI != UE; ++UI) {

    llvm::Instruction *User = cast<Instruction>(UI->getUser());
    if (User == NULL)
      continue;

    ParallelRegion *PRegion = regionOfBlock(User->getParent());

    if (StoreInst *ST = dyn_cast<StoreInst>(User)) {
      // Stores of undefined values need not to be counted as actual stores
      // since what we read from that location after that is undefined,
      // thus could be as well the defined value of the another store.
      if (isa<UndefValue>(ST->getValueOperand()))
        continue;

      Stores++;

      // Another corner case to consider here is the case when an alloca
      // is written inside a conditional block which could be even inside
      // a diverging BB. The first intuition in that case would be to not
      // allow rematerialization since we generally don't know at compile
      // time if the branch is taken or not. However, when it comes to
      // undefined behavior, reads from uninitialized allocas produce undefined,
      // values, meaning that it should be fine to recompute the value
      // unconditionally at its use location, as long as the rematerialized
      // instructions are safe to execute speculatively: The value in the
      // conditional is just one possible value in the set of possible values
      // case the branch was not taken.
      if (Stores == 1) {
        InitializerStore = ST;
      } else {
        InitializerStore = nullptr;
        RematCandidate = false;
        LLVM_DEBUG(dbgs() << "Multiple stores\n");
        LLVM_DEBUG(User->dump());
      }
      if (PrevStoreRegion == nullptr) {
        PrevStoreRegion = PRegion;
      } else if (PrevStoreRegion != PRegion) {
        RematCandidate = false;
        LLVM_DEBUG(dbgs() << "Stores from multiple regions\n");
        LLVM_DEBUG(User->dump());
      }
      if (LI.getLoopFor(ST->getParent()) != nullptr) {
        RematCandidate = false;
        LLVM_DEBUG(dbgs() << "Stores inside a loop\n");
        LLVM_DEBUG(User->dump());
      }
    }

    // If the user is in a block that doesn't belong to a region, the variable
    // itself must be a "work group variable", that is, not dependent on the
    // work item. Most likely an iteration variable of a for loop with a
    // barrier.
    if (PRegion == nullptr) {
      LLVM_DEBUG(dbgs() << "User in a pure uniform block?\n");
      LLVM_DEBUG(User->dump());
      continue;
    }

    if (isa<CallInst>(User)) {
      if (!User->isLifetimeStartOrEnd()) {
        RematCandidate = false;
        LLVM_DEBUG(dbgs() << "Using in an unknown call\n");
        LLVM_DEBUG(User->dump());
      }
    } else if (llvm::AllocaInst *Alloca = dyn_cast<AllocaInst>(Def)) {
      if (!isa<StoreInst>(User) && !isa<LoadInst>(User)) {
        RematCandidate = false;
        LLVM_DEBUG(dbgs() << "Taking address of the alloca?\n");
        LLVM_DEBUG(User->dump());
      } else {
        // If we perform reinterpret casts, let's not rematerialize as it might
        // require to store the value temporarily to stack.
        if ((isa<LoadInst>(User) &&
             User->getType() != Alloca->getAllocatedType()) ||
            (isa<StoreInst>(User) &&
             User->getOperand(0)->getType() != Alloca->getAllocatedType())) {
          LLVM_DEBUG(dbgs() << "Found a user with a different pointee type\n");
          LLVM_DEBUG(User->dump());
          LLVM_DEBUG(Def->dump());
          RematCandidate = false;
        }
      }
    }
    Uses.push_back(User);
  }

  if (RematCandidate && isa<AllocaInst>(Def)) {
    bool CanRemat = true;
    int Depth = 0;
    tryToRematerialize(nullptr, InitializerStore->getValueOperand(), "",
                       &CanRemat, &Depth);

    if (!CanRemat) {
      LLVM_DEBUG(dbgs() << "Cannot remat the initializer.\n");
      LLVM_DEBUG(InitializerStore->getValueOperand());
      RematCandidate = false;
    }
  }

  // Used for tracking the alloca load the instruction refers to.
  std::map<Instruction *, Instruction *> OrigAllocaLoads;
  if (RematCandidate && isa<AllocaInst>(Def)) {
    // When rematerializing an Alloca, rematerialize the users of all loads
    // from the alloca to ensure the values will be resurrected in correct PRs.
    InstructionVec AllocaContentUses;

    for (Instruction *User : Uses) {
      auto *Load = dyn_cast<LoadInst>(User);
      if (Load == nullptr)
        continue;

      for (Use &LU : Load->uses()) {

        llvm::Instruction *LoadResUser = cast<Instruction>(LU.getUser());
        if (LoadResUser == NULL)
          continue;

        LLVM_DEBUG(dbgs() << "Remat at the load result usage.\n");
        LLVM_DEBUG(LoadResUser->dump());
        AllocaContentUses.push_back(LoadResUser);
        OrigAllocaLoads[LoadResUser] = Load;
      }
      continue;
    }
    Uses = AllocaContentUses;
  }

  llvm::AllocaInst *ContextArrayAlloca = nullptr;
  bool PaddingAdded = false;

  for (Instruction *UserI : Uses) {
    Instruction *ContextRestoreLocation = UserI;

    // We break down the PHIs, shouldn't see them here.
    assert(!isa<PHINode>(UserI));

    llvm::Value *RematerializedValue = nullptr;
    if (RematCandidate) {
      if (isa<AllocaInst>(Def)) {
        LLVM_DEBUG(dbgs() << "Rematerializing an alloca use which has a single "
                             "initializer store.\n");
        LLVM_DEBUG(dbgs() << "     Alloca:"; Def->dump());
        LLVM_DEBUG(dbgs() << "Initializer:"; InitializerStore->dump());
        LLVM_DEBUG(dbgs() << "        Use:"; UserI->dump());
        RematerializedValue = tryToRematerialize(
            ContextRestoreLocation, InitializerStore->getValueOperand(),
            Def->getName().str());
        assert(RematerializedValue != nullptr);
      } else {
        RematerializedValue = tryToRematerialize(ContextRestoreLocation, Def,
                                                 Def->getName().str());
      }
    }

    if (RematerializedValue != nullptr) {
      LLVM_DEBUG(dbgs() << "Successful rematerialization:\n");
      LLVM_DEBUG(Def->dump());
      LLVM_DEBUG(RematerializedValue->dump());

      if (isa<AllocaInst>(Def)) {
        if (StoreInst *Store = dyn_cast<StoreInst>(UserI)) {
          // The original store could be left intact, but then we'd need to
          // figure out the materialization-ability beforehand.
          Store->setOperand(0, RematerializedValue);
        } else if (UserI->isLifetimeStartOrEnd()) {
          // We can leave the original lifetime marker for the alloca as is.
        } else {
          // We can get rid of the alloca load altogether and use the
          // rematerialized value directly. Replace the uses of the original
          // load value with the particular rematerialized one.
          UserI->replaceUsesOfWith(OrigAllocaLoads[UserI], RematerializedValue);
          LLVM_DEBUG(dbgs()
                     << "Alloca load's use was converted to a remat value:\n");
          LLVM_DEBUG(UserI->dump());
          LLVM_DEBUG(RematerializedValue->dump());
        }
      } else {
        UserI->replaceUsesOfWith(Def, RematerializedValue);
        LLVM_DEBUG(dbgs() << "The user was converted to a remat value:\n");
        LLVM_DEBUG(UserI->dump());
      }
    } else {
      // Unable to rematerialize the value.
      // Allocate a context data array for the variable.
      if (ContextArrayAlloca == nullptr) {
        ContextArrayAlloca = getContextArray(Def, PaddingAdded);
        addContextSave(Def, ContextArrayAlloca);
      }

      llvm::Value *ContextArrayLoad = addContextRestore(
          UserI, ContextArrayAlloca, Def->getType(), PaddingAdded,
          ContextRestoreLocation, isa<AllocaInst>(Def));

      UserI->replaceUsesOfWith(Def, ContextArrayLoad);

      LLVM_DEBUG(dbgs() << "the user was converted to a context load:\n");
      LLVM_DEBUG(UserI->dump());
    }
  }
}

bool WorkitemLoopsImpl::shouldNotBeContextSaved(llvm::Instruction *Instr) {

  if (isa<BranchInst>(Instr)) return true;

  if (AllocaInst *Alloca = dyn_cast<AllocaInst>(Instr)) {
    // Some of the variables such as B-loop iterators must not be
    // replicated for correctness.
    if (VUA.isPureUniformAlloca(Alloca))
      return true;
  }

  // Generated id loads should not be replicated as it leads to problems in
  // conditional branch case where the header node of the region is shared
  // across the peeled branches and thus the header node's ID loads might get
  // context saved which leads to egg-chicken problems.
  auto *Load = dyn_cast<llvm::LoadInst>(Instr);
  if (Load != NULL && (Load->getPointerOperand() == LocalIdGlobals[0] ||
                       Load->getPointerOperand() == LocalIdGlobals[1] ||
                       Load->getPointerOperand() == LocalIdGlobals[2] ||
                       Load->getPointerOperand() == GlobalIdGlobals[0] ||
                       Load->getPointerOperand() == GlobalIdGlobals[1] ||
                       Load->getPointerOperand() == GlobalIdGlobals[2]))
    return true;

  // In case of uniform variables (same value for all work-items), there is no
  // point to create a context array slot for them, but just use the original
  // value everywhere.

  // Allocas are problematic since they include the de-phi induction variables
  // of the b-loops. In those case each work item has a separate loop iteration
  // variable in LLVM IR but which is really a parallel region loop invariant.
  // But because we cannot separate such loop invariant variables at this point
  // sensibly, let's just replicate the iteration variable to each work item
  // and hope the latter optimizations reduce them back to a single induction
  // variable outside the parallel loop.
  if (!VUA.shouldBePrivatized(Instr->getParent()->getParent(), Instr)) {
    LLVM_DEBUG(dbgs() << "Based on VUA, not context saving:\n");
    LLVM_DEBUG(Instr->dump());
    return true;
  }

  return false;
}

/// Appends a local id loop incrementing basic block.
///
/// \param After the basic block which flows to the increment block.
/// \param Dim the local id dimension to increment.
/// \param Before the basic block before which to add the new one.
/// \param BBName name to give to the basic block.
llvm::BasicBlock *WorkitemLoopsImpl::appendIncBlock(llvm::BasicBlock *After,
                                                    int Dim,
                                                    llvm::BasicBlock *Before,
                                                    const std::string &BBName) {

  llvm::Value *LocalIdVar = LocalIdGlobals[Dim];
  llvm::GlobalVariable *GlobalIdVar = GlobalIdIterators[Dim];

  llvm::LLVMContext &C = After->getContext();

  llvm::BasicBlock *OldExit = After->getTerminator()->getSuccessor(0);
  assert(OldExit != NULL);

  llvm::BasicBlock *ForIncBb =
      BasicBlock::Create(C, "pregion_for_inc", After->getParent());

  After->getTerminator()->replaceUsesOfWith(OldExit, ForIncBb);

  IRBuilder<> Builder(OldExit);

  Builder.SetInsertPoint(ForIncBb);
  // Create the iteration variable increment for both the local and global ids.
  Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(ST, LocalIdVar),
                                        ConstantInt::get(ST, 1)),
                      LocalIdVar);

  Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(ST, GlobalIdVar),
                                        ConstantInt::get(ST, 1)),
                      GlobalIdVar);

  Builder.CreateBr(OldExit);

  return ForIncBb;
}

/// Identifies regions between barrier calls and loops that are annotated
/// parallel around them.
///
/// The loops can be then loop vectorized easily with standard LLVM IR
/// vectorization passes.
bool addWorkItemLoops(llvm::Function &F, llvm::DominatorTree &DT,
                      llvm::PostDominatorTree &PDT, llvm::LoopInfo &LI,
                      VariableUniformityAnalysisResult &VUA) {
  
  WorkitemLoopsImpl WIL(DT, LI, PDT, VUA);

  return WIL.runOnFunction(F);
}

#if 0
bool WorkitemLoops::canHandleKernel(llvm::Function &K,
                                    llvm::FunctionAnalysisManager &AM) {

  // The below cases should be now manageable. TODO: update the check for the
  // unhandled case(s).
  // Do not handle kernels with barriers inside loops which have early exits
  // or continues.
  // It would require additional complexity that is unlikely worth it since
  // the vectorizer won't produce efficient code for such loops anyhow.
  // Tested by tricky_for.cl.
  LoopInfo &LI = AM.getResult<llvm::LoopAnalysis>(K);
  for (auto *L : LI) {
    if (!Barrier::isLoopWithBarrier(*L))
      continue;
    // More than one 'break' point. It would lead to a complex control flow
    // structure which likely ruins loopvec efficiency anyhow.
    if (L->getExitingBlock() == nullptr) {
      LLVM_DEBUG(
          dbgs() << "Multiple breaks inside a barrier loop, won't handle.\n");
      return false;
    }
  }

  // The tricky part here is detecting cases where we have barriers inside
  // uniform ifs inside for-loops of which iteration counts are not known.
  // For the purpose of this check, we treat them as always taken for-loops,
  // relying on the "all or none" barrier semantics. The current checks
  // is "robustness first": It includes all barriers which are made
  // conditional with something else than the loop condition. An optimization
  // would be to allow detected uniform conditions and isolate the if part
  // to a uniform region with a separate parallel region in both branches.

  llvm::PostDominatorTree &PDT = AM.getResult<PostDominatorTreeAnalysis>(K);
  for (Function::iterator FI = K.begin(), FE = K.end(); FI != FE; ++FI) {
    BasicBlock *BB = &*FI;

    for (const auto &Instr : *BB) {
      // (Post)dominator analysis that is used in multiple places gets
      // confused by 'unreachable' instructions. Fall back if finding them.
      // TO DO: Convert unreachables to returns or similar. Or just remove
      // them. They should not be reached after all, so it's undefined
      // what happens if they are.
      if (isa<UnreachableInst>(Instr))
        return false;
    }

    if (!Barrier::hasBarrier(BB)) continue;

    // Unconditional barrier for this purpose postdominates the entry node or
    // the loop header that it's in.
    Loop *L = LI.getLoopFor(BB);
    BasicBlock *PostDomBlock = BB;
    if (L != nullptr) {
      // Treat the loop body separately: If the control flow of a single WI
      // goes there, the rest should follow. Entering the first loop body
      // block matters here.
      PostDomBlock = L->getHeader();
      // If it's a for-loop, the header is an exiting block as well and
      // we need to find the body block.
      if (PostDomBlock->getTerminator()->getNumSuccessors() > 1)
        PostDomBlock =
            LI.getLoopFor(PostDomBlock->getTerminator()->getSuccessor(0)) == L
                ? PostDomBlock->getTerminator()->getSuccessor(0)
                : PostDomBlock->getTerminator()->getSuccessor(1);
      assert(LI.getLoopFor(PostDomBlock) == L);
    }

    if (PDT.dominates(BB, PostDomBlock)) continue;

#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### Detected a conditional barrier not currently supported "
              << "by WILoops:\n";
    BB->dump();
#endif
    return false;
  }
  return true;
}
#endif

} // namespace pocl
