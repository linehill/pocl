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
  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;

  WorkitemHandlerType WIH;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;
  llvm::Module *M;
  llvm::Function *F;

  VariableUniformityAnalysisResult &VUA;

  // Temporary global_id_* iteration variables updated by the work-item
  // loops.
  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;

  bool processFunction(llvm::Function &F);

  bool foldTrivialAllocas();
  bool localizePrivateVariables();
  bool fixMultiRegionVariables();
  void releaseParallelRegions();

  // Returns an instruction in the entry block which computes the
  // total size of work-items in the work-group. If it doesn't
  // exist, creates it to the end of the entry block.
  llvm::Instruction *getWorkGroupSizeInstr(llvm::Function &F);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
  createLoopAround(ParallelRegion &Region, llvm::BasicBlock *EntryBB,
                   llvm::BasicBlock *ExitBB, int Dim,
                   llvm::Value *DynamicLocalSize = nullptr);

  llvm::BasicBlock *appendIncBlock(llvm::BasicBlock *After, int Dim,
                                   llvm::BasicBlock *Before = nullptr,
                                   const std::string &BBName = "");

  llvm::Type *recursivelyAlignArrayType(llvm::Type *ArrayType,
                                        llvm::Type *ElementType,
                                        size_t Alignment,
                                        const llvm::DataLayout &Layout);
};

bool WorkitemLoopsImpl::runOnFunction(Function &Func) {

  M = Func.getParent();
  F = &Func;

  WIH = getWorkitemHandler();

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

        if (shouldNotBeContextSaved(&*Instr, VUA, WIH))
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

llvm::Value *
WorkitemLoopsImpl::getLinearWIIndexInRegion(llvm::Instruction *Instr) {
  ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
  assert(ParRegion != nullptr);
  IRBuilder<> Builder(Instr);
  return getLinearWiIndex(Builder, M, ParRegion, WIH);
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
