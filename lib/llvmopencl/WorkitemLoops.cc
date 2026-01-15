// Header for work-item looping functionality.
//
// Copyright (c) 2012 Pekka Jääskeläinen / TUT
//               2022-2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include "DebugHelpers.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "LoopBarriers.h"
#include "SubgroupBarrier.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkgroupBarrier.h"
#include "WorkitemHandler.h"
#include "WorkitemLoops.h"
#include "pocl_runtime_config.h"

POP_COMPILER_DIAGS

#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "DeSPMD-WIL"

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
  llvm::Instruction *getGlobalIdInRegion(llvm::Instruction *Instr,
                                         size_t Dim) override;

private:
  using BasicBlockVector = std::vector<llvm::BasicBlock *>;
  using InstructionIndex = std::set<llvm::Instruction *>;
  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;

  int StaticSubGSize;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;
  llvm::Module *M;
  llvm::Function *F;

  llvm::GlobalVariable *SgIntraCounter;
  llvm::GlobalVariable *SgInterCounter;
  llvm::GlobalVariable *SgSize;

  llvm::GlobalVariable *YUpperLimit;
  llvm::GlobalVariable *YLowerLimit;

  llvm::GlobalVariable *NXLanes;

  // Total number of subgroup parallel regions in the kernel.
  int SgTotalRegions = 0;
  // Number of subgroup parallel regions that dont have references
  // to 3D indices.
  int LinearizedSGRegions = 0;

  VariableUniformityAnalysisResult &VUA;

  // Temporary global_id_* iteration variables updated by the work-item
  // loops.
  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;
  bool processFunction(llvm::Function &F);

  bool localizePrivateVariables();
  bool fixMultiRegionVariables();
  void releaseParallelRegions();

  // Returns an instruction in the entry block which computes the
  // total size of work-items in the work-group. If it doesn't
  // exist, creates it to the end of the entry block.
  llvm::Instruction *getWorkGroupSizeInstr(llvm::Function &F);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
  createLinearSGLoopAround(ParallelRegion &Region, llvm::BasicBlock *EntryBB,
                           llvm::BasicBlock *ExitBB);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
  createLoopAround(ParallelRegion &Region, llvm::BasicBlock *EntryBB,
                   llvm::BasicBlock *ExitBB, int Dim);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
  createInterSGLoopAround(llvm::BasicBlock *EntryBB, llvm::BasicBlock *ExitBB);

  void appendPRIncBlock(llvm::BasicBlock *ExitBB, int Dim, int PRid,
                        bool SGRegion, bool Linear);

  void appendSRInCBlock(llvm::BasicBlock *After);

  void locateHierarchicalRegionBorders(
      std::vector<std::pair<llvm::BasicBlock *, llvm::BasicBlock *>>
          &SuperRegions);

  llvm::Type *recursivelyAlignArrayType(llvm::Type *ArrayType,
                                        llvm::Type *ElementType,
                                        size_t Alignment,
                                        const llvm::DataLayout &Layout);

  size_t getStaticSubGroupSize();

  void addUnpackLLIDCode(llvm::IRBuilder<> &Builder);

  llvm::Value *addDynamicSizeLoadCode(llvm::IRBuilder<> &Builder, int Dim);

  void collectLoopCounterStores(
      std::vector<llvm::StoreInst *> &LoopCountersToHandle);

  void annotateRegionAsParallel(
      ParallelRegion &Region,
      llvm::SmallPtrSet<llvm::BasicBlock *, 8> &DominatesExitBB,
      Instruction *LoopBranch);
};

void collectOperands(llvm::Value *V, std::set<llvm::Value *> &Visited) {
  if (!V || Visited.count(V))
    return;
  Visited.insert(V);

  if (auto *I = llvm::dyn_cast<llvm::Instruction>(V)) {
    for (llvm::Use &U : I->operands()) {
      collectOperands(U.get(), Visited);
    }
  }
}

/// Add the metadata to mark a parallel loop.
/// The metadata refers to a loop-unique dummy metadata that is not merged
/// automatically.
/// TODO: Merge with the similar code in SubCFGFormation.
/// This creation of the identifier metadata is copied from
/// LLVM's MDBuilder::createAnonymousTBAARoot().
/// \param Region the parallel region which is marked as parallel.
/// \param DominatesExitBB the set of basic blocks within the parallel region
/// that dominate the parallel region exit block. \param LoopBranch the
/// condition check instruction of the loop.
void WorkitemLoopsImpl::annotateRegionAsParallel(
    ParallelRegion &Region,
    llvm::SmallPtrSet<llvm::BasicBlock *, 8> &DominatesExitBB,
    Instruction *LoopBranch) {

  llvm::LLVMContext &C = Region.entryBB()->getContext();

  MDNode *Dummy = MDNode::getTemporary(C, ArrayRef<Metadata *>()).release();
  MDNode *AccessGroupMD = MDNode::getDistinct(C, {});
  MDNode *ParallelAccessMD = MDNode::get(
      C, {MDString::get(C, "llvm.loop.parallel_accesses"), AccessGroupMD});

  MDNode *Root = MDNode::get(C, {Dummy, ParallelAccessMD});

  // At this point we have
  //   !0 = metadata !{}            <- dummy
  //   !1 = metadata !{metadata !0} <- root
  // Replace the dummy operand with the root node itself and delete the
  // dummy.
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

/// Adds code for loading the dynamic size of the workgroup for the given
/// dimension.
///
/// \param Builder used to create the load.
/// \param Dim the workgroup dimension.
llvm::Value *
WorkitemLoopsImpl::addDynamicSizeLoadCode(llvm::IRBuilder<> &Builder, int Dim) {
  GlobalVariable *LocalSizeGlobal = M->getGlobalVariable(LS_G_NAME(Dim));
  if (LocalSizeGlobal == nullptr)
    LocalSizeGlobal = new GlobalVariable(
        *M, ST, true, GlobalValue::CommonLinkage, nullptr, LS_G_NAME(Dim),
        nullptr, GlobalValue::ThreadLocalMode::NotThreadLocal, 0, true);
  llvm::Value *DynSizeLoad = Builder.CreateLoad(ST, LocalSizeGlobal);

  return DynSizeLoad;
}

static Value *getWILoopLowerBound(llvm::IRBuilder<> &B, int Dim) {
  // Note: When static local sizes are present, they are used to initialize
  // loop-bound variables at kernel entry and propagated by optimizations.

  auto *M = B.GetInsertBlock()->getParent()->getParent();
  auto *BoundGV = getOrCreateWILoopLowerBoundGV(M, Dim);

  return B.CreateLoad(BoundGV->getValueType(), BoundGV,
                      Twine("wiloop.lbound.") + Twine(Dim));
}

static Value *getWILoopUpperBound(llvm::IRBuilder<> &B, int Dim) {
  // Note: When static local sizes are present, they are used to initialize
  // loop-bound variables at kernel entry and propagated by optimizations.

  auto *M = B.GetInsertBlock()->getParent()->getParent();
  auto *BoundGV = getOrCreateWILoopUpperBoundGV(M, Dim);

  return B.CreateLoad(BoundGV->getValueType(), BoundGV,
                      Twine("wiloop.ubound.") + Twine(Dim));
}

/// Collects initial loop iterator store instructions for the kernel loops.
///
/// For each loop that contains SG barriers, trace and store the the initial
/// store instruction of the loop iterator variable. \param LoopCountersToHandle
/// vector where store instructions are collected.
void WorkitemLoopsImpl::collectLoopCounterStores(
    std::vector<llvm::StoreInst *> &LoopCountersToHandle) {

  for (llvm::Loop *OuterLoop : LI) {
    auto Loops = OuterLoop->getLoopsInPreorder();

    for (llvm::Loop *L : Loops) {
      BasicBlock *Latch = L->getLoopLatch();

      // Nothing to do if we dont have SG-barriers.
      if (!SubgroupBarrier::isLoopWithSGBarrier(*L))
        continue;

      if (Latch == nullptr)
        continue;

      llvm::StoreInst *Store;
      // Search for the loop iterator increment here:
      BasicBlock *IncBlock = Latch->getSinglePredecessor();

      for (auto &I : *IncBlock) {
        Store = dyn_cast_or_null<StoreInst>(&I);
        // The store of updated loop iterator.
        if (Store) {
          Value *PtrVal = Store->getPointerOperand();
          if (auto *PtrInst = dyn_cast<Instruction>(PtrVal)) {
            // The stack alloca where loop iterator is stored.
            if (auto *Alloca = dyn_cast<AllocaInst>(PtrInst)) {
              // Look for any other (than the updated) store.
              // That should be the initial store instruction.
              for (Instruction::use_iterator UI = Alloca->use_begin(),
                                             UE = Alloca->use_end();
                   UI != UE; ++UI) {
                llvm::User *User = UI->getUser();
                if (auto *StoreInstr = dyn_cast<StoreInst>(User))
                  if (StoreInstr != Store)
                    LoopCountersToHandle.push_back(StoreInstr);
              }
            }
          }
        }
      }
    }
  }
}

/// Add unpacking code to derive {local|global}_id_{x|y|z} from the local linear
/// id.
///
/// The current local linear ID (LLID) is used to derive the 3D workgroup
/// indices. Handles static and dynamic workgroup dimensions. \param Builder the
/// builder used to insert the instructions.
void WorkitemLoopsImpl::addUnpackLLIDCode(llvm::IRBuilder<> &Builder) {

  // The fallback case. Unpack the 3 dimensional ids from the LLID.
  // This is not vectorizable.
  llvm::Value *LinearID = Builder.CreateLoad(ST, LLID);

  llvm::Value *StatSizeX = llvm::ConstantInt::get(ST, WGLocalSizeX);
  llvm::Value *StatSizeY = llvm::ConstantInt::get(ST, WGLocalSizeY);
  llvm::Value *StatSizeZ = llvm::ConstantInt::get(ST, WGLocalSizeZ);

  llvm::Value *DynSizeX = addDynamicSizeLoadCode(Builder, 0);
  llvm::Value *DynSizeY = addDynamicSizeLoadCode(Builder, 1);
  llvm::Value *DynSizeZ = addDynamicSizeLoadCode(Builder, 2);

  llvm::Value *XSize = WGDynamicLocalSize ? DynSizeX : StatSizeX;
  llvm::Value *YSize = WGDynamicLocalSize ? DynSizeY : StatSizeY;
  llvm::Value *ZSize = WGDynamicLocalSize ? DynSizeZ : StatSizeZ;

  llvm::Value *DerivedLX = Builder.CreateURem(LinearID, XSize);
  llvm::Value *TempY = Builder.CreateUDiv(LinearID, XSize);
  llvm::Value *DerivedLY = Builder.CreateURem(TempY, YSize);

  llvm::Value *TempZ = Builder.CreateMul(XSize, YSize);
  llvm::Value *DerivedLZ = Builder.CreateUDiv(LinearID, TempZ);

  Builder.CreateStore(DerivedLX, LocalIdGlobals[0]);
  Builder.CreateStore(DerivedLY, LocalIdGlobals[1]);
  Builder.CreateStore(DerivedLZ, LocalIdGlobals[2]);

  llvm::Instruction *XOff = getGlobalIdOrigin(0);
  llvm::Instruction *YOff = getGlobalIdOrigin(1);
  llvm::Instruction *ZOff = getGlobalIdOrigin(2);

  llvm::Value *DerivedGX = Builder.CreateAdd(XOff, DerivedLX);
  llvm::Value *DerivedGY = Builder.CreateAdd(YOff, DerivedLY);
  llvm::Value *DerivedGZ = Builder.CreateAdd(ZOff, DerivedLZ);

  GlobalVariable *GlobalIdX = GlobalIdIterators[0];
  GlobalVariable *GlobalIdY = GlobalIdIterators[1];
  GlobalVariable *GlobalIdZ = GlobalIdIterators[2];

  Builder.CreateStore(DerivedGX, GlobalIdX);
  Builder.CreateStore(DerivedGY, GlobalIdY);
  Builder.CreateStore(DerivedGZ, GlobalIdZ);
}

/// Get the compile-time subgroup size.
size_t WorkitemLoopsImpl::getStaticSubGroupSize() {

  // If 'intel_reqd_sub_group_size' is not provided, subgroup size defaults to
  // x-size.
  size_t SgSizeRet = WGLocalSizeX;

  if (llvm::MDNode *SGSizeMD = K->getMetadata("intel_reqd_sub_group_size")) {
    if (SGSizeMD->getNumOperands() > 0) {
      if (auto *ConstMd = llvm::dyn_cast<llvm::ConstantAsMetadata>(
              SGSizeMD->getOperand(0))) {
        if (auto *ConstInt =
                llvm::dyn_cast<llvm::ConstantInt>(ConstMd->getValue())) {
          size_t SGSize = ConstInt->getZExtValue();
          SgSizeRet = SGSize;
        }
      }
    }
  }
  return SgSizeRet;
}

/// Determine the entry-exit barrier pairs that constitute the hierarchical
/// regions.
///
/// Hierarchical region is a collection of consecutive parallel regions
/// that are executed one subgroup at a time. Parallel regions inside
/// hierarchial region are referred here as subgroup parallel regions and they
/// are executed in subgroup loops. \param SuperRegions container for barrier
/// pairs.
void WorkitemLoopsImpl::locateHierarchicalRegionBorders(
    std::vector<std::pair<llvm::BasicBlock *, llvm::BasicBlock *>>
        &SuperRegions) {

  // Collect super region entry/exit barriers first:
  std::vector<llvm::BasicBlock *> SuperRegionEntrys;
  std::vector<llvm::BasicBlock *> SuperRegionExits;

  // Some SG-barries are artifacts, collect them here. This is just temporary
  // fix. Make it so that they dont exist in the first place.
  std::vector<llvm::BasicBlock *> FlaggedSGbarriers;

  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {

    ParallelRegion *PRegion = (*PRI);

    llvm::BasicBlock *EntryBarr = PRegion->entryBB()->getSinglePredecessor();
    llvm::BasicBlock *ExitBarr =
        PRegion->exitBB()->getTerminator()->getSuccessor(0);

    // Find the entry and exit barriers for the hierarchical regions (That
    // consist of 'subgroup' regions).
    if (EntryBarr != nullptr && ExitBarr != nullptr) {

      bool EntryHasWGB = WorkgroupBarrier::hasWGBarrier(EntryBarr);
      bool ExitHasWGB = WorkgroupBarrier::hasWGBarrier(ExitBarr);

      bool EntryHasSGB = SubgroupBarrier::hasSGBarrier(EntryBarr);
      bool ExitHasSGB = SubgroupBarrier::hasSGBarrier(ExitBarr);

      // (WG)->[PR]->(SG)
      if (EntryHasWGB && ExitHasSGB) {
        if (std::find(SuperRegionEntrys.begin(), SuperRegionEntrys.end(),
                      ExitBarr) == SuperRegionEntrys.end())
          SuperRegionEntrys.push_back(ExitBarr);

        // (SG)->[PR]->(WG) case
      } else if (EntryHasSGB && ExitHasWGB) {
        if (std::find(SuperRegionExits.begin(), SuperRegionExits.end(),
                      EntryBarr) == SuperRegionExits.end())
          SuperRegionExits.push_back(EntryBarr);

        // [PR]->(SG)->(WG). Cornercase, just before the exit barrier.
      } else if (ExitHasSGB) {

        // Check that this is not function exit block.
        if (ExitBarr->getTerminator()->getNumSuccessors() > 0)
          if (WorkgroupBarrier::hasWGBarrier(
                  ExitBarr->getTerminator()->getSuccessor(0)))
            if (std::find(SuperRegionExits.begin(), SuperRegionExits.end(),
                          ExitBarr) == SuperRegionExits.end())
              SuperRegionExits.push_back(ExitBarr);

      } else if (ExitHasWGB) {
        if (ExitBarr->getTerminator()->getNumSuccessors() > 0)
          if (SubgroupBarrier::hasSGBarrier(
                  ExitBarr->getTerminator()->getSuccessor(0)))
            if (std::find(SuperRegionEntrys.begin(), SuperRegionEntrys.end(),
                          ExitBarr->getTerminator()->getSuccessor(0)) ==
                SuperRegionEntrys.end())
              SuperRegionEntrys.push_back(
                  ExitBarr->getTerminator()->getSuccessor(0));
      }
    }
  }

  assert(SuperRegionEntrys.size() == SuperRegionExits.size() &&
         "Failed to locate SuperRegion entries/exits!");

  // In case of single entry/exit, pairing is simple.
  if (SuperRegionEntrys.size() == 1) {
    SuperRegions.push_back(
        std::make_pair(SuperRegionEntrys[0], SuperRegionExits[0]));
    // For multiple regions, we have to search the corresponding pairs:
  } else {
    // For each entry block, determine which exit block can be reached from it.
    for (llvm::BasicBlock *Entry : SuperRegionEntrys) {
      std::vector<llvm::BasicBlock *> Worklist;
      std::vector<llvm::BasicBlock *> Visited;
      Worklist.push_back(Entry);

      while (!Worklist.empty()) {
        llvm::BasicBlock *Current = Worklist.back();
        Worklist.pop_back();
        Visited.push_back(Current);

        if (std::find(SuperRegionExits.begin(), SuperRegionExits.end(),
                      Current) != SuperRegionExits.end()) {
          SuperRegions.push_back(std::make_pair(Entry, Current));
          break;
        }

        for (llvm::BasicBlock *Succ : successors(Current)) {
          if (std::find(Visited.begin(), Visited.end(), Succ) == Visited.end())
            Worklist.push_back(Succ);
        }
      }
    }
  }
}

bool WorkitemLoopsImpl::runOnFunction(Function &Func) {

  M = Func.getParent();
  F = &Func;

  initialize(cast<Kernel>(&Func), WorkitemHandlerType::LOOPS);

  LLVM_DEBUG(dbgs() << "Before WILoops:\n");
  LLVM_DEBUG(Func.dump());

  GlobalIdIterators = {
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(0), ST)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(1), ST)),
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(2), ST))};

  TempInstructionIndex = 0;

  SgInterCounter = llvm::cast<llvm::GlobalVariable>(
      M->getOrInsertGlobal("_sg_inter_counter", ST));

  SgSize =
      llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(SG_S_NAME, ST));

  SgIntraCounter = llvm::cast<llvm::GlobalVariable>(
      M->getOrInsertGlobal(SG_INTRA_C_NAME, ST));

  NXLanes =
      llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal("_n_x_lanes", ST));

  YUpperLimit = llvm::cast<llvm::GlobalVariable>(
      M->getOrInsertGlobal("_sg_y_upper_limit", ST));

  YLowerLimit = llvm::cast<llvm::GlobalVariable>(
      M->getOrInsertGlobal("_sg_y_lower_limit", ST));

  LLID =
      llvm::cast<llvm::GlobalVariable>(M->getOrInsertGlobal(LLID_G_NAME, ST));

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
/// Creates a linear loop around single parallel (subgroup) region.
///
/// \param Region the parallel region for which the linear loop is applied.
/// \param EntryBB the basic block from which the loop starts.
/// \param ExitBB the basic block to which the loop ends.
std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoopsImpl::createLinearSGLoopAround(ParallelRegion &Region,
                                            llvm::BasicBlock *EntryBB,
                                            llvm::BasicBlock *ExitBB) {

  llvm::BasicBlock *LoopBodyEntryBB = EntryBB;
  llvm::LLVMContext &C = LoopBodyEntryBB->getContext();
  llvm::Function *F = LoopBodyEntryBB->getParent();
  assert(hasInvariantWILoopBounds(F) && "UNSUPPORTED: dynamic WI-loop bounds");

  std::string Prefix = "SG-pregion_";

  LoopBodyEntryBB->setName(Prefix + std::to_string(Region.getID()) +
                           std::string("_for_entry_Linear"));

  llvm::BasicBlock *OldExit = ExitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *ForInitBB = BasicBlock::Create(
      C, Prefix + std::to_string(Region.getID()) + "_for_init_Linear", F,
      LoopBodyEntryBB);

  llvm::BasicBlock *ForCondBB = BasicBlock::Create(
      C, Prefix + std::to_string(Region.getID()) + "_for_cond_Linear", F,
      ExitBB);

  llvm::BasicBlock *LoopEndBB = BasicBlock::Create(
      C, Prefix + std::to_string(Region.getID()) + "_for_end_Linear", F,
      ExitBB);

  DT.reset();
  DT.recalculate(*F);

  llvm::SmallPtrSet<llvm::BasicBlock *, 8> DominatesExitBB;
  for (auto *BB : Region) {
    if (DT.dominates(BB, ExitBB)) {
      DominatesExitBB.insert(BB);
    }
  }

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

  // Handle the initialisation block.
  IRBuilder<> Builder(ForInitBB);
  Builder.CreateStore(ConstantInt::get(ST, 0), SgIntraCounter);

  // Add code for loading the local IDs here, if needed.
  // 3D indices are required if there are references to them within the parallel
  // region.
  if (Region.HasLocalIDReferences())
    addUnpackLLIDCode(Builder);

  Builder.CreateBr(LoopBodyEntryBB);

  ExitBB->getTerminator()->replaceUsesOfWith(OldExit, ForCondBB);

  appendPRIncBlock(ExitBB, 0, Region.getID(), true, true);

  // Handle the condition block.
  Builder.SetInsertPoint(ForCondBB);

  llvm::Value *CmpResult;

  // Add code for comparing the subgroup intra loop counter.
  llvm::Value *SubgroupSize = Builder.CreateLoad(ST, SgSize);
  llvm::Value *SGIntraC = Builder.CreateLoad(ST, SgIntraCounter);
  CmpResult = Builder.CreateICmpULT(SGIntraC, SubgroupSize);

  Instruction *LoopBranch;
  LoopBranch = Builder.CreateCondBr(CmpResult, LoopBodyEntryBB, LoopEndBB);

  if (canAnnotateParallelLoops() && !Region.shouldBeSerialized())
    annotateRegionAsParallel(Region, DominatesExitBB, LoopBranch);

  // Handle the end block:
  Builder.SetInsertPoint(LoopEndBB);

  // After this region, this subgroup will possibly execute another SG-region
  // so we will need to reset the linear id to that of the first work-item of
  // this subgroup. This can be obtained from the SgInterCounter variable.
  llvm::Value *Rst = Builder.CreateLoad(ST, SgInterCounter);
  Builder.CreateStore(Rst, LLID);

  Builder.CreateBr(OldExit);

  return std::make_pair(ForInitBB, LoopEndBB);
}

/// Creates a loop between two basic blocks.
///
/// Works either with WI-loops or SG-loops.
/// 1. Work-item loops over parallel regions.
/// 2. Subgroup work-item-loops over parallel regions.
///   - This is otherwise same as 1, but looping is done withing subgroup, not
///   within workgroup.
/// \param Region the parallel region for which wi-loop is applied.
/// \param EntryBB the basic block from which the loop starts.
/// \param ExitBB the basic block to which the loop ends.
/// \param Dim the dimension of the loop.
std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoopsImpl::createLoopAround(ParallelRegion &Region,
                                    llvm::BasicBlock *EntryBB,
                                    llvm::BasicBlock *ExitBB, int Dim) {

  bool SGRegion = Region.isSGRegion();

  if (SGRegion)
    assert(hasInvariantWILoopBounds(F) &&
           "UNSUPPORTED: dynamic WI-loop bounds");

  Value *LocalIdVar = LocalIdGlobals[Dim];

  size_t LocalSizes[] = {WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ};
  size_t LocalSizeForDim = LocalSizes[Dim];
  Instruction *GlobalIdOrigin = getGlobalIdOrigin(Dim);

  llvm::BasicBlock *LoopBodyEntryBB = EntryBB;
  llvm::LLVMContext &C = LoopBodyEntryBB->getContext();
  llvm::Function *F = LoopBodyEntryBB->getParent();

  std::string Prefix = SGRegion ? "SG-pregion_" : "WG-pregion_";

  if (Dim == 0)
    LoopBodyEntryBB->setName(Prefix + std::to_string(Region.getID()) +
                             std::string("_for_entry"));

  llvm::BasicBlock *OldExit = ExitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *ForInitBB =
      BasicBlock::Create(C,
                         Prefix + std::to_string(Region.getID()) +
                             "_for_init_" + std::string(1, 'X' + Dim),
                         F, LoopBodyEntryBB);

  llvm::BasicBlock *ForCondBB =
      BasicBlock::Create(C,
                         Prefix + std::to_string(Region.getID()) +
                             "_for_cond_" + std::string(1, 'X' + Dim),
                         F, ExitBB);

  llvm::BasicBlock *LoopEndBB =
      BasicBlock::Create(C,
                         Prefix + std::to_string(Region.getID()) + "_for_end_" +
                             std::string(1, 'X' + Dim),
                         F, ExitBB);

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

  // Handle the initialisation block.
  IRBuilder<> Builder(ForInitBB);

  GlobalVariable *GlobalId = GlobalIdIterators[Dim];

  // In the SG-loop variant, we are looping a partial WI-loop.
  if (SGRegion) {

    // For the x and z dim, we always zero in the init.
    if (Dim == 1) {
      // The y-dim is looped from YLowerLimit to YUpperLimit. They are
      // determined in the outer (inter) loop.
      Builder.CreateStore(Builder.CreateLoad(ST, YLowerLimit), LocalIdVar);
      Builder.CreateStore(
          Builder.CreateAdd(GlobalIdOrigin,
                            Builder.CreateLoad(ST, YLowerLimit)),
          GlobalId);
    } else {
      Builder.CreateStore(ConstantInt::get(ST, 0), LocalIdVar);
      Builder.CreateStore(GlobalIdOrigin, GlobalId);
    }

    // The WI-loops variant:
  } else {
    auto *LBound = getWILoopLowerBound(Builder, Dim);
    Builder.CreateStore(LBound, LocalIdVar);
    // Initialize the global id counter with the base.
    // GlobalVariable *GlobalId = GlobalIdIterators[Dim];
    Builder.CreateStore(GlobalIdOrigin, GlobalId);
  }

  Builder.CreateBr(LoopBodyEntryBB);

  ExitBB->getTerminator()->replaceUsesOfWith(OldExit, ForCondBB);

  appendPRIncBlock(ExitBB, Dim, Region.getID(), SGRegion, false);

  // Handle the condition block.
  Builder.SetInsertPoint(ForCondBB);

  llvm::Value *CmpResult;

  if (SGRegion) {
    if (!WGDynamicLocalSize) {
      if (Dim == 0) {
        CmpResult =
            Builder.CreateICmpULT(Builder.CreateLoad(ST, LocalIdVar),
                                  ConstantInt::get(ST, LocalSizeForDim));
      } else if (Dim == 1) {
        CmpResult = Builder.CreateICmpULT(Builder.CreateLoad(ST, LocalIdVar),
                                          Builder.CreateLoad(ST, YUpperLimit));
      }
      // Don't add comparison for Z as Z dim has to be 1 for this looping
      // strategy.
    }
    // WI-Loops looping strategy.
  } else {
    auto *UBound = getWILoopUpperBound(Builder, Dim);
    CmpResult =
        Builder.CreateICmpULT(Builder.CreateLoad(ST, LocalIdVar), UBound);
  }

  Instruction *LoopBranch;

  // Skip the conditional branch for the z dim in sg region:
  if (SGRegion && Dim == 2) {
    LoopBranch = Builder.CreateBr(LoopEndBB);
  } else {
    LoopBranch = Builder.CreateCondBr(CmpResult, LoopBodyEntryBB, LoopEndBB);
  }

  if (canAnnotateParallelLoops() && !Region.shouldBeSerialized())
    annotateRegionAsParallel(Region, DominatesExitBB, LoopBranch);

  Builder.SetInsertPoint(LoopEndBB);

  // For the final end block of the SG region loop, reset the LLID.
  // LLID has to be set correctly between parallel regions because there may be
  // some conditional branches that require context load in the decision making.
  if (SGRegion && Dim == 2)
    Builder.CreateStore(Builder.CreateLoad(ST, SgInterCounter), LLID);
  Builder.CreateBr(OldExit);

  return std::make_pair(ForInitBB, LoopEndBB);
}

/// Create upper level loop between two barriers, spanning multiple parallel
/// regions.
///
/// \param EntryBB the entry barrier to the sg-region.
/// \param ExitBB the exit barrier from the sg-region.
std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoopsImpl::createInterSGLoopAround(llvm::BasicBlock *EntryBB,
                                           llvm::BasicBlock *ExitBB) {

  llvm::BasicBlock *LoopBodyEntryBB = EntryBB;
  llvm::LLVMContext &C = LoopBodyEntryBB->getContext();
  llvm::Function *F = LoopBodyEntryBB->getParent();

  LoopBodyEntryBB->setName(std::string("sg_inter_pregion_for_entry.") +
                           EntryBB->getName().str());

  llvm::BasicBlock *OldExit = ExitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *ForInitBB =
      BasicBlock::Create(C, "sg_inter_pregion_for_init", F, LoopBodyEntryBB);

  llvm::BasicBlock *LoopEndBB =
      BasicBlock::Create(C, "sg_inter_pregion_for_end", F, ExitBB);

  llvm::BasicBlock *ForCondBB =
      BasicBlock::Create(C, "sg_pregion_inter_for_cond", F, ExitBB);

  DT.reset();
  DT.recalculate(*F);

  // Collect the basic blocks in the parallel region that dominate the
  // exit. These are used in determining whether load instructions may
  // be executed unconditionally in the parallel loop (see below).
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

  // Handle the init block, zero all relevant variables.
  IRBuilder<> Builder(ForInitBB);
  for (int Dim = 0; Dim < 3; ++Dim) {
    Builder.CreateStore(ConstantInt::get(ST, 0), LocalIdGlobals[Dim]);
  }

  Builder.CreateStore(ConstantInt::get(ST, 0), SgInterCounter);
  Builder.CreateStore(ConstantInt::get(ST, 0), YLowerLimit);
  Builder.CreateStore(ConstantInt::get(ST, 0), LLID);

  llvm::Value *SGs = Builder.CreateLoad(ST, SgSize);
  llvm::Value *WGSizeXConst = llvm::ConstantInt::get(ST, WGLocalSizeX);
  llvm::Value *Increment = Builder.CreateUDiv(SGs, WGSizeXConst);

  Builder.CreateStore(Increment, NXLanes);

  Builder.CreateStore(Increment, YUpperLimit);

  Builder.CreateBr(LoopBodyEntryBB);

  ExitBB->getTerminator()->replaceUsesOfWith(OldExit, ForCondBB);

  appendSRInCBlock(ExitBB);

  // Handle the conditional block
  Builder.SetInsertPoint(ForCondBB);

  llvm::Value *CmpResult;
  llvm::Value *Counter = Builder.CreateLoad(ST, SgInterCounter);

  // Condition for exiting the inter SG loop.
  // Check if the counter exceeds the total number of work items.
  // Derive the total number of work items in static/dynamic wg cases:
  llvm::Value *NWI;
  if (WGDynamicLocalSize) {
    llvm::Value *XSize = addDynamicSizeLoadCode(Builder, 0);
    llvm::Value *YSize = addDynamicSizeLoadCode(Builder, 1);
    llvm::Value *ZSize = addDynamicSizeLoadCode(Builder, 2);

    llvm::Value *TempMult = Builder.CreateMul(XSize, YSize);
    NWI = Builder.CreateMul(TempMult, ZSize);

  } else {
    NWI =
        llvm::ConstantInt::get(ST, WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ);
  }

  CmpResult = Builder.CreateICmpULT(Counter, NWI);

  Instruction *LoopBranch =
      Builder.CreateCondBr(CmpResult, LoopBodyEntryBB, LoopEndBB);

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

  K->getParallelRegions(LI, &OriginalParallelRegions, VUA);

#ifndef NDEBUG
  for (auto *PR : OriginalParallelRegions)
    PR->verify(/*AbortOnFailure=*/true);
#endif

  bool Changed = false;

  Changed = handleLocalMemAllocas() || Changed;
  Changed = handleWorkitemFunctions() || Changed;

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_before_wiloops" + DotSuffix + ".dot", nullptr,
          &OriginalParallelRegions);
#endif

  std::vector<std::pair<llvm::BasicBlock *, llvm::BasicBlock *>> SuperRegions;

  locateHierarchicalRegionBorders(SuperRegions);

  LLVM_DEBUG(
      dbgs() << "Found " << SuperRegions.size() << " hierarchical regions:\n";
      for (int i = 0; i < SuperRegions.size(); ++i) {
        dbgs() << i << ": " << SuperRegions[i].first->getName().str() << " -> "
               << SuperRegions[i].second->getName().str() << "\n";
      });

  if (localizePrivateVariables()) {
    Changed = true;
    LLVM_DEBUG(dbgs() << "#### after private variable localization:\n");
    LLVM_DEBUG(F.dump());
  }

  if (fixMultiRegionVariables()) {
    Changed = true;
    LLVM_DEBUG(dbgs() << "#### after multi-region variable fixing:\n");
    LLVM_DEBUG(F.dump());
  }

  // In SG-loops, kernel level loops have to be executed separately for each
  // subgroup. In order to achieve this, we have to reset the loop iterators so
  // they appear fresh for each subgroup. This is achieved by collecting the
  // store instructions that set the initial value and then reintroducing these
  // between subgroup executions.
  std::vector<llvm::StoreInst *> LoopCountersToHandle;
  collectLoopCounterStores(LoopCountersToHandle);

  // Create WI-Loops/SG-Loops
  for (ParallelRegion::ParallelRegionVector::iterator
           PRI = OriginalParallelRegions.begin(),
           PRE = OriginalParallelRegions.end();
       PRI != PRE; ++PRI) {

    ParallelRegion *PRegion = (*PRI);

    LLVM_DEBUG(
        dbgs() << "Handling region:\n"; PRegion->dumpNames();
        if (PRegion->isSGRegion()) {
          SgTotalRegions++;
          dbgs() << "Found subgroup parallel region\n";
          if (!PRegion->HasLocalIDReferences()) {
            dbgs() << "Using linearized subgroup inner loop\n";
            LinearizedSGRegions++;
          } else {
            dbgs() << "Using fallback subgroup loop\n";
          }
        } else {
          dbgs() << "Found workgroup parallel region\n";
          dbgs() << "Using WILoops\n";
        });

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

    // Store entry barrier here so we can modify it later for SG-intra loops.
    llvm::BasicBlock *EntryBarr = PRegion->entryBB()->getSinglePredecessor();

    // Decide how to handle looping over the parallel region:
    // Note: this decision is made per parallel region (PR) basis!
    bool UseLinearLoop = false;

    // For the subgroup regions, there are several options:
    // 1. Optimal case is if we dont have references to 3D id variables.
    //    Here, the PR can be looped by local linear ID, which is nice for
    //    autovectorizer.
    // 2. There are references to 3D IDs, BUT the workgroup size is well-aligned
    // with the subgroup size.
    //    Currently, well-aligned means that the local x size divides subgroup
    //    size evenly AND individual subgroups are always restricted on a single
    //    x-y slice. NOTE: This is open for further optimizations in the future.
    //    The current design is optimized for the Gromacs kernels.
    // 3. The fallback case occurs when the kernel references 3D IDs, and the
    // subgroup size does not divide evenly into the workgroup size.
    //    In this case, iteration is performed using the local linear ID, but
    //    the corresponding 3D IDs must be unpacked on each iteration — making
    //    the loop non-vectorizable.
    // Note that the linear loop is used for the most optimal outcome as well as
    // the fallback. Only difference is whether we have to unpack the 3D
    // indices, which makes vectorization impossible.
    if (PRegion->isSGRegion()) {
      assert(hasInvariantWILoopBounds(&F) &&
             "UNSUPPORTED: dynamic WI-loop bounds");

      // Case (1) - The most optimal, vectorizable. Can be with static or
      // dynamic wg sizes.
      if (!PRegion->HasLocalIDReferences()) {
        UseLinearLoop = true;

      } else {
        // With dynamic sizes, we have to always rely on the linear loop.
        if (WGDynamicLocalSize) {
          // This is the fallback case with dynamic wg sizes. Not vectorizable.
          UseLinearLoop = true;
        } else {

          size_t StaticSgSize = getStaticSubGroupSize();

          // Currently only very specific workgroup/subgroup sizes fit the
          // criteria for case (2):
          // - Local X size has to divide subgroup size evenly.
          // - Subgroup size cannot span multiple z dim slices.
          // - Z dimension size has to be of size 1.
          if (!((StaticSgSize % WGLocalSizeX == 0) &&
                (StaticSgSize <= WGLocalSizeX * WGLocalSizeY) &&
                (WGLocalSizeX * WGLocalSizeY % StaticSgSize == 0) &&
                (WGLocalSizeZ == 1))) {
            // Case (3)
            UseLinearLoop = true;
          }
          // Case (2) -- a well-aligned subgroup size.
          // For example: in case of 4x4x4 work group, well-aligned subgroup
          // sizes would be 4, 8, and 16. Note that the if below is the negation
          // of this case, where we fall back to a non-vectorizable linear loop.
          // So case (2) is implied by the default value of UseLinearLoop.
        }
      }
    }

    // Apply the looping method.
    if (UseLinearLoop) {
      WILoop = createLinearSGLoopAround(*PRegion, WILoop.first, WILoop.second);
      for (size_t Dim = 0; Dim < 3; ++Dim) {
        getGlobalIdOrigin(Dim);
      }
    } else {
      for (size_t Dim = 0; Dim < 3; ++Dim) {
        WILoop = createLoopAround(*PRegion, WILoop.first, WILoop.second, Dim);
        // Ensure the global id for is initialized even for a 1-size dimension.
        getGlobalIdOrigin(Dim);
      }
    }

    // Fix the predecessors to jump to the beginning of the new WI loop.
    for (BasicBlockVector::iterator I = Preds.begin(); I != Preds.end(); ++I) {
      llvm::BasicBlock *BB = *I;
      BB->getTerminator()->replaceUsesOfWith(PRegion->entryBB(), WILoop.first);
    }

    if (!PRegion->isSGRegion()) {
      IRBuilder<> Builder(WILoop.first->getTerminator());
      // if (GlobalVariable *LLID = M->getGlobalVariable(LLID_G_NAME))
      Builder.CreateStore(ConstantInt::get(SizeT(M), 0), LLID);
    }
  }

  LLVM_DEBUG(dbgs() << "Inner loop is linearizable in " << LinearizedSGRegions
                    << " subgroup parallel regions (out of " << SgTotalRegions
                    << " subgroup parallel regions)\n");

  // Create the outer sg-loops, that loop the hierarchical regions one subgroup
  // at a time.
  for (const auto &SRegion : SuperRegions) {
    assert(hasInvariantWILoopBounds(&F) &&
           "UNSUPPORTED: dynamic WI-loop bounds");

    llvm::BasicBlock *EntryBarr = SRegion.first;
    llvm::BasicBlock *ExitBarr = SRegion.second;

    createInterSGLoopAround(SRegion.first, SRegion.second);

    // Let's add the resetting of loop counters.
    for (llvm::StoreInst *SI : LoopCountersToHandle) {
      auto *ClonedStore = SI->clone();
      ClonedStore->insertBefore(
          ExitBarr->getTerminator()->getSuccessor(0)->getTerminator());
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

        if (User == nullptr)
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

    LLVM_DEBUG(dbgs() << "#### localized a private variable:\n");
    LLVM_DEBUG(M.Alloca->dump());
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

          if (User == nullptr)
            continue;

          if ((InstructionsInRegion.find(User) == InstructionsInRegion.end() &&
               regionOfBlock(User->getParent()) != nullptr)) {

            ValuesToContextSave.push_back(Instr);
            break;
          }
          if (regionOfBlock(User->getParent()) == nullptr &&
              SubgroupBarrier::hasSGBarrier(User->getParent())) {
            // Context save anyway, need to handle the actual save somehow as
            // the user is not in the PR.
            ValuesToContextSave.push_back(Instr);
            break;
          }
        }
      }
    }
  }

  // Finally generate the context save/restore (or rematerialization) code for
  // the instructions requiring it. First process alloca instructions as they
  // have influence on rematerialization opportunities on non-alloca
  // instructions.

  std::sort(ValuesToContextSave.begin(), ValuesToContextSave.end(),
            [=](const Instruction *Lhs, const Instruction *Rhs) -> bool {
              return !isa<AllocaInst>(Lhs) < !isa<AllocaInst>(Rhs);
            });

  for (auto &I : ValuesToContextSave) {
    LLVM_DEBUG(dbgs() << "#### Adding context/save restore for\n");
    LLVM_DEBUG(I->dump());
    addContextSaveRestore(I, LI);
  }

  return ValuesToContextSave.size() > 0;
}

llvm::Value *
WorkitemLoopsImpl::getLinearWIIndexInRegion(llvm::Instruction *Instr) {

  ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
  if (ParRegion != nullptr) {
    return ParRegion->getOrCreateIDLoad(LLID_G_NAME);
  }
  IRBuilder<> Builder(Instr);
  return Builder.CreateLoad(ST, LLID);
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

llvm::Instruction *
WorkitemLoopsImpl::getGlobalIdInRegion(llvm::Instruction *Instr, size_t Dim) {
  ParallelRegion *ParRegion = regionOfBlock(Instr->getParent());
  if (ParRegion != nullptr) {
    return ParRegion->getOrCreateIDLoad(GID_G_NAME(Dim));
  }
  IRBuilder<> Builder(Instr);
  return Builder.CreateLoad(ST, GlobalIdGlobals[Dim]);
}

/// Appends an id incrementing basic block for the loop around parallel region.
///
/// Increments the local id, global id and the local linear id.
/// \param After the basic block which flows to the increment block.
/// \param Dim the local id dimension to increment.
/// \param PRid the id of the parallel region.
/// \param SGRegion the flag indicating if this is a subgroup region.
/// \param Linear the flag indicating if looping logic is linear.
void WorkitemLoopsImpl::appendPRIncBlock(llvm::BasicBlock *After, int Dim,
                                         int PRid, bool SGRegion, bool Linear) {

  llvm::LLVMContext &C = After->getContext();

  llvm::BasicBlock *OldExit = After->getTerminator()->getSuccessor(0);
  assert(OldExit != nullptr);

  IRBuilder<> Builder(OldExit);

  std::string RegionTypeStr = SGRegion ? "SG-pregion_" : "WG-pregion_";

  std::string DimStr = Linear ? "Linear" : std::string(1, 'X' + Dim);

  llvm::BasicBlock *ForIncBb = BasicBlock::Create(
      C, RegionTypeStr + std::to_string(PRid) + "_for_inc_" + DimStr,
      After->getParent(), After);

  Builder.SetInsertPoint(ForIncBb);

  // For the linear SG-loop
  if (Linear) {
    // Increment the sg intra loop counter and LLID.
    llvm::Value *IntraLoopCounter = Builder.CreateLoad(ST, SgIntraCounter);
    IntraLoopCounter =
        Builder.CreateAdd(ConstantInt::get(ST, 1), IntraLoopCounter);
    Builder.CreateStore(IntraLoopCounter, SgIntraCounter);
    Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(ST, LLID),
                                          ConstantInt::get(ST, 1)),
                        LLID);

    // Add the unpacking code for 3D indices if needed.
    if (regionOfBlock(After)->HasLocalIDReferences())
      addUnpackLLIDCode(Builder);

    // For the WI-loop style sg-loop:
  } else {

    llvm::Value *LocalIdVar = LocalIdGlobals[Dim];
    llvm::GlobalVariable *GlobalIdVar = GlobalIdIterators[Dim];

    Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(ST, LocalIdVar),
                                          ConstantInt::get(ST, 1)),
                        LocalIdVar);
    Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(ST, GlobalIdVar),
                                          ConstantInt::get(ST, 1)),
                        GlobalIdVar);

    if (Dim == 0)
      Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(ST, LLID),
                                            ConstantInt::get(ST, 1)),
                          LLID);
  }

  After->getTerminator()->replaceUsesOfWith(OldExit, ForIncBb);
  Builder.CreateBr(OldExit);
}

/// Appends increment block for the outer subgroup loop, that spans multiple
/// parallel regions.
///
/// The SR in the name stands for 'Super Region' as opposed to 'Parallel
/// Region'. This creates an increment block for a loop that spans multiple
/// subgroup level parallel regions. These parallel regions can have varying
/// looping strategies: i.e. optimal linear, WI-loops style sg-loops, or
/// fallback linear. Thus we have to prepare for all of them. Most importantly,
/// the special helper variables (used in the non-linear strategy) have to be
/// handled. \param After the basic block which flows to the increment block.
void WorkitemLoopsImpl::appendSRInCBlock(llvm::BasicBlock *After) {

  llvm::LLVMContext &C = After->getContext();

  llvm::BasicBlock *OldExit = After->getTerminator()->getSuccessor(0);
  assert(OldExit != nullptr);

  IRBuilder<> Builder(OldExit);
  llvm::BasicBlock *ForIncBb;

  ForIncBb =
      BasicBlock::Create(C, "sg_inter_pregion_for_inc", After->getParent());

  Builder.SetInsertPoint(ForIncBb);

  // The number of 'x lanes' subgroup spans.
  llvm::Value *NX = Builder.CreateLoad(ST, NXLanes);

  // Helper variables for non-linear loop.
  // The looping bounds for the y dim on the previous round.
  llvm::Value *PrevYLower = Builder.CreateLoad(ST, YLowerLimit);
  llvm::Value *PrevYUpper = Builder.CreateLoad(ST, YUpperLimit);

  // Update the new y dim bounds.
  llvm::Value *NextYLower = Builder.CreateAdd(PrevYLower, NX);
  NextYLower =
      Builder.CreateURem(NextYLower, llvm::ConstantInt::get(ST, WGLocalSizeY));
  Builder.CreateStore(NextYLower, YLowerLimit);

  llvm::Value *NextYUpper = Builder.CreateAdd(NextYLower, NX);
  Builder.CreateStore(NextYUpper, YUpperLimit);

  // Update the total WI counter.
  llvm::Value *SgInterCounterInc = Builder.CreateAdd(
      Builder.CreateLoad(ST, SgSize), Builder.CreateLoad(ST, SgInterCounter));
  Builder.CreateStore(SgInterCounterInc, SgInterCounter);

  // Local linear ID has been reset before entering the inc block, so increment
  // it for the next subgroups first work-item.
  llvm::Value *CurrLLID = Builder.CreateLoad(ST, LLID);
  llvm::Value *LLIDinc =
      Builder.CreateAdd(CurrLLID, Builder.CreateLoad(ST, SgSize));
  Builder.CreateStore(LLIDinc, LLID);

  After->getTerminator()->replaceUsesOfWith(OldExit, ForIncBb);
  Builder.CreateBr(OldExit);
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

namespace wiloops {
bool canHandleKernel(llvm::Function &K, llvm::PostDominatorTree &PDT,
                     llvm::LoopInfo &LI) {

  // The below cases should be now manageable. TODO: update the check for the
  // unhandled case(s).
  // Do not handle kernels with barriers inside loops which have early exits
  // or continues.
  // It would require additional complexity that is unlikely worth it since
  // the vectorizer won't produce efficient code for such loops anyhow.
  // Tested by tricky_for.cl.
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

    // Some loops with multiple back-edges form invalid parallel regions
    // currently.
    if (L->getLoopLatch()) {
      LLVM_DEBUG(
          dbgs()
          << "Multiple back-edges inside a barrier loop, won't handle.\n");
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

    LLVM_DEBUG(dbgs() << "### Detected a conditional barrier not currently supported by WILoops:\n");
    LLVM_DEBUG(BB->dump());

    return false;
  }
  return true;
}

} // namespace wiloops
} // namespace pocl
