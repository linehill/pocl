// Fiber workgroup method implementation.
//
// Copyright (c) 2025 Tapio Nevalainen / Tampere University
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

#include "Fiber.h"
#include "Barrier.h"
#include "CanonicalizeBarriers.h"
#include "DebugHelpers.h"
#include "ImplicitConditionalBarriers.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "SubgroupBarrier.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "WorkitemHandlerChooser.h"
#include "pocl_llvm_api.h"
#include "llvm/IR/IRBuilder.h"
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Verifier.h>

#include <iostream>

namespace pocl {

using namespace llvm;

class FiberImpl : public pocl::WorkitemHandler {

public:
  FiberImpl(llvm::DominatorTree &DT, VariableUniformityAnalysisResult &VUA, llvm::LoopInfo &LI)
      : WorkitemHandler(), DT(DT), VUA(VUA), LI(LI) {}

  virtual bool runOnFunction(llvm::Function &F);

protected:
  llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr,
                                        size_t Dim) override;

private:
  using InstructionIndex = std::set<llvm::Instruction *>;
  using InstructionVec = std::vector<llvm::Instruction *>;
  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;

  WorkitemHandlerType WIH;

  llvm::Module *M;
  llvm::Function *F;
  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  VariableUniformityAnalysisResult &VUA;

  StrInstructionMap ContextArrays;

  std::array<llvm::GlobalVariable *, 3> LocalIdIterators;
  std::array<llvm::GlobalVariable *, 3> LocalSizeIterators;
  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;
  std::array<llvm::GlobalVariable *, 3> GroupIdIterators;
  std::array<llvm::Value *, 3> LocalSizeValues;
  llvm::ConstantInt *SgSize;

  size_t TempInstructionIndex;

  std::map<llvm::Instruction *, unsigned> TempInstructionIds;

  std::vector<llvm::AllocaInst *> ContextAllocas;

  void handleWIContextVariables();

  llvm::AllocaInst *allocateStorage(llvm::IRBuilder<> &Builder,
                                    std::string VarName, llvm::Value *Nwi);

  llvm::Value *getNumberOfWIs(llvm::IRBuilder<> &Builder);

  /* void addContextSaveRestore(llvm::Instruction *Instruction); */

  void initializeLocalIds(llvm::BasicBlock *Entry, llvm::IRBuilder<> *Builder);

  void initializeGlobalIterators();
};

/// Handles context save/restore of workitem variables.
///
/// Uses WorkitemHandlers functionality to identify variables that should be
/// context saved. In addition, checks that variable candidate has users,
/// and that the user is not in the same block.
void FiberImpl::handleWIContextVariables() {

  InstructionVec ValuesToContextSave;

  WorkitemHandlerType WIH = getWorkitemHandler();

  // Identify variables to save.
  for (auto &BB : *F) {
    for (auto &Instr : BB) {

      if (shouldNotBeContextSaved(&Instr, VUA, WIH))
        continue;

      for (llvm::Instruction::use_iterator UI = Instr.use_begin(),
                                           UE = Instr.use_end();
           UI != UE; ++UI) {

        llvm::Instruction *User =
            llvm::dyn_cast<llvm::Instruction>(UI->getUser());

        if (User == NULL)
          continue;

        llvm::BasicBlock *CurrentBlock = Instr.getParent();

        llvm::BasicBlock *UserBlock = User->getParent();

        // Context save should not be applied if User is in same block.
        if (CurrentBlock == UserBlock) {
          continue;
        }
        ValuesToContextSave.push_back(&Instr);
        break;
      }
    }
  }

  for (auto &Instr : ValuesToContextSave) {
    addContextSaveRestore(Instr, LI);
  }
}

// Override of WIHandler function, not needed in this pass.
llvm::Instruction *FiberImpl::getLocalIdInRegion(llvm::Instruction *Instr,
                                                 size_t Dim) {
  llvm::IRBuilder<> Builder(Instr);
  return Builder.CreateLoad(ST, LocalIdGlobals[Dim]);
}

// Initialize local ids as zero
void FiberImpl::initializeLocalIds(BasicBlock *Entry, IRBuilder<> *Builder) {

  llvm::GlobalVariable *GVX = LocalIdIterators[0];
  if (GVX != NULL)
    Builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GVX);

  llvm::GlobalVariable *GVY = LocalIdIterators[1];
  if (GVY != NULL)
    Builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GVY);

  llvm::GlobalVariable *GVZ = LocalIdIterators[2];
  if (GVZ != NULL)
    Builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GVZ);
}

// Cast values stored in WorkitemHandler to global variable pointers.
void FiberImpl::initializeGlobalIterators() {

  for (int I = 0; I < 3; I++) {
    // _local_id_xyz
    LocalIdIterators[I] = llvm::cast<llvm::GlobalVariable>(LocalIdGlobals[I]);

    // _local_size_xyz
    LocalSizeIterators[I] =
        llvm::cast<llvm::GlobalVariable>(LocalSizeGlobals[I]);

    // _global_id_xyz
    GlobalIdIterators[I] = llvm::cast<llvm::GlobalVariable>(GlobalIdGlobals[I]);

    // _group_id_xyz
    GroupIdIterators[I] = llvm::cast<llvm::GlobalVariable>(GroupIdGlobals[I]);
  }
}

// Calculate and return the number of work items in workgroup.
llvm::Value *FiberImpl::getNumberOfWIs(llvm::IRBuilder<> &Builder) {

  llvm::Value *Nwi;

  if (WGDynamicLocalSize) {

    llvm::Instruction *LoadX = Builder.CreateLoad(ST, LocalSizeGlobals[0]);
    llvm::Instruction *LoadY = Builder.CreateLoad(ST, LocalSizeGlobals[1]);
    llvm::Instruction *LoadZ = Builder.CreateLoad(ST, LocalSizeGlobals[2]);
    // Store localsizes values for later use.
    LocalSizeValues[0] = LoadX;
    LocalSizeValues[1] = LoadY;
    LocalSizeValues[2] = LoadZ;

    llvm::Value *Xy = Builder.CreateBinOp(llvm::Instruction::Mul, LoadX, LoadY);
    llvm::Value *Xyz = Builder.CreateBinOp(llvm::Instruction::Mul, Xy, LoadZ);

    Xyz->setName("Nwi");
    Nwi = Xyz;

  } else {
    LocalSizeValues[0] = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeX, false);
    LocalSizeValues[1] = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeY, false);
    LocalSizeValues[2] = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeZ, false);

    Nwi =
        llvm::ConstantInt::get(ST, WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ);
  }
  return Nwi;
}

/// Allocates and initializes storage for managing block 'indices'.
///
/// At the end of the dispatcher block, switch statement determines which
/// block is jumped next. When new workitem is scheduled, it retrieves its
/// own block index from this storage, which is used the select correct
/// switch case. Also, before scheduling, workitem stores its next block
/// index, so it can continue when it is scheduled again.
///
/// \param builder The LLVM IR builder used to insert alloca.
/// \param varName The name assigned to the variable in LLVM IR.
/// \param Nwi A pointer to an LLVM `Value` representing the number of work
///        items for which storage needs to be allocated.
/// \return A pointer to `llvm::AllocaInst` representing the allocated storage.
llvm::AllocaInst *FiberImpl::allocateStorage(llvm::IRBuilder<> &Builder,
                                             std::string VarName,
                                             llvm::Value *Nwi) {
  // Dummy LoadInst. Something like this is required to use WIHandler's
  // allocation functionality.
  llvm::Value *DummyValue =
      Builder.CreateLoad(ST, LocalIdIterators[0], "dummy");
  llvm::LoadInst *DummyInst = llvm::dyn_cast<llvm::LoadInst>(DummyValue);
  bool PaddingAdded = false;

  // Use WIHandler's allocation machinery to create a proper alloca.
  // This will handle dynamic/non-dynamic cases.
  llvm::AllocaInst *BlockIDArray = createAlignedAndPaddedContextAlloca(
      DummyInst, DummyInst, VarName, PaddingAdded);

  // This is not needed anymore.
  DummyInst->eraseFromParent();

  llvm::Value *Zero = Builder.getInt8(0);
  llvm::MaybeAlign MaybeAlign(BlockIDArray->getAlign().value());
  llvm::Type *Int64Type = llvm::Type::getInt64Ty(M->getContext());
  uint64_t ElementSizeBytes = M->getDataLayout().getTypeAllocSize(Int64Type);

  // Initialize allocated memory to zero.
  if (WGDynamicLocalSize) {
    llvm::ConstantInt *TypeSizeVal = llvm::ConstantInt::get(
        M->getContext(), llvm::APInt(64, ElementSizeBytes));
    llvm::Value *TotalSize = Builder.CreateMul(Nwi, TypeSizeVal);
    Builder.CreateMemSet(BlockIDArray, Zero, TotalSize, MaybeAlign);
  } else {
    unsigned long TotalSize =
        WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ * ElementSizeBytes;
    Builder.CreateMemSet(BlockIDArray, Zero, TotalSize, MaybeAlign);
  }
  return BlockIDArray;
}

bool FiberImpl::runOnFunction(llvm::Function &Func) {

  M = Func.getParent();
  F = &Func;

  WIH = getWorkitemHandler();

  Initialize(llvm::cast<Kernel>(&Func));

#ifdef DEBUG_FIBER
  std::cerr << "Before fiber:\n";
  F->dump();

  llvm::verifyFunction(Func);
#endif

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(*F, F->getName().str() + "_before_fiber.dot", nullptr, nullptr);
#endif

  // Initialize pointers to global variables:
  initializeGlobalIterators();

  // Expand workitem function calls.
  handleWorkitemFunctions();

  TempInstructionIndex = 0;

  // Context save/restore
  handleWIContextVariables();

  llvm::BasicBlock *EntryBlock = nullptr;

  for (auto &BB : Func) {
    if (BB.getName() == "entry.barrier") {
      EntryBlock = &BB;
      break;
    }
  }

  llvm::IRBuilder<> EntryBlockBuilder(&*(EntryBlock->getFirstInsertionPt()));

  // Initialize local ids to 0
  initializeLocalIds(EntryBlock, &EntryBlockBuilder);

  llvm::Type *Int64Type = llvm::Type::getInt64Ty(M->getContext());

  llvm::Value *Nwi = getNumberOfWIs(EntryBlockBuilder);

  // Stack storage for block IDs of 'next block' for each WI.
  llvm::AllocaInst *NextJumpIndices =
      allocateStorage(EntryBlockBuilder, "jump_indices", Nwi);

  // Allocate counters, used by scheduler, for each subgroup.
  // Will allocate 'number of work items', which is the worst case situation.
  llvm::AllocaInst *SgWiCounter =
      allocateStorage(EntryBlockBuilder, "_sg_wi_counter", Nwi);
  llvm::AllocaInst *SgBarrierCounter =
      allocateStorage(EntryBlockBuilder, "_sg_wi_counter", Nwi);

  // Type for struct that will store the work group data.
  std::vector<llvm::Type *> WgStateData = {
      // x_size of workgroup
      Int64Type,
      // y-size of workgroup
      Int64Type,
      // z-size of workgroup
      Int64Type,
      // sg-size
      Int64Type,
      // n subgroups
      Int64Type,
      // waiting count
      Int64Type,
      // sg barriers active
      Int64Type,
      // Counters
      llvm::PointerType::get(Int64Type, 0),
      llvm::PointerType::get(Int64Type, 0),
  };

  llvm::Instruction *LastInst;

  llvm::StructType *WgState =
      llvm::StructType::get(M->getContext(), WgStateData, "wgState");

  llvm::AllocaInst *WgStateAlloc =
      EntryBlockBuilder.CreateAlloca(WgState, nullptr, "wg_state_data");

  // Get pointers to struct work-group size members.
  llvm::Value *StateLocalSizeX = EntryBlockBuilder.CreateGEP(
      WgState, WgStateAlloc,
      {llvm::ConstantInt::get(Int64Type, 0),
       llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 0)});

  llvm::Value *StateLocalSizeY = EntryBlockBuilder.CreateGEP(
      WgState, WgStateAlloc,
      {llvm::ConstantInt::get(Int64Type, 0),
       llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 1)});

  llvm::Value *StateLocalSizeZ = EntryBlockBuilder.CreateGEP(
      WgState, WgStateAlloc,
      {llvm::ConstantInt::get(Int64Type, 0),
       llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 2)});

  // Store work-group size values to struct.
  llvm::Instruction *LoadXSize =
      EntryBlockBuilder.CreateLoad(ST, LocalSizeGlobals[0]);
  EntryBlockBuilder.CreateStore(LoadXSize, StateLocalSizeX);

  llvm::Instruction *LoadYSize =
      EntryBlockBuilder.CreateLoad(ST, LocalSizeGlobals[1]);
  EntryBlockBuilder.CreateStore(LoadYSize, StateLocalSizeY);

  llvm::Instruction *LoadZSize =
      EntryBlockBuilder.CreateLoad(ST, LocalSizeGlobals[2]);
  EntryBlockBuilder.CreateStore(LoadZSize, StateLocalSizeZ);

  // Pointer to sub-group size member in the struct.
  llvm::Value *SubgSize = EntryBlockBuilder.CreateGEP(
      WgState, WgStateAlloc,
      {llvm::ConstantInt::get(Int64Type, 0),
       llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 3)});

  // Store the sub-group size to struct.
  // If specified with intel_reqd_sub_group_size:
  if (llvm::MDNode *SGSizeMD = F->getMetadata("intel_reqd_sub_group_size")) {

    llvm::ConstantAsMetadata *ConstMD =
        llvm::cast<llvm::ConstantAsMetadata>(SGSizeMD->getOperand(0));

    uint64_t As64Type =
        (llvm::cast<llvm::ConstantInt>(ConstMD->getValue()))->getZExtValue();
    llvm::ConstantInt *SgSize64 = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(F->getContext()), As64Type);
    SgSize = llvm::cast<llvm::ConstantInt>(SgSize64);
    EntryBlockBuilder.CreateStore(SgSize64, SubgSize);

  } else {
    // With dynamic work-group sizes, use the run-time value.
    if (WGDynamicLocalSize) {
      EntryBlockBuilder.CreateStore(LoadXSize, SubgSize);
      // Otherwise use compile-time value.
    } else {
      SgSize = llvm::cast<llvm::ConstantInt>(LocalSizeValues[0]);
      EntryBlockBuilder.CreateStore(SgSize, SubgSize);
    }
  }

  // Scheduler functions.
  llvm::Function *SchedulerInit = M->getFunction("__pocl_fiber_sched_init");

  llvm::FunctionType *FTy = SchedulerInit->getFunctionType();

  llvm::Function *WgbarrierReached =
      M->getFunction("__pocl_fiber_wg_barrier_reached");
  llvm::Function *SgbarrierReached =
      M->getFunction("__pocl_fiber_sg_barrier_reached");

  // This will initialise the data structure on scheduler side.
  LastInst = EntryBlockBuilder.CreateCall(
      SchedulerInit, {WgStateAlloc, SgWiCounter, SgBarrierCounter});

  llvm::BasicBlock *CurrBlock = EntryBlock;

  // Storage for branch instructions after barriers.
  std::vector<llvm::BranchInst *> BarrBrInstrs;

  // Store barriers here, for easier handling.
  std::vector<llvm::BasicBlock *> BarrierBlocks;

  for (auto &Block : Func) {

    if (Barrier::hasBarrier(&Block)) {
      BarrierBlocks.push_back(&Block);
    }
  }

  llvm::Value *ZeroIndex =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()), 0);

  // Store pointer to old exit here
  llvm::BasicBlock *OldExitBlock = nullptr;

  // Create new block for dispatcher; dispathcer block manipulation is
  // done later. Need this for reference for now.
  llvm::BasicBlock *DispatcherBlock =
      llvm::BasicBlock::Create(F->getContext(), "dispatcher", F);

  // Modify the barrier blocks:
  // Three cases that have to be handled somewhat differently.
  // Whenever we have a barrier:
  // (1) Store a copy of the branch instruction in the end of the block.
  // (2) Create branch instruction to dispatcher block.
  // (3) Remove 'old' branch instruction from the current block.
  for (auto &BBlock : BarrierBlocks) {

    // With entry barrier it is not necessary to do the ..
    //
    if (BBlock->getName() == "entry.barrier") {

      if (BBlock->getTerminator()->getNumSuccessors() > 0) {
        assert(isa<BranchInst>(BBlock->getTerminator()) &&
               "Expected a BranchInst!");
        BarrBrInstrs.push_back(
            dyn_cast<BranchInst>(BBlock->getTerminator()->clone()));
      }

      llvm::IRBuilder<> EntryBuilder(BBlock->getTerminator());
      EntryBuilder.CreateBr(DispatcherBlock);

      BBlock->getTerminator()->eraseFromParent();

      // This is the exit block with a barrier.
      // Create additional block to prevent early returns.
      // This way all wis pass through the 'old' exit block.
    } else if (BBlock->getTerminator()->getNumSuccessors() == 0) {

      // New exit block.
      llvm::BasicBlock *NewExitBlock =
          llvm::BasicBlock::Create(F->getContext(), "exit_block", F);

      llvm::IRBuilder<> NewExitBlockBuilder(NewExitBlock);
      NewExitBlockBuilder.CreateRetVoid();

      // Branch to new exit.
      llvm::BranchInst *BranchInst = llvm::BranchInst::Create(NewExitBlock);
      BarrBrInstrs.push_back(BranchInst);

      // Handle for old return block
      llvm::IRBuilder<> OldExitBuilder(BBlock->getTerminator());

      llvm::Value *LocalZ =
          OldExitBuilder.CreateLoad(ST, LocalIdIterators[2], "LocalZ");
      llvm::Value *LocalY =
          OldExitBuilder.CreateLoad(ST, LocalIdIterators[1], "LocalY");
      llvm::Value *LocalX =
          OldExitBuilder.CreateLoad(ST, LocalIdIterators[0], "LocalX");

      llvm::Value *NextBlockPtr;

      // Get the
      if (WGDynamicLocalSize) {
        llvm::Value *LinearID =
            getLinearWiIndex(OldExitBuilder, M, nullptr, WIH);
        NextBlockPtr = OldExitBuilder.CreateGEP(
            NextJumpIndices->getAllocatedType(), NextJumpIndices, {LinearID},
            "exit_block_ptr");
      } else {
        NextBlockPtr = OldExitBuilder.CreateGEP(
            NextJumpIndices->getAllocatedType(), NextJumpIndices,
            {ZeroIndex, LocalZ, LocalY, LocalX}, "exit_block_ptr");
      }

      // Store the next block index for current WI.
      llvm::Value *NextBlockIdx =
          llvm::ConstantInt::get(Int64Type, BarrBrInstrs.size() - 1);
      OldExitBuilder.CreateStore(NextBlockIdx, NextBlockPtr);

      // Barrier in exit block is always work-group barrier.
      OldExitBuilder.CreateCall(WgbarrierReached,
                                {LocalX, LocalY, LocalZ, WgStateAlloc});

      OldExitBuilder.CreateBr(DispatcherBlock);

      // Remove previous 'ret void' instruction.
      BBlock->getTerminator()->eraseFromParent();

      // These are "Explicit" barriers.
    } else {

      assert(isa<BranchInst>(BBlock->getTerminator()) &&
             "Expected a BranchInst!");
      BarrBrInstrs.push_back(
          dyn_cast<BranchInst>(BBlock->getTerminator()->clone()));

      llvm::IRBuilder<> BarrierBlockBuilder(BBlock->getTerminator());

      llvm::Value *LocalZ =
          BarrierBlockBuilder.CreateLoad(ST, LocalIdIterators[2], "LocalZ");
      llvm::Value *LocalY =
          BarrierBlockBuilder.CreateLoad(ST, LocalIdIterators[1], "LocalY");
      llvm::Value *LocalX =
          BarrierBlockBuilder.CreateLoad(ST, LocalIdIterators[0], "LocalX");

      llvm::Value *NextBlockPtr;

      if (WGDynamicLocalSize) {
        llvm::Value *LinearID =
            getLinearWiIndex(BarrierBlockBuilder, M, nullptr, WIH);
        NextBlockPtr = BarrierBlockBuilder.CreateGEP(
            NextJumpIndices->getAllocatedType(), NextJumpIndices, {LinearID},
            "exit_block_ptr");
      } else {
        NextBlockPtr = BarrierBlockBuilder.CreateGEP(
            NextJumpIndices->getAllocatedType(), NextJumpIndices,
            {ZeroIndex, LocalZ, LocalY, LocalX}, "exit_block_ptr");
      }

      llvm::Value *NextBlockIdx =
          llvm::ConstantInt::get(Int64Type, BarrBrInstrs.size() - 1);
      BarrierBlockBuilder.CreateStore(NextBlockIdx, NextBlockPtr);

      // Register work-group/sub-group barrier entry
      if (SubgroupBarrier::hasSGBarrier(BBlock)) {
        BarrierBlockBuilder.CreateCall(SgbarrierReached,
                                       {LocalX, LocalY, LocalZ, WgStateAlloc});
      } else {
        BarrierBlockBuilder.CreateCall(WgbarrierReached,
                                       {LocalX, LocalY, LocalZ, WgStateAlloc});
      }

      // Add branch to dispatcher
      BarrierBlockBuilder.CreateBr(DispatcherBlock);

      // Remove the old branch
      BBlock->getTerminator()->eraseFromParent();
    }
  }

  // Dispatcher implementation
  llvm::IRBuilder<> DBuilder(DispatcherBlock);

  // Function call to __pocl_sched_work_item to retrieve next WI id.
  llvm::Function *SchedFunc = M->getFunction("__pocl_fiber_schedule_work_item");

  // Retrieve the return value, i.e. WI id.
  llvm::Value *LinearWI = DBuilder.CreateCall(SchedFunc, {WgStateAlloc});
  LinearWI->setName("next_linear_wi");

  // 'Unlinearize' the WI id.
  // X
  llvm::Value *LocX =
      DBuilder.CreateBinOp(llvm::Instruction::BinaryOps::SRem, LinearWI,
                           LocalSizeValues[0], "loc_id_x");

  llvm::Value *Xtimesy =
      DBuilder.CreateBinOp(llvm::Instruction::BinaryOps::Mul,
                           LocalSizeValues[0], LocalSizeValues[1]);

  // Y
  llvm::Value *LocYtmp = DBuilder.CreateBinOp(
      llvm::Instruction::BinaryOps::SRem, LinearWI, Xtimesy, "loc_id_y_tmp");
  llvm::Value *LocY = DBuilder.CreateBinOp(llvm::Instruction::UDiv, LocYtmp,
                                           LocalSizeValues[0], "loc_id_y");

  // Z
  llvm::Value *LocZ = DBuilder.CreateBinOp(llvm::Instruction::UDiv, LinearWI,
                                           Xtimesy, "loc_id_z");

  // Store new local ids.
  DBuilder.CreateStore(LocX, LocalIdIterators[0]);
  DBuilder.CreateStore(LocY, LocalIdIterators[1]);
  DBuilder.CreateStore(LocZ, LocalIdIterators[2]);

  // Calculate global ids.
  llvm::Value *Xgid = DBuilder.CreateLoad(ST, GroupIdGlobals[0], "group_id_x");
  llvm::Value *Ygid = DBuilder.CreateLoad(ST, GroupIdGlobals[1], "group_id_y");
  llvm::Value *Zgid = DBuilder.CreateLoad(ST, GroupIdGlobals[2], "group_id_z");

  llvm::Value *MultX = DBuilder.CreateMul(LocalSizeValues[0], Xgid, "mulx");
  llvm::Value *MultY = DBuilder.CreateMul(LocalSizeValues[1], Ygid, "muly");
  llvm::Value *MultZ = DBuilder.CreateMul(LocalSizeValues[2], Zgid, "mulz");

  llvm::Value *MulXLoc = DBuilder.CreateAdd(MultX, LocX, "mul_x_loc");
  llvm::Value *MulYLoc = DBuilder.CreateAdd(MultY, LocY, "mul_y_loc");
  llvm::Value *MulZLoc = DBuilder.CreateAdd(MultZ, LocZ, "mul_z_loc");

  llvm::GlobalVariable *OffsetXPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_global_offset_x", ST));
  llvm::GlobalVariable *OffsetYPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_global_offset_y", ST));
  llvm::GlobalVariable *OffsetZPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_global_offset_z", ST));

  llvm::Value *OffsetX = DBuilder.CreateLoad(ST, OffsetXPtr, "offset_x");
  llvm::Value *OffsetY = DBuilder.CreateLoad(ST, OffsetYPtr, "offset_y");
  llvm::Value *OffsetZ = DBuilder.CreateLoad(ST, OffsetZPtr, "offset_z");

  llvm::Value *GidX = DBuilder.CreateAdd(MulXLoc, OffsetX, "gid_x");
  llvm::Value *GidY = DBuilder.CreateAdd(MulYLoc, OffsetY, "gid_y");
  llvm::Value *GidZ = DBuilder.CreateAdd(MulZLoc, OffsetZ, "gid_z");

  // Store global ids.
  DBuilder.CreateStore(GidX, GlobalIdIterators[0]);
  DBuilder.CreateStore(GidY, GlobalIdIterators[1]);
  LastInst = DBuilder.CreateStore(GidZ, GlobalIdIterators[2]);

  // Pointer to next block for current WI.
  llvm::Value *NextBlockPtr;
  if (WGDynamicLocalSize) {
    llvm::Value *LinearID = getLinearWiIndex(DBuilder, M, nullptr, WIH);
    NextBlockPtr =
        DBuilder.CreateGEP(NextJumpIndices->getAllocatedType(), NextJumpIndices,
                           {LinearID}, "exit_block_ptr");
  } else {
    NextBlockPtr =
        DBuilder.CreateGEP(NextJumpIndices->getAllocatedType(), NextJumpIndices,
                           {ZeroIndex, LocZ, LocY, LocX}, "exit_block_ptr");
  }

  // Retrieve next block index.
  llvm::Value *LoadedValue =
      DBuilder.CreateLoad(DBuilder.getInt64Ty(), NextBlockPtr, "next_block");

  // Add the switch statement and handle jumping to 'after-barrier' blocks.
  if (BarrBrInstrs.size() > 0) {

    llvm::SwitchInst *SwitchInst;

    // For each after-barrier block, create a 'helper block' in which there
    // will be:
    // (1) load from condition variable, IF branch is conditional.
    // (2) branch (either conditional or unconditional) to after-barrier block.
    for (int I = 0; I < BarrBrInstrs.size(); I++) {

      std::string CaseBlockName = "case" + std::to_string(I);
      llvm::BasicBlock *CaseBlock =
          llvm::BasicBlock::Create(F->getContext(), CaseBlockName, F);

      llvm::IRBuilder<> CaseBlockBuilder(CaseBlock);

      // In case of conditional branch, allocate storage in entry block
      // and add manual context save/restore.
      // TODO: has to be array instead of single boolean (one for each WI).
      if (BarrBrInstrs[I]->isConditional()) {

        llvm::Type *CondType = BarrBrInstrs[I]->getCondition()->getType();
        llvm::AllocaInst *Alloc =
            EntryBlockBuilder.CreateAlloca(CondType, nullptr, "disp_br_cond");

        // Context save the condition variable where calculated.
        llvm::Value *Condition = BarrBrInstrs[I]->getCondition();

        llvm::Instruction *DefiningInst =
            llvm::dyn_cast<llvm::Instruction>(Condition);

        llvm::IRBuilder<> Builder(DefiningInst->getNextNode());
        llvm::Instruction *Saved = Builder.CreateStore(DefiningInst, Alloc);

        llvm::Instruction *Restore =
            CaseBlockBuilder.CreateLoad(CondType, Alloc, "cond_restore");

        BarrBrInstrs[I]->setCondition(Restore);
      }

      CaseBlockBuilder.Insert(BarrBrInstrs[I]);

      llvm::ConstantInt *CaseValue =
          llvm::ConstantInt::get(DBuilder.getInt64Ty(), I);

      if (I == 0)
        SwitchInst = DBuilder.CreateSwitch(LoadedValue, CaseBlock);
      else
        SwitchInst->addCase(CaseValue, CaseBlock);
    }
  }

  handleLocalMemAllocas();

#ifdef DEBUG_FIBER
  std::cerr << "After fiber:\n";
  F->dump();
  std::string Log;
  llvm::raw_string_ostream OS(Log);
  bool BrokenDebugInfo = false;

  llvm::verifyModule(*M, &OS, &BrokenDebugInfo);
  if (!Log.empty()) {
    std::cerr << "Module verification errors:\n" << Log << std::endl;
  }
  llvm::verifyFunction(Func);
#endif

#ifdef POCL_KERNEL_COMPILER_DUMP_CFGS
  dumpCFG(*F, F->getName().str() + "_after_fiber.dot", nullptr, nullptr);
#endif

  removeBarrierCalls();

  return true;
}

bool addFiberExecution(llvm::Function &F, llvm::DominatorTree &DT,
                       llvm::PostDominatorTree &PDT, llvm::LoopInfo &LI,
                       VariableUniformityAnalysisResult &VUA) {

  FiberImpl Fiber(DT, VUA, LI);
  return Fiber.runOnFunction(F);
}

} // namespace pocl
