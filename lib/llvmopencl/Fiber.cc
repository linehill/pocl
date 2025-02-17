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
#include "DebugHelpers.h"
#include "SubgroupBarrier.h"
#include "WorkitemHandlerChooser.h"
#include "llvm/IR/IRBuilder.h"
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Verifier.h>

#include <iostream>

#define INT_ZERO 0
#define N_DIM 3

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
  WorkitemHandlerType WIH;

  llvm::Module *M;
  llvm::Function *F;
  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  VariableUniformityAnalysisResult &VUA;


  std::array<llvm::GlobalVariable *, 3> LocalIdIterators;
  std::array<llvm::GlobalVariable *, 3> LocalSizeIterators;
  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;
  std::array<llvm::GlobalVariable *, 3> GroupIdIterators;
  std::array<llvm::Value *, 3> LocalSizeValues;
  llvm::ConstantInt *SGSize;

  std::vector<llvm::AllocaInst *> ContextAllocas;

  llvm::BasicBlock *DispatcherBlock;

  std::vector<llvm::BasicBlock *> BarrierBlocks;

  // Storage for blocks immediately after barriers.
  std::vector<llvm::BasicBlock *> BarrierExitBlocks;

  // Stack storage for block IDs of 'next block' for each WI.
  llvm::AllocaInst *NextJumpIndices;

  llvm::Type *Int64Type;

  // Pointer to workgroup data structure alloca.
  llvm::AllocaInst *WGStateAlloc;

  void handleWIContextVariables();

  llvm::AllocaInst *allocateStorage(llvm::IRBuilder<> &Builder,
                                    std::string VarName, llvm::Value *Nwi);

  llvm::Value *getNumberOfWIs(llvm::IRBuilder<> &Builder);

  bool processFunction(llvm::Function &F);

  void initializeLocalIds(llvm::BasicBlock *Entry, llvm::IRBuilder<> *Builder);

  void processBarriers();

  void handleBarrierReached(llvm::IRBuilder<> *Builder,
                            llvm::BasicBlock *BBlock);

  void initializeWGDataStruct(llvm::IRBuilder<> *Builder);

  void generateDispatcherBody(llvm::IRBuilder<> *Builder);
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

/// Creates a load for local id in desired dimension.
/// @param Instr Instruction used to initialize builder.
/// @param Dim Dimension of id.
/// @return Pointer to load instruction.
llvm::Instruction *FiberImpl::getLocalIdInRegion(llvm::Instruction *Instr,
                                                 size_t Dim) {
  llvm::IRBuilder<> Builder(Instr);
  return Builder.CreateLoad(ST, LocalIdGlobals[Dim]);
}

/// Sets the global variable iterators and generates LLVM IR for the
/// initialisation of local IDs.
/// Todo: Refactor to WorkitemHandler.
/// @param Entry the entry block in which local ids are initialised.
/// @param Builder the builder used to insert the instructions.
void FiberImpl::initializeLocalIds(BasicBlock *Entry, IRBuilder<> *Builder) {

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

    llvm::GlobalVariable *GvXyz = LocalIdIterators[I];
    if (GvXyz != NULL)
      Builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GvXyz);
  }
}

/// Generates LLVM IR for calculating the size of workgroup.
/// Also, stores the sizes of dimensions within the pass.
/// Todo: Refactor to WorkitemHandler.
/// @param Builder the builder for inserting the LLVM IR.
/// @return the total number of work-items in workgroup.
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

/// Adds store for the index of next block after barrier.
/// At the barrier, each WI stores its next block after the barrier, and then
/// loads it when continuing execution after barrier is resolved.
/// The block indices correspond to the switch case indices in the dispatcher
/// block.
/// @param Builder the builder used to insert the instructions.
/// @param BBlock the current block containing a barrier.
void FiberImpl::handleBarrierReached(llvm::IRBuilder<> *Builder,
                                     llvm::BasicBlock *BBlock) {

  llvm::Value *ZeroIndex =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()), INT_ZERO);

  // Fiber-scheduler functions for registering barriers.
  llvm::Function *WGBarrierReached =
      M->getFunction("__pocl_fiber_wg_barrier_reached");

  llvm::Function *SGBarrierReached =
      M->getFunction("__pocl_fiber_sg_barrier_reached");

  llvm::Value *LocalZ = Builder->CreateLoad(ST, LocalIdIterators[2], "LocalZ");
  llvm::Value *LocalY = Builder->CreateLoad(ST, LocalIdIterators[1], "LocalY");
  llvm::Value *LocalX = Builder->CreateLoad(ST, LocalIdIterators[0], "LocalX");

  llvm::Value *NextBlockPtr;

  // Get the appropriate index for storage.
  if (WGDynamicLocalSize) {
    llvm::Value *LinearID = getLinearWiIndex(*Builder, M, nullptr, WIH);
    NextBlockPtr =
        Builder->CreateGEP(NextJumpIndices->getAllocatedType(), NextJumpIndices,
                           {LinearID}, "exit_block_ptr");
  } else {
    NextBlockPtr = Builder->CreateGEP(
        NextJumpIndices->getAllocatedType(), NextJumpIndices,
        {ZeroIndex, LocalZ, LocalY, LocalX}, "exit_block_ptr");
  }

  llvm::Value *NextBlockIdx =
      llvm::ConstantInt::get(Int64Type, BarrierExitBlocks.size() - 1);

  Builder->CreateStore(NextBlockIdx, NextBlockPtr);

  // Register work-group/sub-group barrier entry
  if (SubgroupBarrier::hasSGBarrier(BBlock)) {
    Builder->CreateCall(SGBarrierReached,
                        {LocalX, LocalY, LocalZ, WGStateAlloc});
  } else {
    Builder->CreateCall(WGBarrierReached,
                        {LocalX, LocalY, LocalZ, WGStateAlloc});
  }
}

/// Modifies the barrier blocks, enabling dispatching.
/// For each barrier:
/// (1) Store the next block after the barrier.
/// (2) Create branch instruction to dispatcher block.
/// (3) Remove 'old' branch instruction from the current block.
void FiberImpl::processBarriers() {

  // There are 3 different cases that are handled separately.
  for (auto &BBlock : BarrierBlocks) {

    llvm::IRBuilder<> Builder(BBlock->getTerminator());

    // (1) This is the entry barrier.
    // Not part of the kernel logic so no need to notify scheduler.
    if (BBlock->getName() == "entry.barrier") {

      if (BBlock->getTerminator()->getNumSuccessors() > 0) {
        assert(isa<BranchInst>(BBlock->getTerminator()) &&
               "Expected a BranchInst!");

        BarrierExitBlocks.push_back(BBlock->getTerminator()->getSuccessor(0));
      }

      // This is the exit block with a barrier.
      // Create additional block to prevent early returns.
      // This way all wis pass through the 'old' exit block.
    } else if (BBlock->getTerminator()->getNumSuccessors() == 0) {

      // New exit block.
      llvm::BasicBlock *NewExitBlock =
          llvm::BasicBlock::Create(F->getContext(), "exit_block", F);

      llvm::IRBuilder<> NewExitBlockBuilder(NewExitBlock);
      NewExitBlockBuilder.CreateRetVoid();

      BarrierExitBlocks.push_back(NewExitBlock);

      handleBarrierReached(&Builder, BBlock);

      // These are "Explicit" barriers.
    } else {

      assert(isa<BranchInst>(BBlock->getTerminator()) &&
             "Expected a BranchInst!");

      // Store next block after barrier block.
      BarrierExitBlocks.push_back(BBlock->getTerminator()->getSuccessor(0));

      handleBarrierReached(&Builder, BBlock);
    }

    // Add new branch to dispatcher.
    Builder.CreateBr(DispatcherBlock);

    // Remove the old branch.
    BBlock->getTerminator()->eraseFromParent();
  }
}

/// Initializes the data structure used in communication with fiber-scheduler.
/// Contains workgroup-related information related to dimensions, subgroups
/// and barrier bookkeeping.
/// @param Builder Used in initialisation.
void FiberImpl::initializeWGDataStruct(llvm::IRBuilder<> *Builder) {

  // Type for struct that will store the work group execution state data.
  std::vector<llvm::Type *> WGStateData = {
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
      llvm::PointerType::get(Int64Type, INT_ZERO),
      llvm::PointerType::get(Int64Type, INT_ZERO),
  };

  llvm::StructType *WGState =
      llvm::StructType::get(M->getContext(), WGStateData, "wgState");

  WGStateAlloc = Builder->CreateAlloca(WGState, nullptr, "wg_state_data");

  llvm::Instruction *LoadSize;

  // Store work-group size values to struct.
  // Use reverse order so we are left with x-size which is needed later.
  for (int Idx = N_DIM - 1; Idx >= INT_ZERO; Idx--) {
    llvm::Value *StateLocalSize = Builder->CreateGEP(
        WGState, WGStateAlloc,
        {llvm::ConstantInt::get(Int64Type, INT_ZERO),
         llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), Idx)});

    LoadSize = Builder->CreateLoad(ST, LocalSizeGlobals[Idx]);
    Builder->CreateStore(LoadSize, StateLocalSize);
  }

  // Pointer to sub-group size member in the struct.
  llvm::Value *SubGroupSize = Builder->CreateGEP(
      WGState, WGStateAlloc,
      {llvm::ConstantInt::get(Int64Type, 0),
       llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 3)});

  // Store the sub-group size to struct.
  // If specified with intel_reqd_sub_group_size:
  if (llvm::MDNode *SGSizeMD = F->getMetadata("intel_reqd_sub_group_size")) {

    llvm::ConstantAsMetadata *ConstMD =
        llvm::cast<llvm::ConstantAsMetadata>(SGSizeMD->getOperand(0));

    uint64_t As64Type =
        (llvm::cast<llvm::ConstantInt>(ConstMD->getValue()))->getZExtValue();
    llvm::ConstantInt *SGSize64 = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(F->getContext()), As64Type);
    SGSize = llvm::cast<llvm::ConstantInt>(SGSize64);
    Builder->CreateStore(SGSize64, SubGroupSize);

  } else {
    // With dynamic work-group sizes, use the run-time value.
    if (WGDynamicLocalSize) {
      Builder->CreateStore(LoadSize, SubGroupSize);
      // Otherwise use compile-time value.
    } else {
      SGSize = llvm::cast<llvm::ConstantInt>(LocalSizeValues[0]);
      Builder->CreateStore(SGSize, SubGroupSize);
    }
  }
}

/// Allocates and initializes storage for managing block 'indices'.
///
/// At the end of the dispatcher block, a switch statement determines which
/// block is jumped to next. When a new workitem is scheduled, it retrieves its
/// own block index from this storage, which is used to select the correct
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

/// Generates body of the dispatcher block.
/// - Call to fiber-scheduler to decide next WI.
/// - Unlinearisation of the WI id.
/// - Update local and global ids.
/// - Population of switch-statement for after-barrier blocks.
/// @param EntryBlockBuilder the builder used to set up the block body.
void FiberImpl::generateDispatcherBody(llvm::IRBuilder<> *EntryBlockBuilder) {

  llvm::IRBuilder<> DBuilder(DispatcherBlock);

  // Function call to __pocl_sched_work_item to retrieve next WI id.
  llvm::Function *SchedFunc = M->getFunction("__pocl_fiber_schedule_work_item");

  // Retrieve the return value, i.e. WI id.
  llvm::Value *LinearWI = DBuilder.CreateCall(SchedFunc, {WGStateAlloc});
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
  DBuilder.CreateStore(GidZ, GlobalIdIterators[2]);

  llvm::Value *ZeroIndex =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()), INT_ZERO);

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
  if (BarrierExitBlocks.size() > 0) {

    // Default for entry.
    llvm::SwitchInst *SwitchInst =
        DBuilder.CreateSwitch(LoadedValue, BarrierExitBlocks[0]);

    // Add case for each barrier exit.
    for (int I = 1; I < BarrierExitBlocks.size(); I++) {
      llvm::ConstantInt *CaseValue =
          llvm::ConstantInt::get(DBuilder.getInt64Ty(), I);
      SwitchInst->addCase(CaseValue, BarrierExitBlocks[I]);
    }
  }
}

bool FiberImpl::processFunction(llvm::Function &F) {

  Int64Type = llvm::Type::getInt64Ty(M->getContext());

  llvm::BasicBlock *EntryBlock = nullptr;

  // Ger pointer to the entry block and collect the blocks that have barriers.
  for (auto &Block : F) {
    if (Block.getName() == "entry.barrier") {
      EntryBlock = &Block;
    }
    if (Barrier::hasBarrier(&Block)) {
      BarrierBlocks.push_back(&Block);
    }
  }

  llvm::IRBuilder<> EntryBlockBuilder(&*(EntryBlock->getFirstInsertionPt()));

  initializeLocalIds(EntryBlock, &EntryBlockBuilder);

  handleWIContextVariables();

  handleWorkitemFunctions();

  llvm::Instruction *WGSize = getWorkGroupSizeInstr();

  llvm::Value *Nwi = getNumberOfWIs(EntryBlockBuilder);

  NextJumpIndices = allocateStorage(EntryBlockBuilder, "jump_indices", WGSize);

  initializeWGDataStruct(&EntryBlockBuilder);

  // Allocate counters, used by scheduler, for each subgroup.
  // Will allocate 'number of work items', which is the worst case situation.
  llvm::AllocaInst *SGWiCounter =
      allocateStorage(EntryBlockBuilder, "_sg_wi_counter", WGSize);
  llvm::AllocaInst *SGBarrierCounter =
      allocateStorage(EntryBlockBuilder, "_sg_barrier_counter", WGSize);

  llvm::Function *SchedulerInit = M->getFunction("__pocl_fiber_sched_init");

  // Will initialise the data structure on fiber-scheduler side.
  EntryBlockBuilder.CreateCall(SchedulerInit,
                               {WGStateAlloc, SGWiCounter, SGBarrierCounter});

  // This will be the block from which WI continues its execution after
  // barrier release.
  DispatcherBlock = llvm::BasicBlock::Create(F.getContext(), "dispatcher", &F);

  processBarriers();

  generateDispatcherBody(&EntryBlockBuilder);

  handleLocalMemAllocas();

  removeBarrierCalls();

  return true;
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

  bool Changed = processFunction(Func);

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

  return Changed;
}

bool addFiberExecution(llvm::Function &F, llvm::DominatorTree &DT,
                       llvm::PostDominatorTree &PDT, llvm::LoopInfo &LI,
                       VariableUniformityAnalysisResult &VUA) {

  FiberImpl Fiber(DT, VUA, LI);
  return Fiber.runOnFunction(F);
}

} // namespace pocl
