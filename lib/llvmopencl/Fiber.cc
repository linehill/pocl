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

#include "LLVMUtils.h"
#include "ImplicitConditionalBarriers.h"
#include "Fiber.h"
#include "CanonicalizeBarriers.h"
#include "WorkitemHandlerChooser.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "llvm/IR/IRBuilder.h"
#include "DebugHelpers.h"
#include "KernelCompilerUtils.h"
#include <llvm/IR/Verifier.h>
#include <llvm/IR/DataLayout.h>
#include "Barrier.h"
#include "SubgroupBarrier.h"
#include "pocl_llvm_api.h"
#include <llvm/Analysis/PostDominators.h>

#include <iostream>


#define PASS_NAME "fiber"
#define PASS_CLASS pocl::Fiber
#define PASS_DESC "Simple and robust work group function generator"

// #define DEBUG_FIBER

namespace pocl{

using namespace llvm;

class FiberImpl : public pocl::WorkitemHandler{

public:
    FiberImpl(llvm::DominatorTree &DT,
              VariableUniformityAnalysisResult &VUA)
              : WorkitemHandler(), DT(DT), VUA(VUA) {}

    virtual bool runOnFunction(llvm::Function &F);

protected:
  llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr) override;
  llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr,
                                        size_t Dim) override;

private:
  using InstructionIndex = std::set<llvm::Instruction *>;
  using InstructionVec = std::vector<llvm::Instruction *>;
  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;
  
  llvm::Module *M;
  llvm::Function *F;
  llvm::DominatorTree &DT;
  VariableUniformityAnalysisResult &VUA;

  StrInstructionMap ContextArrays;
    
  std::array<llvm::GlobalVariable *, 3> LocalIdIterators;
  std::array<llvm::GlobalVariable *, 3> LocalSizeIterators;
  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;
  std::array<llvm::GlobalVariable *, 3> GroupIdIterators;
  std::array<llvm::Value *, 3> LocalSizeValues;
  llvm::ConstantInt* sgSize;

  size_t TempInstructionIndex;

  std::map<llvm::Instruction *, unsigned> TempInstructionIds;
    
  std::vector<llvm::Instruction*> contextVars;
  std::vector<llvm::AllocaInst*> contextAllocas;

  void identifyContextVars();
  
  llvm::AllocaInst *allocateStorage(llvm::IRBuilder<> &builder,
                                      std::string varName, llvm::Value *nWI);

  llvm::Value *getNumberOfWIs(llvm::IRBuilder<> &Builder);

  llvm::AllocaInst *getContextArray(llvm::Instruction *Inst,
                                      bool &PoclWrapperStructAdded);

  void addContextSaveRestore(llvm::Instruction *instruction);

  llvm::Instruction *addContextSave(llvm::Instruction *Def,
                                      llvm::AllocaInst *AllocaI);

  llvm::Instruction *
    addContextRestore(llvm::Value *Val, llvm::AllocaInst *AllocaI,
                      llvm::Type *LoadInstType, bool PaddingWasAdded,
                      llvm::Instruction *Before = nullptr,
                      bool isAlloca = false);

  void initializeLocalIds(llvm::BasicBlock *Entry, llvm::IRBuilder<> *bldr);

  void initializeGlobalIterators();
                                          
};



llvm::Instruction *FiberImpl::addContextRestore(
    llvm::Value *Val, llvm::AllocaInst *AllocaI, llvm::Type *LoadInstType,
    bool PaddingWasAdded, llvm::Instruction *Before, bool isAlloca) {

  assert(Before != nullptr);

  llvm::Instruction *GEP =
      createContextArrayGEP(AllocaI, Before, PaddingWasAdded);
  if (isAlloca) {
    return GEP;
  }
  llvm::IRBuilder<> Builder(Before);
  return Builder.CreateLoad(LoadInstType, GEP);
}


llvm::Instruction *
FiberImpl::addContextSave(llvm::Instruction *Def,
                          llvm::AllocaInst *AllocaI) {

  if (llvm::isa<llvm::AllocaInst>(Def)) {
    return NULL;
  }

  /* Save the produced variable to the array. */
  llvm::BasicBlock::iterator definition =
      (llvm::dyn_cast<llvm::Instruction>(Def))->getIterator();
  ++definition;
  while (llvm::isa<llvm::PHINode>(definition)) ++definition;

  llvm::IRBuilder<> builder(&*definition);
  std::vector<llvm::Value *> gepArgs;

  if (WGDynamicLocalSize) {
    gepArgs.push_back(getLinearWIIndexInRegion(Def));
  } else {
    llvm::Value *local_x =
        builder.CreateLoad(ST,LocalIdIterators[0],"local_x");
    llvm::Value *local_y =
        builder.CreateLoad(ST,LocalIdIterators[1],"local_y");
    llvm::Value *local_z =
        builder.CreateLoad(ST,LocalIdIterators[2],"local_z");
    gepArgs.push_back(llvm::ConstantInt::get(ST, 0));
    gepArgs.push_back(local_z);
    gepArgs.push_back(local_y);
    gepArgs.push_back(local_x);
  }
  return builder.CreateStore(
      Def,
#if LLVM_MAJOR < 15
      builder.CreateGEP(AllocaI->getType()->getPointerElementType(), AllocaI,
                        gepArgs));
#else
      builder.CreateGEP(AllocaI->getAllocatedType(), AllocaI, gepArgs));
#endif
}

// This is slightly modified version of context save from WIloops.
void FiberImpl::addContextSaveRestore(llvm::Instruction *Def) {

  // Allocate the context data array for the variable.
  bool PaddingAdded = false;
  llvm::AllocaInst *Alloca = getContextArray(Def, PaddingAdded);
  llvm::Instruction *TheStore = addContextSave(Def, Alloca);

  InstructionVec Uses;

  // Find out the uses to fix first as fixing them invalidates the iterator.
  for (llvm::Instruction::use_iterator UI = Def->use_begin(),
       UE = Def->use_end(); UI != UE; ++UI) {
    llvm::Instruction *User = llvm::cast<llvm::Instruction>(UI->getUser());
    if (User == NULL || User == TheStore) continue;
      Uses.push_back(User);
  }

  for (InstructionVec::iterator I = Uses.begin(); I != Uses.end(); ++I) {
    llvm::Instruction *UserI = *I;
    llvm::Instruction *ContextRestoreLocation = UserI;

    llvm::PHINode* Phi = llvm::dyn_cast<llvm::PHINode>(UserI);
    if (Phi != NULL) {
      
      llvm::BasicBlock *IncomingBB = NULL;
      for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues();
           ++Incoming) {
        llvm::Value *Val = Phi->getIncomingValue(Incoming);
        llvm::BasicBlock *BB = Phi->getIncomingBlock(Incoming);
        if (Val == Def)
          IncomingBB = BB;
      }
      assert(IncomingBB != NULL);
      ContextRestoreLocation = IncomingBB->getTerminator();
    }
    llvm::Value *LoadedValue =
        addContextRestore(UserI, Alloca, Def->getType(), PaddingAdded,
                          ContextRestoreLocation, isa<llvm::AllocaInst>(Def));
    UserI->replaceUsesOfWith(Def, LoadedValue);
  }
}

// Override of WIHandler function, not needed in this pass.
llvm::Instruction *
FiberImpl::getLocalIdInRegion(llvm::Instruction *Instr, size_t Dim) {
  
  llvm::IRBuilder<> Builder(Instr);
  return Builder.CreateLoad(ST, LocalIdGlobals[Dim]);
}


// Calculate and return the linear WI id.
llvm::Value *FiberImpl::getLinearWIIndexInRegion(llvm::Instruction* Instr) {

  assert(LocalSizeIterators[0] != NULL && LocalSizeIterators[1] != NULL);

  llvm::IRBuilder<> builder(Instr);

  llvm::LoadInst *LoadXSize =
      builder.CreateLoad(ST, LocalSizeIterators[0], "ls_x");
  llvm::LoadInst *LoadYSize =
      builder.CreateLoad(ST, LocalSizeIterators[1], "ls_y");

  llvm::LoadInst *LoadXId =
      builder.CreateLoad(ST, LocalIdIterators[0], "id_x");
  llvm::LoadInst *LoadYId =
      builder.CreateLoad(ST, LocalIdIterators[1], "id_y");
  llvm::LoadInst *LoadZId =
      builder.CreateLoad(ST, LocalIdIterators[2], "id_z");

  llvm::Value* LocalSizeXTimesY =
      builder.CreateBinOp(llvm::Instruction::Mul, LoadXSize, LoadYSize,
                          "ls_xy");
  llvm::Value *ZPart =
      builder.CreateBinOp(llvm::Instruction::Mul, LocalSizeXTimesY, LoadZId,
                          "tmp");
  llvm::Value *YPart =
      builder.CreateBinOp(llvm::Instruction::Mul, LoadXSize, LoadYId,
                          "ls_x_y");

  llvm::Value* ZYSum =
      builder.CreateBinOp(llvm::Instruction::Add, ZPart, YPart,
                          "zy_sum");
  return builder.CreateBinOp(llvm::Instruction::Add, ZYSum,
                              LoadXId,"linear_xyz_idx");
}

// Collect variables that should be context saved.
void FiberImpl::identifyContextVars()
{ 
  
  for (auto &BB : *F) {
    for (auto &Instr : BB) {

      if (isa<GetElementPtrInst>(&Instr)) {
        GetElementPtrInst *GEP = cast<GetElementPtrInst>(&Instr);
        contextVars.push_back(&Instr);
        continue;
      }
      if (isa<BranchInst>(&Instr))
        continue;

      if (isa<CallInst>(Instr)) {
        Function *F = cast<CallInst>(&Instr)->getCalledFunction();
        if (F && (F == LocalMemAllocaFuncDecl || F == WorkGroupAllocaFuncDecl))
          continue;
      }

      llvm::LoadInst *Load = dyn_cast<llvm::LoadInst>(&Instr);
      if (Load != NULL && (Load->getPointerOperand() == LocalIdGlobals[0] ||
                       Load->getPointerOperand() == LocalIdGlobals[1] ||
                       Load->getPointerOperand() == LocalIdGlobals[2] ||
                       Load->getPointerOperand() == GlobalIdGlobals[0] ||
                       Load->getPointerOperand() == GlobalIdGlobals[1] ||
                       Load->getPointerOperand() == GlobalIdGlobals[2]))
        continue;

      if (!VUA.shouldBePrivatized(Instr.getParent()->getParent(), &Instr))
        continue;

      for (llvm::Instruction::use_iterator UI = Instr.use_begin(),
           UE = Instr.use_end(); UI != UE; ++UI) {
            
        llvm::Instruction *User =
            llvm::dyn_cast<llvm::Instruction>(UI->getUser());

        if (User == NULL)
          continue;

        llvm::BasicBlock* currentBlock = Instr.getParent();

        llvm::BasicBlock* userBlock = User->getParent();

        // Context save not needed if user is in same block
        if (currentBlock == userBlock) {
          continue;
        }
        contextVars.push_back(&Instr);
        break;
      }
    }
  }
}

// Return pointer to 'context array'
llvm::AllocaInst *FiberImpl::getContextArray(llvm::Instruction *Inst,
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

  Var << ".pocl_context";
  std::string CArrayName = Var.str();

  if (ContextArrays.find(CArrayName) != ContextArrays.end())
    return ContextArrays[CArrayName];

  llvm::BasicBlock &Entry = K->getEntryBlock();
  return ContextArrays[CArrayName] = createAlignedAndPaddedContextAlloca(
    Inst, &*(Entry.getFirstInsertionPt()), CArrayName, PaddingAdded);
}


// Initialize local ids as zero
void FiberImpl::initializeLocalIds(BasicBlock *Entry, IRBuilder<> *builder) {

  llvm::GlobalVariable *GVX = LocalIdIterators[0];
  if (GVX != NULL)
    builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GVX);

  llvm::GlobalVariable *GVY = LocalIdIterators[1];
  if (GVY != NULL)
    builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GVY);

  llvm::GlobalVariable *GVZ = LocalIdIterators[2];
  if (GVZ != NULL)
    builder->CreateStore(llvm::ConstantInt::getNullValue(ST), GVZ);
}

// Cast values stored in WorkitemHandler to global variable pointers.
void FiberImpl::initializeGlobalIterators(){

  for (int i = 0; i < 3; i++) {
    // _local_id_xyz
    LocalIdIterators[i] = 
    llvm::cast<llvm::GlobalVariable>(LocalIdGlobals[i]);

    // _local_size_xyz
    LocalSizeIterators[i] = 
    llvm::cast<llvm::GlobalVariable>(LocalSizeGlobals[i]);

    // _global_id_xyz
    GlobalIdIterators[i] =
    llvm::cast<llvm::GlobalVariable>(GlobalIdGlobals[i]);

    // _group_id_xyz
    GroupIdIterators[i] =
    llvm::cast<llvm::GlobalVariable>(GroupIdGlobals[i]);
  }
}

// Calculate and return the number of work items in workgroup.
llvm::Value *FiberImpl::getNumberOfWIs(llvm::IRBuilder<> &Builder){

  llvm::Value *nWI;

  if (WGDynamicLocalSize) {

    llvm::Instruction *loadX = Builder.CreateLoad(ST, LocalSizeGlobals[0]);
    llvm::Instruction *loadY = Builder.CreateLoad(ST, LocalSizeGlobals[1]);
    llvm::Instruction *loadZ = Builder.CreateLoad(ST, LocalSizeGlobals[2]);
    // Store localsizes values for later use.
    LocalSizeValues[0] = loadX;
    LocalSizeValues[1] = loadY;
    LocalSizeValues[2] = loadZ;

    llvm::Value *xy =
        Builder.CreateBinOp(llvm::Instruction::Mul, loadX, loadY);
    llvm::Value *xyz = Builder.CreateBinOp(llvm::Instruction::Mul, xy, loadZ);
        
    xyz->setName("nWI");
    nWI = xyz;

  } else {
    LocalSizeValues[0] = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeX, false);
    LocalSizeValues[1] = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeY, false);
    LocalSizeValues[2] = llvm::ConstantInt::get(
        llvm::Type::getInt64Ty(F->getContext()), WGLocalSizeZ, false);

    nWI =
        llvm::ConstantInt::get(ST,WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ);
  }    
  return nWI;
}


/// @brief Allocates and initializes storage for managing block IDs in the
/// fiber implementation.
/// @param builder The LLVM IR builder used to insert alloca.
/// @param varName The name assigned to the variable in LLVM IR.
/// @param nWI A pointer to an LLVM `Value` representing the number of work
/// items for which storage needs to be allocated.
/// @return A pointer to `llvm::AllocaInst` representing the allocated storage.
llvm::AllocaInst *FiberImpl::allocateStorage(llvm::IRBuilder<> &builder,
                                             std::string varName,
                                             llvm::Value *nWI){
  // Dummy LoadInst. Something like this is required to use WIHandler's
  // allocation functionality.
  llvm::Value *dummyValue = builder.CreateLoad(ST, LocalIdIterators[0],
                                                "dummy");
  llvm::LoadInst *dummyInst = llvm::dyn_cast<llvm::LoadInst>(dummyValue);
  bool PaddingAdded = false;

  // Use WIHandler's allocation machinery to create a proper alloca.
  // This will handle dynamic/non-dynamic cases.
  llvm::AllocaInst *blockIDArray = 
      createAlignedAndPaddedContextAlloca(dummyInst,dummyInst,varName,
                                          PaddingAdded);
    
  // This is not needed anymore.
  dummyInst->eraseFromParent();

  llvm::Value *zero = builder.getInt8(0);
  llvm::MaybeAlign maybeAlign(blockIDArray->getAlign().value());
  llvm::Type *Int64Type = llvm::Type::getInt64Ty(M->getContext());
  uint64_t ElementSizeBytes =
      M->getDataLayout().getTypeAllocSize(Int64Type);

  // Initialize allocated memory to zero.
  if (WGDynamicLocalSize) {
    llvm::ConstantInt *typeSizeVal =
        llvm::ConstantInt::get(M->getContext(),
                                llvm::APInt(64,ElementSizeBytes));
    llvm::Value *totalSize = builder.CreateMul(nWI, typeSizeVal);
    builder.CreateMemSet(blockIDArray, zero, totalSize, maybeAlign);
  } else {
    unsigned long totalSize = WGLocalSizeX * WGLocalSizeY *
                              WGLocalSizeZ * ElementSizeBytes;
    builder.CreateMemSet(blockIDArray, zero,totalSize, maybeAlign);
  }   
  return blockIDArray;
}


bool FiberImpl::runOnFunction(llvm::Function &Func) {

  M = Func.getParent();
  F = &Func;

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
  identifyContextVars();
  for(auto &instr : contextVars){
    addContextSaveRestore(instr);
  }

  llvm::BasicBlock *EntryBlock = nullptr;

  for (auto &BB : Func) {
    if (BB.getName() == "entry.barrier") {
      EntryBlock = &BB;
      break;
    }
  }


  llvm::IRBuilder<> entryBlockBuilder(&*(EntryBlock->getFirstInsertionPt()));

  // Initialize local ids to 0
  initializeLocalIds(EntryBlock, &entryBlockBuilder);

  llvm::Type *Int64Type = llvm::Type::getInt64Ty(M->getContext());

  llvm::Value *nWI = getNumberOfWIs(entryBlockBuilder);

  // Stack storage for block IDs of 'next block' for each WI.
  llvm::AllocaInst *nextJumpIndices = allocateStorage(entryBlockBuilder,
                                                      "jump_indices", nWI);
    
  // Allocate counters, used by scheduler, for each subgroup.
  // Will allocate 'number of work items', which is the worst case situation. 
  llvm::AllocaInst *sg_wi_counter = allocateStorage(entryBlockBuilder,
                                                    "_sg_wi_counter", nWI);
  llvm::AllocaInst *sg_barrier_counter =
      allocateStorage(entryBlockBuilder,"_sg_wi_counter", nWI);
      
  // Type for struct that will store the work group data.
  std::vector<llvm::Type *> wgStateData = {
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
      //sg barriers active
      Int64Type,
      // Counters
      llvm::PointerType::get(Int64Type, 0),
      llvm::PointerType::get(Int64Type, 0),
  };

  llvm::Instruction *lastInst;

  llvm::StructType *wgState = 
      llvm::StructType::get(M->getContext(), wgStateData, "wgState");

  llvm::AllocaInst *wgStateAlloc = 
      entryBlockBuilder.CreateAlloca(wgState, nullptr, "wg_state_data");

  // Get pointers to struct work-group size members.
  llvm::Value *state_local_size_x =
      entryBlockBuilder.CreateGEP(wgState, wgStateAlloc, 
      {llvm::ConstantInt::get(Int64Type, 0), 
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 0)});

  llvm::Value *state_local_size_y =
      entryBlockBuilder.CreateGEP(wgState, wgStateAlloc, 
      {llvm::ConstantInt::get(Int64Type, 0),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 1)});

  llvm::Value *state_local_size_z =
      entryBlockBuilder.CreateGEP(wgState, wgStateAlloc, 
      {llvm::ConstantInt::get(Int64Type, 0),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 2)});

  // Store work-group size values to struct.
  llvm::Instruction *load_x_size =
      entryBlockBuilder.CreateLoad(ST, LocalSizeGlobals[0]);
  entryBlockBuilder.CreateStore(load_x_size, state_local_size_x);

  llvm::Instruction *load_y_size =
      entryBlockBuilder.CreateLoad(ST, LocalSizeGlobals[1]);
  entryBlockBuilder.CreateStore(load_y_size, state_local_size_y);

  llvm::Instruction *load_z_size =
      entryBlockBuilder.CreateLoad(ST, LocalSizeGlobals[2]);
  entryBlockBuilder.CreateStore(load_z_size, state_local_size_z);


  // Pointer to sub-group size member in the struct.
  llvm::Value *sg_size = entryBlockBuilder.CreateGEP(wgState, wgStateAlloc,
      {llvm::ConstantInt::get(Int64Type, 0), 
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), 3)});


  // Store the sub-group size to struct.
  // If specified with intel_reqd_sub_group_size:
  if (llvm::MDNode *SGSizeMD = F->getMetadata("intel_reqd_sub_group_size")) {
        
    llvm::ConstantAsMetadata *ConstMD =
        llvm::cast<llvm::ConstantAsMetadata>(SGSizeMD->getOperand(0));

    uint64_t as64Type =
        (llvm::cast<llvm::ConstantInt>(ConstMD->getValue()))->getZExtValue();
    llvm::ConstantInt *sgSize64 =
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(F->getContext()),
                                 as64Type);
    sgSize = llvm::cast<llvm::ConstantInt>(sgSize64);
    entryBlockBuilder.CreateStore(sgSize64, sg_size);
        
  } else {
    // With dynamic work-group sizes, use the run-time value.
    if (WGDynamicLocalSize) {
      entryBlockBuilder.CreateStore(load_x_size, sg_size);
    // Otherwise use compile-time value.
    } else {
      sgSize = llvm::cast<llvm::ConstantInt>(LocalSizeValues[0]);
      entryBlockBuilder.CreateStore(sgSize, sg_size);
    }
  }

  // Scheduler functions.
  llvm::Function *schedulerInit = M->getFunction("__pocl_fiber_sched_init");
  llvm::Function *wgbarrierReached =
      M->getFunction("__pocl_fiber_wg_barrier_reached");
  llvm::Function *sgbarrierReached =
      M->getFunction("__pocl_fiber_sg_barrier_reached");

  // This will initialise the data structure on scheduler side.
  lastInst = entryBlockBuilder.CreateCall(
      schedulerInit,
      { wgStateAlloc, sg_wi_counter, sg_barrier_counter });
    
  llvm::BasicBlock *currBlock = EntryBlock;

  // Storage for branch instructions after barriers.
  std::vector<llvm::BranchInst*> barrBrInstrs;

  // Store barriers here, for easier handling.
  std::vector<llvm::BasicBlock*> barrierBlocks;
  
  for (auto &Block : Func) {

    if (Barrier::hasBarrier(&Block)) {
        barrierBlocks.push_back(&Block);
    }
  }

  llvm::Value *zeroIndex =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(M->getContext()), 0);
  
  // Store pointer to old exit here
  llvm::BasicBlock* oldExitBlock = nullptr;

  // Create new block for dispatcher; dispathcer block manipulation is
  // done later. Need this for reference for now.
  llvm::BasicBlock *dispatcherBlock =
      llvm::BasicBlock::Create(F->getContext(), "dispatcher", F);

  // Modify the barrier blocks:
  // Three cases that have to be handled somewhat differently.
  // Whenever we have a barrier:
  // (1) Store a copy of the branch instruction in the end of the block.
  // (2) Create branch instruction to dispatcher block.
  // (3) Remove 'old' branch instruction from the current block.
  for (auto &BBlock : barrierBlocks) {

    // With entry barrier it is not necessary to do the ..
    // 
    if (BBlock->getName() == "entry.barrier") {  
      
      if (BBlock->getTerminator()->getNumSuccessors() > 0) {
        assert(isa<BranchInst>(BBlock->getTerminator())
                               && "Expected a BranchInst!");
        barrBrInstrs.push_back(dyn_cast<BranchInst>(
            BBlock->getTerminator()->clone()));
      }
        
      llvm::IRBuilder<> entryBuilder(BBlock->getTerminator());
      entryBuilder.CreateBr(dispatcherBlock);
      
      BBlock->getTerminator()->eraseFromParent();

    // This is the exit block with a barrier.
    // Create additional block to prevent early returns.
    // This way all wis pass through the 'old' exit block.
    } else if (BBlock->getTerminator()->getNumSuccessors() == 0){
             
      // New exit block.
      llvm::BasicBlock *newExitBlock =
          llvm::BasicBlock::Create(F->getContext(), "exit_block", F);

      llvm::IRBuilder<> newExitBlockBuilder(newExitBlock);
      newExitBlockBuilder.CreateRetVoid();
              
      // Branch to new exit.
      llvm::BranchInst *branchInst = llvm::BranchInst::Create(newExitBlock);
      barrBrInstrs.push_back(branchInst);
    
      // Handle for old return block
      llvm::IRBuilder<> oldExitBuilder(BBlock->getTerminator());
      
      llvm::Value *local_z =
          oldExitBuilder.CreateLoad(ST,LocalIdIterators[2],"local_z");
      llvm::Value *local_y =
          oldExitBuilder.CreateLoad(ST,LocalIdIterators[1],"local_y");
      llvm::Value *local_x =
          oldExitBuilder.CreateLoad(ST,LocalIdIterators[0],"local_x");

      llvm::Value *next_block_ptr;

      // Get the 
      if (WGDynamicLocalSize) {
        llvm::Value *linearID =
            getLinearWIIndexInRegion(BBlock->getTerminator());
        next_block_ptr = oldExitBuilder.CreateGEP(
            nextJumpIndices->getAllocatedType(),
            nextJumpIndices,
            {linearID},
            "exit_block_ptr");
      } else {      
        next_block_ptr = oldExitBuilder.CreateGEP(
            nextJumpIndices->getAllocatedType(),
            nextJumpIndices, 
            {zeroIndex, local_z, local_y, local_x},
            "exit_block_ptr");
      }

      // Store the next block index for current WI.
      llvm::Value *next_block_idx =
          llvm::ConstantInt::get(Int64Type, barrBrInstrs.size()-1);
      oldExitBuilder.CreateStore(next_block_idx, next_block_ptr);
           
      // Barrier in exit block is always work-group barrier.
      oldExitBuilder.CreateCall(
          wgbarrierReached,
          {local_x, local_y, local_z, wgStateAlloc});
            
      oldExitBuilder.CreateBr(dispatcherBlock);

      // Remove previous 'ret void' instruction.
      BBlock->getTerminator()->eraseFromParent();

      // These are "Explicit" barriers.
    } else {

      assert(isa<BranchInst>(BBlock->getTerminator())
          && "Expected a BranchInst!");
      barrBrInstrs.push_back(dyn_cast<BranchInst>(
          BBlock->getTerminator()->clone()));
            
      llvm::IRBuilder<> barrierBlockBuilder(BBlock->getTerminator());
        
      llvm::Value *local_z =
          barrierBlockBuilder.CreateLoad(ST,LocalIdIterators[2],"local_z");
      llvm::Value *local_y =
          barrierBlockBuilder.CreateLoad(ST,LocalIdIterators[1],"local_y");
      llvm::Value *local_x =
          barrierBlockBuilder.CreateLoad(ST,LocalIdIterators[0],"local_x");

      llvm::Value *next_block_ptr;

      if (WGDynamicLocalSize) {
        llvm::Value *linearID =
            getLinearWIIndexInRegion(BBlock->getTerminator());
        next_block_ptr = barrierBlockBuilder.CreateGEP(
            nextJumpIndices->getAllocatedType(),
            nextJumpIndices,
            {linearID},
            "exit_block_ptr");
      } else {      
        next_block_ptr = barrierBlockBuilder.CreateGEP(
            nextJumpIndices->getAllocatedType(),
            nextJumpIndices, 
            {zeroIndex, local_z, local_y, local_x},
            "exit_block_ptr");
      }
        
      llvm::Value *next_block_idx =
          llvm::ConstantInt::get(Int64Type, barrBrInstrs.size()-1);
      barrierBlockBuilder.CreateStore(next_block_idx, next_block_ptr);

      // Register work-group/sub-group barrier entry
      if (SubgroupBarrier::hasSGBarrier(BBlock)) {
        barrierBlockBuilder.CreateCall(
            sgbarrierReached,
            {local_x, local_y, local_z, wgStateAlloc});
      } else {
        barrierBlockBuilder.CreateCall(
            wgbarrierReached,
            {local_x, local_y, local_z,wgStateAlloc});
      }
      
      // Add branch to dispatcher
      barrierBlockBuilder.CreateBr(dispatcherBlock);

      // Remove the old branch
      BBlock->getTerminator()->eraseFromParent();
    }
  }

  // Dispatcher implementation
  llvm::IRBuilder<> bBuilder(dispatcherBlock);

  // Function call to __pocl_sched_work_item to retrieve next WI id.
  llvm::Function *schedFunc =
      M->getFunction("__pocl_fiber_schedule_work_item");

  // Retrieve the return value, i.e. WI id.
  llvm::Value *linearWI = bBuilder.CreateCall(schedFunc, {wgStateAlloc});
  linearWI->setName("next_linear_wi");

  // 'Unlinearize' the WI id.
  // X 
  llvm::Value *loc_x = bBuilder.CreateBinOp(
      llvm::Instruction::BinaryOps::SRem,
      linearWI,
      LocalSizeValues[0],
      "loc_id_x");
    
  llvm::Value *xtimesy = bBuilder.CreateBinOp(
      llvm::Instruction::BinaryOps::Mul,
      LocalSizeValues[0],
      LocalSizeValues[1]);
    
  // Y
  llvm::Value *loc_y_tmp = bBuilder.CreateBinOp(
      llvm::Instruction::BinaryOps::SRem,
      linearWI,xtimesy,
      "loc_id_y_tmp");
  llvm::Value *loc_y = bBuilder.CreateBinOp(
      llvm::Instruction::UDiv,
      loc_y_tmp,
      LocalSizeValues[0],
      "loc_id_y");

  // Z
  llvm::Value *loc_z = bBuilder.CreateBinOp(
      llvm::Instruction::UDiv, linearWI, xtimesy, "loc_id_z");

  // Store new local ids.
  bBuilder.CreateStore(loc_x, LocalIdIterators[0]);
  bBuilder.CreateStore(loc_y, LocalIdIterators[1]);
  bBuilder.CreateStore(loc_z, LocalIdIterators[2]);

  // Calculate global ids.
  llvm::Value* x_gid =
      bBuilder.CreateLoad(ST, GroupIdGlobals[0], "group_id_x");
  llvm::Value* y_gid =
      bBuilder.CreateLoad(ST, GroupIdGlobals[1], "group_id_y");
  llvm::Value* z_gid =
      bBuilder.CreateLoad(ST, GroupIdGlobals[2], "group_id_z");

  llvm::Value* multX = bBuilder.CreateMul(LocalSizeValues[0], x_gid, "mulx");
  llvm::Value* multY = bBuilder.CreateMul(LocalSizeValues[1], y_gid, "muly");
  llvm::Value* multZ = bBuilder.CreateMul(LocalSizeValues[2], z_gid, "mulz");

  llvm::Value* mul_x_loc = bBuilder.CreateAdd(multX, loc_x, "mul_x_loc");
  llvm::Value* mul_y_loc = bBuilder.CreateAdd(multY, loc_y, "mul_y_loc");
  llvm::Value* mul_z_loc = bBuilder.CreateAdd(multZ, loc_z, "mul_z_loc");

  llvm::GlobalVariable *offsetXPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_global_offset_x", ST));
  llvm::GlobalVariable *offsetYPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_global_offset_y", ST));
  llvm::GlobalVariable *offsetZPtr =
      cast<GlobalVariable>(M->getOrInsertGlobal("_global_offset_z", ST));

  llvm::Value* offset_x = bBuilder.CreateLoad(ST, offsetXPtr,"offset_x");
  llvm::Value* offset_y = bBuilder.CreateLoad(ST, offsetYPtr,"offset_y");
  llvm::Value* offset_z = bBuilder.CreateLoad(ST, offsetZPtr,"offset_z");

  llvm::Value* gid_x = bBuilder.CreateAdd(mul_x_loc, offset_x, "gid_x");
  llvm::Value* gid_y = bBuilder.CreateAdd(mul_y_loc, offset_y, "gid_y");
  llvm::Value* gid_z = bBuilder.CreateAdd(mul_z_loc, offset_z, "gid_z");

  // Store global ids.
  bBuilder.CreateStore(gid_x, GlobalIdIterators[0]);
  bBuilder.CreateStore(gid_y, GlobalIdIterators[1]);
  lastInst = bBuilder.CreateStore(gid_z, GlobalIdIterators[2]);

  // Pointer to next block for current WI.
  llvm::Value *next_block_ptr;
  if (WGDynamicLocalSize) {
    llvm::Value *linearID = getLinearWIIndexInRegion(lastInst);
    next_block_ptr = bBuilder.CreateGEP(
        nextJumpIndices->getAllocatedType(),
        nextJumpIndices,
        {linearID},
        "exit_block_ptr");
  } else {
    next_block_ptr = bBuilder.CreateGEP(
        nextJumpIndices->getAllocatedType(),
        nextJumpIndices,
        {zeroIndex, loc_z, loc_y, loc_x},
        "exit_block_ptr");
  }

  // Retrieve next block index.
  llvm::Value *loadedValue = bBuilder.CreateLoad(
      bBuilder.getInt64Ty(),
      next_block_ptr,
      "next_block");
    

  // Add the switch statement and handle jumping to 'after-barrier' blocks.
  if (barrBrInstrs.size() > 0) {

    llvm::SwitchInst *switchInst;

    // For each after-barrier block, create a 'helper block' in which there 
    // will be:
    // (1) load from condition variable, IF branch is conditional.
    // (2) branch (either conditional or unconditional) to after-barrier block.
    for(int i = 0; i < barrBrInstrs.size(); i++){
      
      std::string caseBlockName = "case"+std::to_string(i);
      llvm::BasicBlock *caseBlock =
          llvm::BasicBlock::Create(F->getContext(), caseBlockName, F);

      llvm::IRBuilder<> caseBlockBuilder(caseBlock);

      // In case of conditional branch, allocate storage in entry block
      // and add manual context save/restore.
      // TODO: has to be array instead of single boolean (one for each WI).
      if(barrBrInstrs[i]->isConditional()){
        
        llvm::Type* condType = barrBrInstrs[i]->getCondition()->getType();
        llvm::AllocaInst *alloc =
            entryBlockBuilder.CreateAlloca(condType, nullptr, "disp_br_cond");

        // Context save the condition variable where calculated.
        llvm::Value *condition = barrBrInstrs[i]->getCondition();

        llvm::Instruction *definingInst =
            llvm::dyn_cast<llvm::Instruction>(condition);

        llvm::IRBuilder<>builder(definingInst->getNextNode());
        llvm::Instruction* saved = builder.CreateStore(definingInst,alloc);

        llvm::Instruction* restore =
            caseBlockBuilder.CreateLoad(condType, alloc, "cond_restore");
        
        barrBrInstrs[i]->setCondition(restore);
      }

      caseBlockBuilder.Insert(barrBrInstrs[i]);
       
      llvm::ConstantInt *caseValue =
          llvm::ConstantInt::get(bBuilder.getInt64Ty(), i);


      if(i == 0)
        switchInst = bBuilder.CreateSwitch(loadedValue, caseBlock);
      else
        switchInst->addCase(caseValue, caseBlock);

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
 
    
bool addFiber(llvm::Function &F, llvm::DominatorTree &DT,
              llvm::PostDominatorTree &PDT, llvm::LoopInfo &LI,
              VariableUniformityAnalysisResult &VUA) {
  
  WorkitemHandlerType WIH = getWorkitemHandler();

  if (WIH != WorkitemHandlerType::FIBER) {
    return false;
  }

  FiberImpl fiber(DT, VUA);
  return fiber.runOnFunction(F);
}

} // namespace pocl
