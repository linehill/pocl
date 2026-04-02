// Header for WorkitemHandler, a parent class for all implementations of
// work-group generation.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
//               2024-2025 Pekka Jääskeläinen / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef POCL_WORKITEM_HANDLER_H
#define POCL_WORKITEM_HANDLER_H

#include "config.h"

#include "Kernel.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "WorkitemHandlerChooser.h"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>
#include <llvm/Support/CommandLine.h>

#include <iostream>

namespace llvm {
class DominatorTree;
}

namespace pocl {

// Common base class for work-group function generators that includes
// utility functionality.
class WorkitemHandler {
public:
  // Should be called when starting to process a new kernel.
  void initialize(pocl::Kernel *K, WorkitemHandlerType WIH);

protected:
  WorkitemHandlerType WIH = WorkitemHandlerType::INVALID;

  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;
  using InstructionVec = std::vector<llvm::Instruction *>;
  ParallelRegion::ParallelRegionVector OriginalParallelRegions;

  llvm::Instruction *getGlobalSize(int Dim);
  llvm::Instruction *getGlobalIdOrigin(int dim);

  void GenerateGlobalIdComputation();

  llvm::AllocaInst *createAlignedAndPaddedContextAlloca(
      llvm::Instruction *Inst, llvm::Instruction *Before,
      const std::string &Name, bool &PaddingAdded);

  // The handler should override these to return the linear and n-dimensional
  // work-item index in the parallel region with the given \param Instr.
  // The Value should be reachable by the given \param Instr.
  virtual llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr) {
    assert(false &&
           "getLinearWIIndexInRegion() not implemented for the handler");
    return nullptr;
  };
  virtual llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr,
                                                size_t Dim) {
    return nullptr;
  };

  virtual llvm::Instruction *getGlobalIdInRegion(llvm::Instruction *Instr,
                                                 size_t Dim) {
    return nullptr;
  };

  bool shouldNotBeContextSaved(llvm::Instruction *Instr,
                               VariableUniformityAnalysisResult &VUA,
                               WorkitemHandlerType WIH);
  llvm::AllocaInst *getContextArray(llvm::Instruction *Inst,
                                    bool &PoclWrapperStructAdded);
  llvm::Value *getLinearWiIndex(llvm::IRBuilder<> &Builder, llvm::Module *M,
                                ParallelRegion *Region,
                                WorkitemHandlerType WIH);
  llvm::Instruction *addContextSave(llvm::Instruction *Def,
                                    llvm::AllocaInst *AllocaI,
                                    ParallelRegion *Region);
  llvm::Instruction *
  addContextRestore(llvm::Value *Val, llvm::AllocaInst *AllocaI,
                    llvm::Type *LoadInstType, bool PaddingWasAdded,
                    llvm::Instruction *Before = nullptr, bool IsAlloca = false);
  ParallelRegion *regionOfBlock(llvm::BasicBlock *BB);

  void addContextSaveRestore(llvm::Instruction *Instruction, llvm::LoopInfo &LI,
                             VariableUniformityAnalysisResult *VUA = nullptr,
                             llvm::DominatorTree *DT = nullptr);

  llvm::GetElementPtrInst *
  createContextArrayGEP(llvm::AllocaInst *CtxArrayAlloca,
                        llvm::Instruction *Before, bool AlignPadding);

  bool canAnnotateParallelLoops();
  bool handleLocalMemAllocas();

  /// Removes (pseudo)barrier calls from the current kernel.
  bool removeBarrierCalls();
  bool handleWorkitemFunctions();

  llvm::Instruction *getWorkGroupSizeInstr();

  llvm::Value *
  tryToRematerialize(llvm::Instruction *Before, llvm::Value *Def,
                     std::string NamePrefix,
                     VariableUniformityAnalysisResult *VUA = nullptr,
                     llvm::DominatorTree *DT = nullptr, bool *CanDoIt = nullptr,
                     int *Depth = 0);

  // The type of size_t for the current target.
  llvm::Type *ST;
  // The width of size_t for the current target.
  int SizeTWidth;

  // The Module global variables that hold the place of the current local
  // id until privatized.
  std::array<llvm::Value *, 3> LocalIdGlobals;
  std::array<llvm::Value *, 3> LocalSizeGlobals;
  std::array<llvm::Value *, 3> GlobalIdGlobals;
  std::array<llvm::Value *, 3> GroupIdGlobals;
  std::array<llvm::Value *, 3> NumGroupsGlobals;
  std::array<llvm::Value *, 3> GlobalOffsetGlobals;

  llvm::Value *LLID;

  // Keeps track of subgroups in SG-loops.
  llvm::Value *SgInterCounter;
  // Keeps track of work-items in SG-loops.
  llvm::Value *SgIntraCounter;

  llvm::Value *NXLanes;

  // Local id:s of the first WI of the currently executing subgroup.
  llvm::Value *sg_locals_x;
  llvm::Value *sg_locals_y;
  llvm::Value *sg_locals_z;

  llvm::Value *sg_localg_x;
  llvm::Value *sg_localg_y;
  llvm::Value *sg_localg_z;

  llvm::Value *Y_LowerLimit;
  llvm::Value *Y_UpperLimit;

  llvm::Value *sg_current_llid;

  llvm::Value *LocalLinearId;

  // Points to the global size computation instructions in the entry
  // block of the currently handled kernel.
  std::array<llvm::Instruction *, 3> GlobalSizes;

  // Points to the global id origin computation instructions in the entry
  // block of the currently handled kernel. To get the global id, the current
  // local id must be added to it.
  std::array<llvm::Instruction *, 3> GlobalIdOrigins;

  // Points to the __pocl_local_mem_alloca pseudo function declaration, if
  // it's been referred to in the currently processed module.
  llvm::Function *LocalMemAllocaFuncDecl;

  // Points to the __pocl_work_group_alloca pseudo function declaration, if
  // it's been referred to in the currently processed module.
  llvm::Function *WorkGroupAllocaFuncDecl;

  // Points to the work-group size computation instruction in the entry
  // block of the currently handled kernel. Will be created with the first
  // call to getWorkGroupSizeInstr.
  llvm::Instruction *WGSizeInstr;

  // The currently handled Kernel/Function and its Module.
  pocl::Kernel *K;
  llvm::Module *M;

  // Copies of compilation parameters
  std::string KernelName;
  unsigned long AddressBits;
  bool WGAssumeZeroGlobalOffset;
  bool WGDynamicLocalSize;
  bool DeviceUsingArgBufferLauncher;
  bool DeviceIsSPMD;
  unsigned long WGLocalSizeX;
  unsigned long WGLocalSizeY;
  unsigned long WGLocalSizeZ;
  unsigned long WGMaxGridDimWidth;

  std::map<llvm::Instruction *, unsigned> TempInstructionIds;
  size_t TempInstructionIndex;
  StrInstructionMap ContextArrays;
};

}

#endif
