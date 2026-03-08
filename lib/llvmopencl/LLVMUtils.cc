// Implementation of LLVMUtils, useful common LLVM-related functionality.
//
// Copyright (c) 2013-2019 Pekka Jääskeläinen
//               2023-2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/ADT/SmallSet.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ReplaceConstant.h>

// include all passes & analysis
#include "AllocasToEntry.h"
#include "AutomaticLocals.h"
#include "Barrier.h"
#include "CanonicalizeBarriers.h"
#include "DeSPMD.h"
#include "DebugHelpers.h"
#include "Fiber.h"
#include "Flatten.hh"
#include "FlattenBarrierSubs.hh"
#include "FlattenGlobals.hh"
#include "HandleSamplerInitialization.h"
#include "ImplicitConditionalBarriers.h"
#include "ImplicitLoopBarriers.h"
#include "InlineKernels.hh"
#include "IsolateRegions.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "LoopBarriers.h"
#include "MarkAllInlineable.hh"
#include "MinLegalVecSize.hh"
#include "OptimizeBuiltins.h"
#include "OptimizeWorkItemGVars.h"
#include "PHIsToAllocas.h"
#include "ParallelRegion.h"
#include "SanitizeUBofDivRem.h"
#include "SubCFGFormation.h"
#include "SubgroupBarrier.h"
#include "UnreachablesToReturns.h"
#include "VariableUniformityAnalysis.h"
#include "Workgroup.h"
#include "WorkgroupBarrier.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"

POP_COMPILER_DIAGS


#include "pocl_llvm_api.h"
#include "pocl_spir.h"

#include <iostream>
#include <regex>
#include <set>

using namespace llvm;

//#define DEBUG_LLVM_UTILS

static void findInstructionUsesImpl(Use &U, std::vector<Use *> &Uses,
                                    std::set<Use *> &Visited) {
  if (Visited.count(&U))
    return;
  Visited.insert(&U);

  assert(isa<Constant>(*U));
  if (isa<Instruction>(U.getUser())) {
    Uses.push_back(&U);
    return;
  }
  if (isa<Constant>(U.getUser())) {
    for (auto &U : U.getUser()->uses())
      findInstructionUsesImpl(U, Uses, Visited);
    return;
  }

  // Catch other user kinds - we may need to process them (somewhere but not
  // here).
  llvm_unreachable("Unexpected user kind.");
}

// Return list of non-constant leaf use edges whose users are instructions.
static std::vector<Use *> findInstructionUses(GlobalVariable *GVar) {
  std::vector<Use *> Uses;
  std::set<Use *> Visited;
  for (auto &U : GVar->uses())
    findInstructionUsesImpl(U, Uses, Visited);
  return Uses;
}


// Remove address space qualifiers like U3AS4, U3AS1, etc. from mangled symbol
static std::string stripAddressSpaces(const std::string &MangledName) {

    // Pattern: U followed by digits, then AS, then a digit
    std::regex Pattern(R"(U\d+AS\d)");
    std::string Result = std::regex_replace(MangledName, Pattern, "");

    return Result;
}


namespace pocl {

std::string tryDemangleWithoutAddressSpaces(const std::string& MangledName) {

    std::string Demangled = llvm::demangle(MangledName);
    if (Demangled != MangledName) {
        return Demangled;
    }

    std::string Stripped = stripAddressSpaces(MangledName);

    Demangled = llvm::demangle(Stripped);
    if (Demangled != Stripped) {
        return Demangled;
    }

    return MangledName; // Failed
}

/**
 * Regenerates the metadata that points to the original kernel
 * (of which finger print was modified) to point to the new
 * kernel.
 *
 * Only checks if the first operand of the metadata is the kernel
 * function.
 */
void regenerateKernelMetadata(llvm::Module &M, FunctionMapping &kernels) {
  // reproduce the opencl.kernel_wg_size_info metadata

  NamedMDNode *WgSizes = M.getNamedMetadata("opencl.kernel_wg_size_info");
  if (WgSizes != NULL && WgSizes->getNumOperands() > 0) {
    for (std::size_t Mni = 0; Mni < WgSizes->getNumOperands(); ++Mni) {
      MDNode *WgSizeMD = dyn_cast<MDNode>(WgSizes->getOperand(Mni));
      for (FunctionMapping::const_iterator I = kernels.begin(),
                                           E = kernels.end();
           I != E; ++I) {
        Function *OldKernel = (*I).first;
        Function *NewKernel = (*I).second;
        Function *FuncFromMd;
        FuncFromMd = dyn_cast<Function>(
            dyn_cast<ValueAsMetadata>(WgSizeMD->getOperand(0))->getValue());
        if (OldKernel == NewKernel || WgSizeMD->getNumOperands() == 0 ||
            FuncFromMd != OldKernel)
          continue;
        // found a wg size metadata that points to the old kernel, copy its
        // operands except the first one to a new MDNode
        SmallVector<Metadata *, 8> Operands;
        Operands.push_back(llvm::ValueAsMetadata::get(NewKernel));
        for (unsigned Opr = 1; Opr < WgSizeMD->getNumOperands(); ++Opr) {
          Operands.push_back(WgSizeMD->getOperand(Opr));
        }
        MDNode *NewWgMd = MDNode::get(M.getContext(), Operands);
        WgSizes->addOperand(NewWgMd);
      }
    }
  }

  // reproduce the opencl.kernels metadata, if it exists
  // unconditionally adding opencl.kernels confuses the
  // metadata parser in pocl_llvm_metadata.cc, which uses
  // "opencl.kernels" to distinguish old SPIR format from new
  NamedMDNode *Nmd = M.getNamedMetadata("opencl.kernels");
  if (Nmd) {
    M.eraseNamedMetadata(Nmd);

    Nmd = M.getOrInsertNamedMetadata("opencl.kernels");
    for (FunctionMapping::const_iterator I = kernels.begin(), E = kernels.end();
         I != E; ++I) {
      MDNode *Md = MDNode::get(
          M.getContext(),
          ArrayRef<Metadata *>(llvm::ValueAsMetadata::get((*I).second)));
      Nmd->addOperand(Md);
    }
  }
}

#if LLVM_MAJOR == 18
// Recursively descend a Value's users and convert any constant expressions into
// regular instructions.
void breakConstantExpressions(llvm::Value *Val, llvm::Function *Func) {
  std::vector<llvm::Value *> Users(Val->user_begin(), Val->user_end());
  for (auto *U : Users) {
    if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(U)) {
      // First, make sure no users of this constant expression are themselves
      // constant expressions.
      breakConstantExpressions(U, Func);

      // Convert this constant expression to an instruction.
      llvm::Instruction *I = CE->getAsInstruction();
      I->insertBefore(&*Func->begin()->begin());
      CE->replaceAllUsesWith(I);
      CE->destroyConstant();
    }
  }
}
#else
void breakConstantExpressions(llvm::Value *Val, llvm::Function *Func) {
  std::vector<llvm::Constant *> Constants;
  for (auto *U : Val->users()) {
    if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(U))
      Constants.push_back(CE);
  }
  convertUsersOfConstantsToInstructions(Constants, Func, true, true);
}
#endif

static void
recursivelyFindCalledFunctions(llvm::SmallSet<llvm::Function *, 12> &FSet,
                               llvm::Function *F) {
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr))
        continue;
      CallInst *CallInstr = dyn_cast<CallInst>(Instr);
      Function *Callee = CallInstr->getCalledFunction();
      if (!Callee)
        continue;
      if (Callee->isDeclaration())
        continue;
      if (FSet.contains(Callee))
        continue;
      FSet.insert(Callee);
      recursivelyFindCalledFunctions(FSet, Callee);
    }
  }
}

bool isGVarUsedByFunction(llvm::GlobalVariable *GVar, llvm::Function *F) {
  std::vector<Use *> Uses = findInstructionUses(GVar);
  // we must recursively search for each function called by F, because
  // this (isGVarUsedByFunction) is called by isAutomaticLocal(),
  // which in turn is called on "unprocessed" LLVM bitcode (or SPIRV),
  // where we haven't run any LLVM passes yet; in particular the pass
  // that inlines all functions using "special" variables and kernels
  llvm::SmallSet<llvm::Function *, 12> CalledFunctionSet;
  CalledFunctionSet.insert(F);
  recursivelyFindCalledFunctions(CalledFunctionSet, F);
  std::vector<Function *> Funcs;
  for (auto &U : Uses) {
    if (Instruction *I = dyn_cast<Instruction>(U->getUser()))
    {
      if (CalledFunctionSet.contains(I->getFunction()))
        return true;
    }
  }
  return false;
}

// Erase lifetime.start markers which reference invalid values
// Only valid values are Poison & Alloca
void eraseInvalidLifetimeMarkers(llvm::Function *F) {
    for (BasicBlock &BB : *F) {
        for (Instruction &I : llvm::make_early_inc_range(BB)) {
            auto *II = dyn_cast<LifetimeIntrinsic>(&I);
            if (!II)
                continue;

            Value *Mem = II->getOperand(0);
            if (isa<PoisonValue>(Mem) || isa<AllocaInst>(Mem))
                continue;

            II->eraseFromParent();
        }
    }
}

bool
isAutomaticLocal(llvm::Function *F, llvm::GlobalVariable &Var) {
  // Without the fake address space IDs, there is no reliable way to figure out
  // if the address space is local from the bitcode. We could check its AS
  // against the device's local address space id, but for now lets rely on the
  // naming convention only. Only relying on the naming convention has the problem
  // that LLVM can move private const arrays to the global space which make
  // them look like local arrays (see Github Issue 445). This should be properly
  // fixed in Clang side with e.g. a naming convention for the local arrays to
  // detect them robstly without having logical address space info in the IR.
  std::string FuncName = F->getName().str();
  if (!llvm::isa<llvm::PointerType>(Var.getType()) || Var.isConstant())
    return false;
  if (Var.getName().starts_with(FuncName + ".")) {
    return true;
  }

  // handle SPIR local AS (3)
  if (Var.getParent() && Var.getParent()->getNamedMetadata("spirv.Source") &&
      (Var.getType()->getAddressSpace() == SPIR_ADDRESS_SPACE_LOCAL)) {

    if (!Var.hasName())
      Var.setName(llvm::Twine(FuncName, ".__anon_gvar"));
    // check it's used by this particular function
    return isGVarUsedByFunction(&Var, F);
  }

  return false;
}

void eraseFunctionAndCallers(llvm::Function *Function) {
  if (!Function)
    return;

  std::vector<llvm::Value *> Callers(Function->user_begin(),
                                     Function->user_end());
  for (auto &U : Callers) {
    llvm::CallInst *Call = llvm::dyn_cast<llvm::CallInst>(U);
    if (!Call)
      continue;
    Call->eraseFromParent();
  }
  Function->eraseFromParent();
}

int getConstantIntMDValue(Metadata *MD) {
  ConstantInt *CI = mdconst::extract<ConstantInt>(MD);
  return CI->getLimitedValue();
}

llvm::Metadata *createConstantIntMD(llvm::LLVMContext &C, int32_t Val) {
  IntegerType *I32Type = IntegerType::get(C, 32);
  return ConstantAsMetadata::get(ConstantInt::get(I32Type, Val));
}

llvm::DISubprogram *mimicDISubprogram(llvm::DISubprogram *Old,
                                      const llvm::StringRef &NewFuncName,
                                      llvm::DIScope *Scope) {

  return DISubprogram::getDistinct(
      Old->getContext(), Old->getScope(), NewFuncName, "", Old->getFile(),
      Old->getLine(), Old->getType(), Old->getScopeLine(),
      Old->getContainingType(), Old->getVirtualIndex(),
      Old->getThisAdjustment(), Old->getFlags(), Old->getSPFlags(),
      Old->getUnit(), Old->getTemplateParams(), Old->getDeclaration());
}

bool isLocalMemFunctionArg(llvm::Function *F, unsigned ArgIndex) {

  MDNode *MD = F->getMetadata("kernel_arg_addr_space");

  if (MD == nullptr || MD->getNumOperands() <= ArgIndex)
    return false;
  else
    return getConstantIntMDValue(MD->getOperand(ArgIndex)) ==
           SPIR_ADDRESS_SPACE_LOCAL;
}

bool isProgramScopeVariable(GlobalVariable &GVar, unsigned DeviceLocalAS) {

  bool RetVal = false;

  // no need to handle constants
  if (GVar.isConstant()) {
    RetVal = false;
    goto END;
  }

  // program-scope variables from direct Clang compilation have external
  // linkage with Target AS numbers
  if (GVar.getLinkage() == GlobalValue::LinkageTypes::ExternalLinkage) {
    RetVal = true;
    goto END;
  }

#ifdef DEBUG_LLVM_UTILS
  std::cerr << "isProgramScopeVariable: checking variable: " <<
            GVar.getName().str() << "\n";
#endif

  // global variables from SPIR-V have internal linkage with SPIR AS numbers
  if (GVar.getLinkage() == GlobalValue::LinkageTypes::InternalLinkage) {
#ifdef DEBUG_LLVM_UTILS
    std::cerr << "isProgramScopeVariable: checking internal linkage\n";
#endif
    PointerType *GVarT = GVar.getType();
    assert(GVarT != nullptr);
    unsigned AddrSpace = GVarT->getAddressSpace();

    if (AddrSpace == SPIR_ADDRESS_SPACE_GLOBAL) {
#ifdef DEBUG_LLVM_UTILS
      std::cerr << "isProgramScopeVariable: AS = SPIR Global AS\n";
#endif
      if (!GVar.hasName()) {
        GVar.setName("__anonymous_gvar");
      }
      RetVal = true;
    }

    // variables in local AS cannot have initializer (OpenCL standard).
    // for CPU target, Local AS = Global AS = 0, and
    // function-scope variables ("static global X = {...};")
    // must be recognized as program-scope variables
    if (GVar.hasInitializer()) {
      Constant *C = GVar.getInitializer();
      bool isUndef = isa<UndefValue>(C);
      if (AddrSpace == DeviceLocalAS && !isUndef) {
#ifdef DEBUG_LLVM_UTILS
        std::cerr << "isProgramScopeVariable: AS = device's Local AS && "
                     "isUndef == false\n";
#endif
        if (!GVar.hasName()) {
          GVar.setName("__anonymous_gvar");
        }
        RetVal = true;
      }
    }
  }

END:
#ifdef DEBUG_LLVM_UTILS
  std::cerr << "isProgramScopeVariable: \n"
            << "Variable: " << GVar.getName().str()
            << " is ProgramScope variable: " << retval << "\n";

#endif
  return RetVal;
}

bool isPureUniformBlock(BasicBlock *BB) {
  return BB->getTerminator()->hasMetadata(PoCLMDKind::PureUniformBasicBlock);
}

void markAsPureUniformBlock(BasicBlock *BB, std::string Reason) {
  BB->getTerminator()->setMetadata(
      PoCLMDKind::PureUniformBasicBlock,
      llvm::MDNode::get(BB->getContext(),
                        {llvm::MDString::get(BB->getContext(), Reason)}));
}

void unmarkAsPureUniformBlock(llvm::BasicBlock *BB) {
  assert(BB);
  unsigned TargetKind = BB->getParent()->getParent()->getMDKindID(
      PoCLMDKind::PureUniformBasicBlock);
  auto *TI = cast<Instruction>(BB->getTerminator());
  TI->eraseMetadataIf([=](unsigned MDKind, MDNode *Node) -> bool {
    return MDKind == TargetKind;
  });
}

void copyPureUniformMD(llvm::BasicBlock *Source,
                       llvm::BasicBlock *Destination) {
  if (!isPureUniformBlock(Source))
    return;
  Destination->getTerminator()->setMetadata(
      PoCLMDKind::PureUniformBasicBlock,
      Source->getTerminator()->getMetadata(PoCLMDKind::PureUniformBasicBlock));
}


void setFuncArgAddressSpaceMD(llvm::Function *F, unsigned ArgIndex,
                              unsigned AS) {

  unsigned MDKind = F->getContext().getMDKindID("kernel_arg_addr_space");
  MDNode *OldMD = F->getMetadata(MDKind);

  assert(OldMD == nullptr || OldMD->getNumOperands() >= ArgIndex);

  LLVMContext &C = F->getContext();

  llvm::SmallVector<llvm::Metadata *, 8> AddressQuals;
  for (unsigned I = 0; I < ArgIndex; ++I) {
    AddressQuals.push_back(createConstantIntMD(
        C, OldMD != nullptr ? getConstantIntMDValue(OldMD->getOperand(I))
                            : SPIR_ADDRESS_SPACE_GLOBAL));
  }
  AddressQuals.push_back(createConstantIntMD(C, AS));
  F->setMetadata(MDKind, MDNode::get(F->getContext(), AddressQuals));
}

void markFunctionAlwaysInline(llvm::Function *F) {
  F->removeFnAttr(Attribute::NoInline);
  F->removeFnAttr(Attribute::OptimizeNone);
  F->addFnAttr(Attribute::AlwaysInline);
  // remove noInline from the callsite. otherwise it could cause alwaysInline
  // pass to skip the inlining
  for (auto U: F->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      // If the call has vector-function-abi-variant, removing NoInline from it
      // would prevent the attribute from working correctly.
      if (!CI->hasFnAttr("vector-function-abi-variant"))
        CI->removeFnAttr(Attribute::NoInline);

      CI->removeFnAttr(Attribute::NoBuiltin);
      CI->removeFnAttr(Attribute::OptimizeNone);
    }
  }
}

// Returns true in case the given function is a kernel that
// should be processed by the kernel compiler.
bool isKernelToProcess(const llvm::Function &F) {

  const Module *M = F.getParent();

  if (F.getMetadata("kernel_arg_access_qual") &&
      F.getMetadata("pocl_generated") == nullptr)
    return true;

  if (F.isDeclaration())
    return false;
  if (!F.hasName())
    return false;
  if (F.getName().starts_with("@llvm"))
    return false;

  NamedMDNode *Kernels = M->getNamedMetadata("opencl.kernels");
  if (Kernels == NULL) {

    std::string KernelName;
    bool HasMeta = getModuleStringMetadata(*M, "KernelName", KernelName);

    if (HasMeta && KernelName.size() && F.getName().str() == KernelName)
      return true;

    return false;
  }

  for (unsigned I = 0, E = Kernels->getNumOperands(); I != E; ++I) {
    if (Kernels->getOperand(I)->getOperand(0) == NULL)
      continue; // globaldce might have removed uncalled kernels
    Function *K = cast<Function>(
        dyn_cast<ValueAsMetadata>(Kernels->getOperand(I)->getOperand(0))
            ->getValue());
    if (&F == K)
      return true;
  }

  return false;
}

// Returns true in case the given function is a kernel with work-group
// barriers inside it.
bool hasWorkgroupBarriers(const llvm::Function &F) {
  for (llvm::Function::const_iterator I = F.begin(), E = F.end(); I != E; ++I) {
    const llvm::BasicBlock *BB = &*I;
    if (pocl::Barrier::hasBarrier(BB)) {

      // Ignore the implicit entry and exit barriers.
      if (pocl::Barrier::hasOnlyBarrier(BB) && BB == &F.getEntryBlock())
        continue;

      if (pocl::Barrier::hasOnlyBarrier(BB) &&
          BB->getTerminator()->getNumSuccessors() == 0)
        continue;

      return true;
    }
  }
  return false;
}

// walks through a Module's global variables,
// determines which ones are OpenCL program-scope variables
// and checks all of those have definitions
bool areAllGvarsDefined(llvm::Module *Program, std::string &log,
                        std::set<llvm::GlobalVariable *> &GVarSet,
                        unsigned DeviceLocalAS) {

  bool FoundAllReferences = true;

  for (GlobalVariable &GVar : Program->globals()) {

    if (isProgramScopeVariable(GVar, DeviceLocalAS)) {

      assert(GVar.hasName());
      // adding GV declarations to the module also changes
      // the global iteration to include them
      if (GVarSet.count(&GVar) != 0)
        continue;

      if (GVar.isDeclaration()) {
        log.append("Undefined reference for program scope variable: ");
        log.append(GVar.getName().data());
        log.append("\n");
        FoundAllReferences = false;
      } else {
        GVarSet.insert(&GVar);
      }
    }
  }

  return FoundAllReferences;
}

// for a set of program scope variables,
// calculate their offsets & sizes for later replacement with
// indexing into a single large buffer
// @returns the total size of all variables
size_t
calculateGVarOffsetsSizes(const DataLayout &DL,
                          std::map<GlobalVariable *, uint64_t> &GVarOffsets,
                          std::set<llvm::GlobalVariable *> &GVarSet) {

  std::map<GlobalVariable *, uint64_t> GVarSizes;

  // offset into the storage buffer for all of this program's global variables
  size_t CurrentOffset = 0;

  for (GlobalVariable *GVar : GVarSet) {
    assert(GVar->hasInitializer());

    // if the current offset into the buffer is not aligned enough, fix it
    Align GVarA = GVar->getAlign().valueOrOne();
    uint64_t GVarAlign = GVarA.value();

    if (GVarAlign > 0 && CurrentOffset % GVarAlign) {
      CurrentOffset |= (GVarAlign - 1);
      ++CurrentOffset;
    }
    GVarOffsets[GVar] = CurrentOffset;

    // add to the offset the required amount of storage for the global variable
    TypeSize GVSize = DL.getTypeAllocSize(GVar->getValueType());
    assert(GVSize.isScalable() == false);
    GVarSizes[GVar] = GVSize.getFixedValue();
    CurrentOffset += GVarSizes[GVar];

#ifdef POCL_DEBUG_PROGVARS
    std::cerr << "@@@ GlobalVar: " << GVar->getName().str()
              << "\n   OFFSET: " << GVarOffsets[GVar]
              << "\n   SIZE: " << GVarSizes[GVar] << "\n";
#endif
  }

  size_t TotalSize = CurrentOffset;
  return TotalSize;
}

const char *WorkgroupVariablesArray[NumWorkgroupVariables+1] = {"_local_id_x",
                                    "_local_id_y",
                                    "_local_id_z",
                                    "_local_size_x",
                                    "_local_size_y",
                                    "_local_size_z",
                                    "_work_dim",
                                    "_num_groups_x",
                                    "_num_groups_y",
                                    "_num_groups_z",
                                    "_group_id_x",
                                    "_group_id_y",
                                    "_group_id_z",
                                    "_global_offset_x",
                                    "_global_offset_y",
                                    "_global_offset_z",
                                    "_global_id_x",
                                    "_global_id_y",
                                    "_global_id_z",
                                    "_pocl_sub_group_size",
                                    "_local_linear_id",
                                    "_sg_intra_counter",
                                    "_sg_inter_counter",
                                    "_n_x_lanes",
                                    "_sg_y_lower_limit",
                                    "_sg_y_upper_limit",
                                    PoclGVarBufferName,
                                    NULL};

const std::vector<std::string>
    WorkgroupVariablesVector(WorkgroupVariablesArray,
                             WorkgroupVariablesArray+NumWorkgroupVariables);

const char *WIFuncNameArray[] = {
    GID_BUILTIN_NAME,          GOFF_BUILTIN_NAME,    GS_BUILTIN_NAME,
    GROUP_ID_BUILTIN_NAME,     LID_BUILTIN_NAME,     LS_BUILTIN_NAME,
    ENQUEUE_LS_BUILTIN_NAME,   NGROUPS_BUILTIN_NAME, GLID_BUILTIN_NAME,
    LLID_BUILTIN_NAME,         WDIM_BUILTIN_NAME,    LLID_BUILTIN_NAME,

    GSID_BUILTIN_NAME,         SGS_BUILTIN_NAME,     NSGROUPS_BUILTIN_NAME,
    MAXSGS_BUILTIN_NAME,       GESID_BUILTIN_NAME,   GSLID_BUILTIN_NAME,
    "__pocl_work_group_alloca"};

constexpr unsigned NumWIFuncNames =
    sizeof(WIFuncNameArray) / sizeof(const char *);

const std::vector<std::string> WIFuncNameVec(WIFuncNameArray,
                                             WIFuncNameArray + NumWIFuncNames);

const char *DIFuncNameArray[NumDIFuncNames] = {GID_BUILTIN_NAME,
                                               GOFF_BUILTIN_NAME,
                                               GS_BUILTIN_NAME,
                                               GROUP_ID_BUILTIN_NAME,
                                               LID_BUILTIN_NAME,
                                               LS_BUILTIN_NAME,
                                               ENQUEUE_LS_BUILTIN_NAME,
                                               NGROUPS_BUILTIN_NAME,
                                               GLID_BUILTIN_NAME,
                                               LLID_BUILTIN_NAME,
                                               WDIM_BUILTIN_NAME,
                                               "pocl_printf_alloc",
                                               "pocl_printf_alloc_stub"};

const std::vector<std::string> DIFuncNameVec(DIFuncNameArray,
                                             DIFuncNameArray + NumDIFuncNames);

// register all PoCL analyses & passes with an LLVM PassBuilder instance,
// so that it can parse them from string representation
void registerPassBuilderPasses(llvm::PassBuilder &PB) {
  AllocasToEntry::registerWithPB(PB);
  AutomaticLocals::registerWithPB(PB);
  SanitizeUBofDivRem::registerWithPB(PB);
  ConvertUnreachablesToReturns::registerWithPB(PB);
  llvm::DeSPMDPass::registerWithPB(PB);
  FlattenAll::registerWithPB(PB);
  FlattenBarrierSubs::registerWithPB(PB);
  FlattenGlobals::registerWithPB(PB);
  HandleSamplerInitialization::registerWithPB(PB);
  InlineKernels::registerWithPB(PB);
  IsolateRegions::registerWithPB(PB);
  FixMinVecSize::registerWithPB(PB);
  OptimizeWorkItemGVars::registerWithPB(PB);
  OptimizeBuiltins::registerWithPB(PB);
  SubCFGFormation::registerWithPB(PB);
  Workgroup::registerWithPB(PB);
  PoCLCFGPrinter::registerWithPB(PB);
  MarkAllInlineable::registerWithPB(PB);
}

void registerFunctionAnalyses(llvm::PassBuilder &PB) {
  VariableUniformityAnalysis::registerWithPB(PB);
}

llvm::Type *SizeT(llvm::Module *M) {
  unsigned long AddressBits;
  getModuleIntMetadata(*M, "device_address_bits", AddressBits);
  return IntegerType::get(M->getContext(), AddressBits);
}

const std::array<std::string, 8> CompilerExpandableBuiltinNames = {
    GID_BUILTIN_NAME,  GS_BUILTIN_NAME,  GROUP_ID_BUILTIN_NAME,
    LID_BUILTIN_NAME,  LS_BUILTIN_NAME,  NGROUPS_BUILTIN_NAME,
    GOFF_BUILTIN_NAME, LLID_BUILTIN_NAME};

bool isWorkitemFunctionWithOnlyCompilerExpandableCalls(
    const llvm::Function &F) {

  if (std::find(CompilerExpandableBuiltinNames.begin(),
                CompilerExpandableBuiltinNames.end(),
                F.getName()) == CompilerExpandableBuiltinNames.end())
    return false;

  for (const auto &U : F.uses()) {
    llvm::CallInst *Call = dyn_cast<llvm::CallInst>(U.getUser());
    if (Call == nullptr)
      continue;
    if (!isCompilerExpandableWIFunctionCall(*Call))
      return false;
  }
  return true;
}

bool isCompilerExpandableWIFunctionCall(const llvm::CallInst &Call) {
  auto Callee = Call.getCalledFunction();
  if (Callee == nullptr /* Inline asm? */)
    return false;

  if (std::find(CompilerExpandableBuiltinNames.begin(),
                CompilerExpandableBuiltinNames.end(),
                Callee->getName()) == CompilerExpandableBuiltinNames.end())
    return false;
  // Can expand only if the builtin doesn't take any arguments
  // (get_local_linear_id) or the argument (dimension) is known at compile time.
  return Call.arg_size() == 0 || isa<llvm::ConstantInt>(Call.getArgOperand(0));
}

bool removeClangGeneratedKernelStubs(llvm::Module *Program) {
#if LLVM_MAJOR > 20
#ifdef DEBUG_LLVM_UTILS
  std::cerr << "removeClangGeneratedKernelStubs: Dump of Program BEFORE:\n";
  Program->dump();
#endif
  // For now, inline & remove all Clang-generated kernel wrappers
  llvm::SmallSet<llvm::Function *, 8> RemoveFunctionList;
  llvm::Module::iterator FI, FE;

  for (FI = Program->begin(), FE = Program->end(); FI != FE; FI++) {
    if (FI->hasName() && FI->getName().starts_with("__clang_ocl_kern_imp")) {
      RemoveFunctionList.insert(&*FI);
    }
  }
  bool retval = true;
  for (auto F : RemoveFunctionList) {
#ifdef DEBUG_LLVM_UTILS
    std::cerr << "Erasing function : " << F->getName().str() << "  "
              << " Num uses: " << (unsigned)F->getNumUses() << "\n";
#endif
    llvm::SmallSet<Value *, 8> FUsers;
    for (auto U : F->users()) {
      FUsers.insert(U);
    }
    for (auto U : FUsers) {
      CallInst *CInstr = dyn_cast<CallInst>(U);
      if (CInstr) {
#ifdef DEBUG_LLVM_UTILS
        CInstr->dump();
        std::cerr << "Use is CallInstr, inlining\n";
#endif
        InlineFunctionInfo IFI;
        InlineResult IR = llvm::InlineFunction(*CInstr, IFI);
        if (!IR.isSuccess()) {
#ifdef DEBUG_LLVM_UTILS
          std::cerr << "Inlining failed with reason: %s \n"
                    << IR.getFailureReason()) << "\n";
#endif
          retval = false;
        }
      } else {
#ifdef DEBUG_LLVM_UTILS
        std::cerr << "UNKNOWN Use: \n";
        U->dump();
#endif
      }
    }
    if (F->getNumUses() == 0) {
#ifdef DEBUG_LLVM_UTILS
      std::cerr << "Zero Uses remain, erasing \n";
#endif
      F->eraseFromParent();
    } else {
#ifdef DEBUG_LLVM_UTILS
      std::cerr << "NOT DELETING: Uses remain: " << (unsigned)F->getNumUses()
                << "\n";
#endif
      retval = false;
    }
  }

#ifdef DEBUG_LLVM_UTILS
  std::cerr << "removeClangGeneratedKernelStubs: Dump of Program AFTER:\n";
  Program->dump();
#endif

  return retval;
#else
  return true;
#endif
}

bool removeMetadataFromClangStubs(llvm::Module *Program) {
#if LLVM_MAJOR > 20
  // For now, remove all Clang-generated kernel wrappers
  // these have incorrect metadata, which causes an assertion later (in metadata
  // extraction)
  llvm::Module::iterator FI, FE;

  for (FI = Program->begin(), FE = Program->end(); FI != FE; FI++) {
    if (FI->hasName() && FI->getName().starts_with("__clang_ocl_kern_imp")) {
      // remove the OpenCL kernel argument metadata from the stub function.
      // the function will not be recognized as a kernel by other code
      FI->setMetadata("kernel_arg_addr_space", nullptr);
      FI->setMetadata("kernel_arg_access_qual", nullptr);
      FI->setMetadata("kernel_arg_type", nullptr);
      FI->setMetadata("kernel_arg_base_type", nullptr);
      FI->setMetadata("kernel_arg_type_qual", nullptr);
      FI->setMetadata("kernel_arg_name", nullptr);
    }
  }
#ifdef DEBUG_LLVM_UTILS
  std::cerr << "removeMetadataFromClangStubs: Dump of Program AFTER:\n";
  Program->dump();
#endif
#endif
  return true;
}

/// Replace subgroup barrier with workgroup barrier, and vice versa.
///
/// \param BB the basic block in which replacement happens.
void switchBarrierGranularity(llvm::BasicBlock *Bb) {

  assert(pocl::Barrier::hasBarrier(Bb) &&
         "Basic block does not contain a barrier to swap!");

  bool HasSGBarr = SubgroupBarrier::hasSGBarrier(Bb);

  if (HasSGBarr)
    WorkgroupBarrier::createAtEnd(Bb);
  else
    SubgroupBarrier::createAtEnd(Bb);

  llvm::Instruction *BarrierToRemove;

  for (auto It = Bb->begin(); It != Bb->end(); ++It) {
    llvm::Instruction *Current = &*It;
    if (isa<SubgroupBarrier>(Current) && HasSGBarr)
      BarrierToRemove = Current;
    else if (isa<WorkgroupBarrier>(Current) && !HasSGBarr)
      BarrierToRemove = Current;
  }
  BarrierToRemove->eraseFromParent();
}

static GlobalVariable *getOrCreateWILoopBoundGV(Module *M, StringRef Name) {
  assert(M);
  auto *Poison = PoisonValue::get(SizeT(M));

  if (auto *GV = M->getGlobalVariable(Name, /*AllowInternal=*/true)) {
    if (!GV->hasInitializer())
      GV->setInitializer(Poison);
    GV->setLinkage(GlobalValue::InternalLinkage);
    return GV;
  }

  return new GlobalVariable(*M, SizeT(M), false, GlobalValue::InternalLinkage,
                            Poison, Name);
}

// Set Range metadata with range [Min, Max] to the given instruction.
void setRangeMetadata(llvm::Instruction *Instr, size_t Min, size_t Max) {
  assert(Min != Max + 1 && "Empty/full range!");

  MDBuilder MDB(Instr->getContext());
  size_t BitWidth = Instr->getType()->getIntegerBitWidth();

  assert(isUIntN(BitWidth, Min) && "Min value doesn't fit into RangeMD!");
  assert(isUIntN(BitWidth, Max + 1) && "Max+1 value doesn't fit into RangeMD!");

  MDNode *Range =
      MDB.createRange(APInt(BitWidth, Min), APInt(BitWidth, Max + 1));
  Instr->setMetadata(LLVMContext::MD_range, Range);
}

/// Get or create lower work-item loop bound global variable.
GlobalVariable *getOrCreateWILoopLowerBoundGV(Module *M, unsigned Dim) {
  return getOrCreateWILoopBoundGV(M, WILOOP_LOWER_BOUND_NAME(Dim));
}

/// Get or create upper work-item loop bound global variable.
GlobalVariable *getOrCreateWILoopUpperBoundGV(Module *M, unsigned Dim) {
  return getOrCreateWILoopBoundGV(M, WILOOP_UPPER_BOUND_NAME(Dim));
}

/// Materialize the result of get_local_size(<Dim>) - possibly as constant if
/// it's known.
Value *getWorkgroupLocalSize(Module *M, unsigned Dim,
                             BasicBlock::iterator InsPt) {
  assert(M);
  assert(Dim < 3);

  auto *ST = SizeT(M);
  bool WGDynamicLocalSize = true;
  getModuleBoolMetadata(*M, "WGDynamicLocalSize", WGDynamicLocalSize);

  if (!WGDynamicLocalSize) {
    static const char *WGSizeNames[] = {"WGLocalSizeX", "WGLocalSizeY",
                                        "WGLocalSizeZ"};
    uint64_t WGSize = 0;
    if (getModuleIntMetadata(*M, WGSizeNames[Dim], WGSize)) {
      assert(WGSize && "Invalid static workgroup size");
      return ConstantInt::get(ST, WGSize);
    }

    // TODO: what is correct resolution for missing WG-local size metadata?
  }

  auto *WGSizeGV = M->getGlobalVariable(LS_G_NAME(Dim));
  if (!WGSizeGV)
    WGSizeGV = new GlobalVariable(
        *M, ST, true, GlobalValue::CommonLinkage, nullptr, LS_G_NAME(Dim),
        nullptr, GlobalValue::ThreadLocalMode::NotThreadLocal, 0, true);

  IRBuilder<> B(&*InsPt);
  return B.CreateLoad(ST, WGSizeGV, Twine("wg_size_") + Twine(Dim));
}

/// Indicates whether the work-item loop bounds may change during execution of
/// the kernel.
///
/// If false, the loop bounds may change. If true, the loop bounds are
/// unchanging and covers all WIs in the WG.
///
/// This predicate is only applicable for kernels using
/// WorkitemHandlerType::LOOPS method.
bool hasInvariantWILoopBounds(Function *F) {
  bool Result;
  if (getModuleBoolMetadata(*F->getParent(), "pocl.invariant_wiloop_bounds",
                            Result))
    return Result;

  // Absent pocl.invariant_wiloop_bounds MD is meant to imply (potentially)
  // dynamic WI-loop bounds.
  return false;
}

/// Returns true if 'F' may call a function by 'CalleeName' name.
bool hasCallTo(Function *F, StringRef CalleeName) {
  auto *M = F->getParent();
  auto *Callee = M->getFunction(CalleeName);
  if (!Callee)
    return false;

  SmallPtrSet<Function *, 8> Visited;
  SmallVector<Function *> Worklist({Callee});

  while (!Worklist.empty()) {
    auto *Caller = Worklist.pop_back_val();
    if (Caller == F)
      return true;

    for (Use &U : Caller->uses()) {
      auto *CI = dyn_cast<CallInst>(U.getUser());
      if (!CI)
        continue;
      auto *NextCaller = CI->getParent()->getParent();
      if (!Visited.contains(NextCaller)) {
        Visited.insert(NextCaller);
        Worklist.push_back(NextCaller);
      }
    }
  }

  return false;
}

/// Creates a call to a function with a function type corresponding to
/// 'size_t(int)' in OpenCL C.
static Value *createWorkItemFnCall(Module *M, const char *Name, Value *ArgV,
                                   BasicBlock::iterator InsPt) {
  assert(Name);
  auto *ArgT = IntegerType::get(M->getContext(), 32);
  auto *RetT = SizeT(M);

  FunctionType *FTy = FunctionType::get(RetT, {ArgT}, /*isVarArg=*/false);
  auto FC = M->getOrInsertFunction(Name, FTy);
  assert(FC.getFunctionType() == FTy && "Function type mismatch!");

  IRBuilder<> B(InsPt->getParent(), InsPt);
  return B.CreateCall(FC, {ArgV}, Name);
}

static Value *createWorkItemFnCall(Module *M, const char *Name, unsigned Dim,
                                   BasicBlock::iterator InsPt) {
  assert(Dim <= 3);
  auto *ArgT = IntegerType::get(M->getContext(), 32);
  auto *ArgV = ConstantInt::get(ArgT, Dim, /*IsSigned=*/false);
  return createWorkItemFnCall(M, Name, ArgV, InsPt);
}

Value *createGetGroupID(Module *M, unsigned Dim, BasicBlock::iterator InsPt) {
  return createWorkItemFnCall(M, GROUP_ID_BUILTIN_NAME, Dim, InsPt);
}

Value *createGetLocalSize(Module *M, unsigned Dim, BasicBlock::iterator InsPt) {
  return createWorkItemFnCall(M, LS_BUILTIN_NAME, Dim, InsPt);
}

/// Emit code for computing value of 'get_group_id(Dim) * get_local_size(Dim)'
Value *createBaseGlobalID(Module *M, unsigned Dim, BasicBlock::iterator InsPt) {

  IRBuilder<> B(InsPt->getParent(), InsPt);
  auto *GroupID = createGetGroupID(M, Dim, InsPt);
  auto *LocalSize = createGetLocalSize(M, Dim, InsPt);
  auto Name = Twine("gid.base.") + Twine(Dim);
  return B.CreateMul(GroupID, LocalSize, Name, /*NUW=*/true);
}

llvm::Value *createGlobalID(llvm::Module *M, Value *Dim,
                            llvm::BasicBlock::iterator InsPt) {
  IRBuilder<> B(InsPt->getParent(), InsPt);
  return createWorkItemFnCall(M, GID_BUILTIN_NAME, Dim, InsPt);
}

/// Same as Value::getNameOrAsOperand() but available with non-debug LLVM before
/// LLVM 21.
///
/// This method return the name assigned to the V or its SSA number.
std::string getNameOrAsOperand(Value *V) {
#if LLVM_MAJOR >= 21
  // Before LLVM-21 this method is guarded by NDEBUG.
  return V->getNameOrAsOperand();
#else
  // Copied from Value::getNameOrAsOperand().
  if (!V->getName().empty())
    return std::string(V->getName());

  std::string BBName;
  raw_string_ostream OS(BBName);
  V->printAsOperand(OS, false);
  return OS.str();
#endif
}

} // namespace pocl
