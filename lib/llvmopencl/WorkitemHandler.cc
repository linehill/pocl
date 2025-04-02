// Base class for passes that generate work-group functions out of a bunch
// of work-items.
//
// Copyright (c) 2011-2012 Carlos Sánchez de La Lama / URJC and
//               2012-2019 Pekka Jääskeläinen
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
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/Support/CommandLine.h>

#include "DebugHelpers.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "WorkitemHandler.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"

POP_COMPILER_DIAGS

#include "pocl_llvm_api.h"

#include <iostream>
#include <sstream>

#include "Barrier.h"

POP_COMPILER_DIAGS

namespace pocl {

using namespace llvm;

// Compiler-expanded function that can be used to allocate "local memory"
// dynamically in the work-group function. Used by SG/WG shuffle implementations
// as temporary storage.
constexpr const char *POCL_LOCAL_MEM_ALLOCA_FUNC_NAME =
    "__pocl_local_mem_alloca";

// Another which multiplies the given size by the number of WIs in the WG.
constexpr const char *POCL_WORK_GROUP_ALLOCA_FUNC_NAME =
    "__pocl_work_group_alloca";

/// Start processing a new kernel.
///
/// Should be invoked from the work-item handlers to initialize the internal
/// per-kernel data.
void WorkitemHandler::Initialize(Kernel *K_) {

  K = K_;
  M = K->getParent();

  WIH = getWorkitemHandler();

  LocalMemAllocaFuncDecl =
      K->getParent()->getFunction(POCL_LOCAL_MEM_ALLOCA_FUNC_NAME);

  WorkGroupAllocaFuncDecl =
      K->getParent()->getFunction(POCL_WORK_GROUP_ALLOCA_FUNC_NAME);

  WGSizeInstr = nullptr;

  getModuleIntMetadata(*M, "device_address_bits", AddressBits);

  getModuleStringMetadata(*M, "KernelName", KernelName);
  getModuleIntMetadata(*M, "WGMaxGridDimWidth", WGMaxGridDimWidth);
  getModuleIntMetadata(*M, "WGLocalSizeX", WGLocalSizeX);
  getModuleIntMetadata(*M, "WGLocalSizeY", WGLocalSizeY);
  getModuleIntMetadata(*M, "WGLocalSizeZ", WGLocalSizeZ);
  getModuleBoolMetadata(*M, "WGDynamicLocalSize", WGDynamicLocalSize);
  getModuleBoolMetadata(*M, "WGAssumeZeroGlobalOffset",
                        WGAssumeZeroGlobalOffset);

  if (WGLocalSizeX == 0)
    WGLocalSizeX = 1;
  if (WGLocalSizeY == 0)
    WGLocalSizeY = 1;
  if (WGLocalSizeZ == 0)
    WGLocalSizeZ = 1;

  SizeTWidth = AddressBits;
  ST = pocl::SizeT(M);

  LocalIdGlobals = {M->getOrInsertGlobal(LID_G_NAME(0), ST),
                    M->getOrInsertGlobal(LID_G_NAME(1), ST),
                    M->getOrInsertGlobal(LID_G_NAME(2), ST)};

  LocalSizeGlobals = {M->getOrInsertGlobal(LS_G_NAME(0), ST),
                      M->getOrInsertGlobal(LS_G_NAME(1), ST),
                      M->getOrInsertGlobal(LS_G_NAME(2), ST)};

  GlobalIdGlobals = {M->getOrInsertGlobal(GID_G_NAME(0), ST),
                     M->getOrInsertGlobal(GID_G_NAME(1), ST),
                     M->getOrInsertGlobal(GID_G_NAME(2), ST)};

  GroupIdGlobals = {M->getOrInsertGlobal(GROUP_ID_G_NAME(0), ST),
                    M->getOrInsertGlobal(GROUP_ID_G_NAME(1), ST),
                    M->getOrInsertGlobal(GROUP_ID_G_NAME(2), ST)};

  NumGroupsGlobals = {M->getOrInsertGlobal(NGROUPS_G_NAME(0), ST),
                      M->getOrInsertGlobal(NGROUPS_G_NAME(1), ST),
                      M->getOrInsertGlobal(NGROUPS_G_NAME(2), ST)};

  GlobalOffsetGlobals = {M->getOrInsertGlobal(GOFFS_G_NAME(0), ST),
                         M->getOrInsertGlobal(GOFFS_G_NAME(1), ST),
                         M->getOrInsertGlobal(GOFFS_G_NAME(2), ST)};

  GlobalIdOrigins = {0, 0, 0};
  GlobalSizes = {0, 0, 0};
}

/// Determines whether the given instruction should be context saved.
///
/// Note that there are a few cases where the behavior differs between
/// workitem handlers. Workgroup methods call this method on their side
/// and perform additional filtering (excluding some variables that
/// this method flags for saving).
///
/// \param Instr The Instruction which is the context save candidate.
/// \param VUA The VariableUniformityAnalysisResult.
/// \param WIH The workitem handler type.
/// \return A boolean, whether to context save Instr or not.
bool WorkitemHandler::shouldNotBeContextSaved(
    llvm::Instruction *Instr, VariableUniformityAnalysisResult &VUA,
    WorkitemHandlerType WIH) {

  if (isa<BranchInst>(Instr))
    return true;

  if (AllocaInst *Alloca = dyn_cast<AllocaInst>(Instr)) {
    // Some of the variables such as B-loop iterators must not be
    // replicated for correctness.
    if (VUA.isPureUniformAlloca(Alloca))
      return true;
  }

  // Skip everything else in case of Fiber for now. Not optimal, but causes
  // problems with current implementation.
  if (WIH == WorkitemHandlerType::FIBER)
    return false;

  // Generated id loads should not be replicated as it leads to problems in
  // conditional branch case where the header node of the region is shared
  // across the peeled branches and thus the header node's ID loads might get
  // context saved which leads to egg-chicken problems.
  llvm::LoadInst *Load = dyn_cast<llvm::LoadInst>(Instr);
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
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### based on VUA, not context saving:";
    Instr->dump();
#endif
    return true;
  }

  return false;
}

/// Returns the context array (alloca) for the given \param Inst, creates it if
/// not found.
///
/// \param PaddingAdded will be set to true in case a wrapper struct was
/// added for padding in order to enforce proper alignment to the elements of
/// the array. Such padding might be needed to ensure aligned accessed from
/// single work-items accessing aggregates in the context data.
llvm::AllocaInst *WorkitemHandler::getContextArray(llvm::Instruction *Inst,
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

/// Adds a value store to the context array after the given defining
/// instruction.
///
/// \param Def The instruction that defines the original value.
/// \param AllocaI The alloca created for for the context array.
/// \TODO synch by hand from WorkitemLoops.cc upstream.
llvm::Instruction *WorkitemHandler::addContextSave(llvm::Instruction *Def,
                                                   llvm::AllocaInst *AllocaI,
                                                   ParallelRegion *Region) {

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

  if (WGDynamicLocalSize) {
    if (WIH == WorkitemHandlerType::FIBER) {
      GepArgs.push_back(getLinearWiIndex(Builder, M, nullptr, WIH));
    } else {
      Module *M = AllocaI->getParent()->getParent()->getParent();
      GepArgs.push_back(getLinearWiIndex(Builder, M, Region, WIH));
    }
  } else {
    GepArgs.push_back(ConstantInt::get(ST, 0));

    if (WIH == WorkitemHandlerType::FIBER) {
      GepArgs.push_back(Builder.CreateLoad(ST, LocalIdGlobals[2], "LocalZ"));
      GepArgs.push_back(Builder.CreateLoad(ST, LocalIdGlobals[1], "LocalY"));
      GepArgs.push_back(Builder.CreateLoad(ST, LocalIdGlobals[0], "LocalX"));
    } else {
      GepArgs.push_back(Region->getOrCreateIDLoad(LID_G_NAME(2)));
      GepArgs.push_back(Region->getOrCreateIDLoad(LID_G_NAME(1)));
      GepArgs.push_back(Region->getOrCreateIDLoad(LID_G_NAME(0)));
    }
  }

  return Builder.CreateStore(
      Def,
#if LLVM_MAJOR < 15
      builder.CreateGEP(AllocaI->getType()->getPointerElementType(), AllocaI,
                        gepArgs));
#else
      Builder.CreateGEP(AllocaI->getAllocatedType(), AllocaI, GepArgs));
#endif
}

llvm::Instruction *WorkitemHandler::addContextRestore(
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

ParallelRegion *WorkitemHandler::regionOfBlock(llvm::BasicBlock *BB) {
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

/// Tries to rematerialize the given value-defining instruction.
///
/// Rematerialization in this context means recomputing the value produced
/// in the use site instead of storing and loading a once-computed variable
/// from the context.
///
/// \param Before the instruction before which the cloned instructions should
/// be added.
/// \param Def is the produced value to attempt to clone recursively.
/// \param NamePrefix a prefix string to add to the name of the cloned
/// instructions.
/// \param CanDoIt can be set to a true-initialized boolean in which case the
/// cloning is not actually done, but only its possibility is investigated.
/// \param Depth the recursion depth. Used to limit rematerialization size.
/// \return The rematerialized instruction if possible and beneficial.
/// \TODO synch this by hand from upstream since it's in WorkitemLoops.cc
/// there.
llvm::Value *WorkitemHandler::tryToRematerialize(llvm::Instruction *Before,
  llvm::Value *Def,
  std::string NamePrefix,
  bool *CanDoIt, int *Depth) {

  auto DbgRemat = [=](const std::string &Reason) {
#ifdef DEBUG_WORK_ITEM_LOOPS
  std::cerr << "##### " << Reason << "\n";
  Def->dump();
#endif
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
    dyn_cast<AllocaInst>(Def)->getParent() != &K->getEntryBlock()) {
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
    Copy->insertBefore(Before);
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
/// First attemps to rematerialize the value instead of storing it to memory.
/// \todo SYNCH by hand from upstream WorkitemHandler::addContextSaveRestore
void WorkitemHandler::addContextSaveRestore(llvm::Instruction *Def, llvm::LoopInfo &LI) {

  InstructionVec Uses;
  // Restore the produced variable before each use to ensure the correct
  // context copy is used.

  bool RematCandidate = true;

  // In case of a rematerialized alloca with only a single store, this will have
  // the store that initializes it.
  StoreInst *InitializerStore = nullptr;
  size_t Stores = 0;
  ParallelRegion *PrevStoreRegion = nullptr;

  // Find out the uses to fix first as fixing them invalidates the iterator.
  for (Instruction::use_iterator UI = Def->use_begin(), UE = Def->use_end();
       UI != UE; ++UI) {

    llvm::Instruction *User = cast<Instruction>(UI->getUser());

    if (WIH == WorkitemHandlerType::FIBER) {
      Uses.push_back(User);
      continue;
    }

    if (User == NULL)
      continue;

    ParallelRegion *PRegion = regionOfBlock(User->getParent());

    if (StoreInst *ST = dyn_cast<StoreInst>(User)) {
      if (!isa<UndefValue>(ST->getValueOperand())) {
        Stores++;

        if (Stores == 1) {
          InitializerStore = ST;
        } else {
          InitializerStore = nullptr;
          RematCandidate = false;
#ifdef DEBUG_WORK_ITEM_LOOPS
          std::cerr << "#### Multiple stores\n";
          User->dump();
#endif
        }
        if (PrevStoreRegion == nullptr) {
          PrevStoreRegion = PRegion;
        } else if (PrevStoreRegion != PRegion) {
          RematCandidate = false;
#ifdef DEBUG_WORK_ITEM_LOOPS
          std::cerr << "#### Stores from multiple regions\n";
          User->dump();
#endif
        }

        if (LI.getLoopFor(ST->getParent()) != nullptr) {
          RematCandidate = false;
#ifdef DEBUG_WORK_ITEM_LOOPS
          std::cerr << "#### Stores from inside a loop\n";
          User->dump();
#endif
        }
      }
    }

    // If the user is in a block that doesn't belong to a region, the variable
    // itself must be a "work group variable", that is, not dependent on the
    // work item. Most likely an iteration variable of a for loop with a
    // barrier.
    if (PRegion == nullptr) {
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "#### user in a pure uniform block?\n";
      User->dump();
#endif
      continue;
    }

    if (isa<CallInst>(User)) {
      if (!User->isLifetimeStartOrEnd()) {
        RematCandidate = false;
#ifdef DEBUG_WORK_ITEM_LOOPS
        std::cerr << "#### using in an unknown call\n";
        User->dump();
#endif
      }
    } else if (llvm::AllocaInst *Alloca = dyn_cast_or_null<AllocaInst>(Def)) {
      if (!isa<StoreInst>(User) && !isa<LoadInst>(User)) {
        RematCandidate = false;
#ifdef DEBUG_WORK_ITEM_LOOPS
        std::cerr << "#### taking address of the alloca?\n";
        User->dump();
#endif
      } else {
        // If we perform reinterpret casts, let's not rematerialize as it might
        // require to store the value temporarily to stack.
        if ((isa<LoadInst>(User) &&
             User->getType() != Alloca->getAllocatedType()) ||
            (isa<StoreInst>(User) &&
             User->getOperand(0)->getType() != Alloca->getAllocatedType())) {
#ifdef DEBUG_WORK_ITEM_LOOPS
          std::cerr << "#### Found a user with a different pointee type\n";
          User->dump();
          Def->dump();
#endif
          RematCandidate = false;
        }
      }
    }

    Uses.push_back(User);
  }

  llvm::AllocaInst *ContextArrayAlloca = nullptr;
  bool PaddingAdded = false;

  for (Instruction *UserI : Uses) {
    Instruction *ContextRestoreLocation = UserI;

    PHINode* Phi = dyn_cast<PHINode>(UserI);
    if (Phi != NULL) {
      // TODO: This is now obsolete. For source input we work on unoptimized
      // clang output and for SPIR-V we break down the PHIs.

      // In case of PHI nodes, we cannot just insert the context restore code
      // before it in the same basic block because it is assumed there are no
      // non-phi Instructions before PHIs which the context restore code
      // constitutes to. Add the context restore to the incomingBB instead.

      // There can be values in the PHINode that are incoming from another
      // region even though the decision BB is within the region. For those
      // values we need to add the context restore code in the incoming BB
      // (which is known to be inside the region due to the assumption of not
      // having to touch PHI nodes in PRentry BBs).

      // PHINodes at region entries are broken down earlier.
      assert ("Cannot add context restore for a PHI node at the region entry!"
               && regionOfBlock(
                Phi->getParent())->entryBB() != Phi->getParent());
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "#### adding context restore code before PHI" << std::endl;
      UserI->dump();
      std::cerr << "#### in BB:" << std::endl;
      UserI->getParent()->dump();
#endif
      BasicBlock *IncomingBB = NULL;
      for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues();
           ++Incoming) {
        Value *Val = Phi->getIncomingValue(Incoming);
        BasicBlock *BB = Phi->getIncomingBlock(Incoming);
        if (Val == Def)
          IncomingBB = BB;
      }
      assert(IncomingBB != NULL);
      ContextRestoreLocation = IncomingBB->getTerminator();
    }

    if(WIH == WorkitemHandlerType::FIBER)
      RematCandidate = false;

    llvm::Value *RematerializedValue = nullptr;
    if (RematCandidate) {
      if (isa<AllocaInst>(Def))
        RematerializedValue = tryToRematerialize(
            ContextRestoreLocation, InitializerStore->getValueOperand(),
            Def->getName().str());
      else
        RematerializedValue = tryToRematerialize(ContextRestoreLocation, Def,
                                                 Def->getName().str());
    }

    if (RematerializedValue != nullptr) {
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "#### successful rematerialization:\n";
      RematerializedValue->dump();
#endif
      if (isa<AllocaInst>(Def)) {
        if (StoreInst *Store = dyn_cast<StoreInst>(UserI)) {
          // The original store could be left intact, but then we'd need to
          // figure out the materialization-ability beforehand.
          Store->setOperand(0, RematerializedValue);
        } else if (LoadInst *Load = dyn_cast<LoadInst>(UserI)) {
          // We can get rid of the alloca load altogether and use the
          // rematerialized value directly.
          UserI->replaceAllUsesWith(RematerializedValue);
#ifdef DEBUG_WORK_ITEM_LOOPS
          std::cerr << "#### alloca load was converted to a remat value:"
                    << std::endl;
          UserI->dump();
          RematerializedValue->dump();
#endif
        } else if (UserI->isLifetimeStartOrEnd()) {
          // We can leave the original lifetime marker for the alloca as is.
        } else {
          llvm_unreachable("Unexpected alloca usage.");
        }
      } else {
        UserI->replaceUsesOfWith(Def, RematerializedValue);
#ifdef DEBUG_WORK_ITEM_LOOPS
        std::cerr << "#### the user was converted to a remat value:"
                  << std::endl;
        UserI->dump();
#endif
      }
    } else {
      // Unable to rematerialize the value.
      // Allocate a context data array for the variable.
      if (ContextArrayAlloca == nullptr) {
        ContextArrayAlloca = getContextArray(Def, PaddingAdded);

        if (WIH != WorkitemHandlerType::FIBER) {
          ParallelRegion *Region = regionOfBlock(Def->getParent());
          assert(
              "Adding context save outside any region produces illegal code." &&
              Region != NULL);
          addContextSave(Def, ContextArrayAlloca, Region);
        } else {
          addContextSave(Def, ContextArrayAlloca, nullptr);
        }
      }

      llvm::Value *ContextArrayLoad = addContextRestore(
          UserI, ContextArrayAlloca, Def->getType(), PaddingAdded,
          ContextRestoreLocation, isa<AllocaInst>(Def));

      UserI->replaceUsesOfWith(Def, ContextArrayLoad);

#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "#### the user was converted to a context load:"
                << std::endl;
      UserI->dump();
#endif
    }
  }
}

/// Returns the instruction in the entry block which computes the global
/// size for the given \param Dim.
llvm::Instruction *WorkitemHandler::getGlobalSize(int Dim) {
  llvm::Instruction *GSize = GlobalSizes[Dim];
  if (GSize != nullptr)
    return GSize;

  GlobalVariable *LocalSize = cast<GlobalVariable>(LocalSizeGlobals[Dim]);
  GlobalVariable *GroupCount = cast<GlobalVariable>(M->getOrInsertGlobal(
      std::string("_num_groups_") + (char)('x' + Dim), ST));

  CreateBuilder(Builder, K->getEntryBlock());

  GSize = cast<llvm::Instruction>(
      Builder.CreateBinOp(Instruction::Mul, Builder.CreateLoad(ST, LocalSize),
                          Builder.CreateLoad(ST, GroupCount),
                          std::string("_global_size_") + (char)('x' + Dim)));
  GlobalSizes[Dim] = GSize;
  return GSize;
}

/// Returns the instruction in the entry block which computes the "base" for
/// the global id which has all components except the local id offset included.
llvm::Instruction *WorkitemHandler::getGlobalIdOrigin(int Dim) {
  llvm::Instruction *Origin = GlobalIdOrigins[Dim];
  if (Origin != nullptr)
    return Origin;

  GlobalVariable *LocalSize = cast<GlobalVariable>(M->getOrInsertGlobal(
      std::string("_local_size_") + (char)('x' + Dim), ST));
  GlobalVariable *GlobalOffset = cast<GlobalVariable>(M->getOrInsertGlobal(
      std::string("_global_offset_") + (char)('x' + Dim), ST));
  GlobalVariable *GroupId = cast<GlobalVariable>(
      M->getOrInsertGlobal(std::string("_group_id_") + (char)('x' + Dim), ST));

  assert(LocalSize != nullptr);
  assert(GlobalOffset != nullptr);
  assert(GroupId != nullptr);

  CreateBuilder(Builder, K->getEntryBlock());

  Origin = cast<llvm::Instruction>(
      Builder.CreateBinOp(Instruction::Mul, Builder.CreateLoad(ST, LocalSize),
                          Builder.CreateLoad(ST, GroupId)));

  Origin = cast<llvm::Instruction>(Builder.CreateBinOp(
      Instruction::Add, Builder.CreateLoad(ST, GlobalOffset), Origin));

  GlobalIdOrigins[Dim] = Origin;

  llvm::GlobalVariable *GlobalId =
      cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME(Dim), ST));

  // Initialize the global id to the first value just in case we won't create
  // a loop for a 1-sized dimensions which would create the monotonically
  // incrementing GID.
  Builder.CreateStore(Origin, GlobalId);

  return Origin;
}

/**
 * Scans for usages of global id and replaces with global_id_base + local_id.
 *
 * This should be called for WorkitemHandlers that do not produce the global
 * id within the handler like WILoops does.
 */
void WorkitemHandler::GenerateGlobalIdComputation() {
  for (Function::iterator FI = K->begin(), FE = K->end(); FI != FE; ++FI) {
    for (BasicBlock::iterator II = FI->begin(), IE = FI->end(); II != IE;) {
      llvm::LoadInst *GIdLoad = dyn_cast<llvm::LoadInst>(II);
      ++II;
      if (GIdLoad == NULL)
        continue;

      for (int Dim = 0; Dim < 3; ++Dim) {
        GlobalVariable *GlobalId = M->getGlobalVariable(GID_G_NAME(Dim));
        if (GIdLoad->getOperand(0) != GlobalId) {
          continue;
        }
        IRBuilder<> FBuilder(GIdLoad);

        Instruction *LocalId =
            FBuilder.CreateLoad(ST, M->getGlobalVariable(LID_G_NAME(Dim)));
        Instruction *GlobalIdOrigin = getGlobalIdOrigin(Dim);

        Instruction *GidStore = FBuilder.CreateStore(
            FBuilder.CreateAdd(GlobalIdOrigin, LocalId), GlobalId);

        break;
      }
    }
  }
}

// this must be at least the alignment of largest OpenCL type (= 128 bytes)
#define CONTEXT_ARRAY_ALIGN MAX_EXTENDED_ALIGNMENT

/// Creates a well aligned and padded context array for the given value.
///
/// This is not entirely trivial to get right since we want to align the
/// innermost dimension with natural alignment in order to enable vectorized
/// accesses to intra-kernel arrays from the different work-items.
/// In the case of unaligned kernel arrays we have to add padding to make
/// each WI's array nicely aligned.
///
/// \param Instr the original per work-item instruction.
/// \param Before the instruction before which to create the alloca.
/// \param Name for the context array.
/// \param PaddingAdded set to true in case padding was added to align the
/// arrayified object.
llvm::AllocaInst *WorkitemHandler::createAlignedAndPaddedContextAlloca(
    llvm::Instruction *Inst, llvm::Instruction *Before, const std::string &Name,
    bool &PaddingAdded) {

  PaddingAdded = false;
  BasicBlock &BB = Inst->getParent()->getParent()->getEntryBlock();
  IRBuilder<> Builder(Before);
  Function *FF = Inst->getParent()->getParent();
  Module *M = Inst->getParent()->getParent()->getParent();
  const llvm::DataLayout &Layout = M->getDataLayout();
  DICompileUnit *CU = nullptr;
  std::unique_ptr<DIBuilder> DB;
  if (M->debug_compile_units_begin() != M->debug_compile_units_end()) {
    CU = *M->debug_compile_units_begin();
    DB = std::unique_ptr<DIBuilder>{new DIBuilder(*M, true, CU)};
  }

  // find the original debug metadata corresponding to the variable
  Value *DebugVal = nullptr;
  IntrinsicInst *DebugCall = nullptr;
  if (CU != nullptr) {
    for (BasicBlock &BB : (*FF)) {
      for (Instruction &I : BB) {
        IntrinsicInst *CI = dyn_cast<IntrinsicInst>(&I);
        if (CI && (CI->getIntrinsicID() == llvm::Intrinsic::dbg_declare)) {
          Metadata *Meta =
              cast<MetadataAsValue>(CI->getOperand(0))->getMetadata();
          if (isa<ValueAsMetadata>(Meta)) {
            Value *V = cast<ValueAsMetadata>(Meta)->getValue();
            if (Inst == V) {
              DebugVal = V;
              DebugCall = CI;
              break;
            }
          }
        }
      }
    }
  }

#ifdef DEBUG_DEBUG_DATA_GENERATION
  if (DebugVal && DebugCall) {
    std::cerr << "### DI INTRIN: \n";
    DebugCall->dump();
    std::cerr << "### DI VALUE:  \n";
    DebugVal->dump();
  }
#endif

  llvm::Type *ElementType = nullptr;
  Type *AllocType = nullptr;

  if (AllocaInst *SrcAlloca = dyn_cast<AllocaInst>(Inst)) {
    // If the variable to be context saved was itself an alloca, create one
    // big alloca that stores the data of all the work-items and directly
    // return pointers to that array. This enables moving all the allocas to
    // the entry node without breaking the parallel loop. Otherwise we would
    // need to rely on a dynamic alloca to allocate unique stack space to all
    // the work-items when its wiloop iteration is executed.
    ElementType = SrcAlloca->getAllocatedType();
    AllocType = ElementType;

    uint64_t Alignment = SrcAlloca->getAlign().value();
    uint64_t StoreSize = Layout.getTypeStoreSize(SrcAlloca->getAllocatedType());

    if ((Alignment > 1) && (StoreSize & (Alignment - 1))) {
      uint64_t AlignedSize = (StoreSize & (~(Alignment - 1))) + Alignment;
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### unaligned type found: padding " << StoreSize << " to "
                << AlignedSize << "\n";
#endif
      assert(AlignedSize > StoreSize);
      uint64_t RequiredExtraBytes = AlignedSize - StoreSize;

      // n-dim context array: In case the elementType itself is an array or
      // a struct, we must take into account it could be alloca-ed with
      // alignment and loads or stores might use vectorized instructions
      // expecting proper alignment.
      // Because of that, we cannot simply allocate x*y*z*(size), but must
      // pad the inner row to ensure the alignment to the next element.
      if (isa<ArrayType>(ElementType)) {

        ArrayType *StructPadding = ArrayType::get(
            Type::getInt8Ty(M->getContext()), RequiredExtraBytes);

        std::vector<Type *> PaddedStructElements;
        PaddedStructElements.push_back(ElementType);
        PaddedStructElements.push_back(StructPadding);
        const ArrayRef<Type *> NewStructElements(PaddedStructElements);
        AllocType = StructType::get(M->getContext(), NewStructElements, true);
        PaddingAdded = true;
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);

      } else if (isa<StructType>(ElementType)) {
        StructType *OldStruct = dyn_cast<StructType>(ElementType);

        ArrayType *StructPadding = ArrayType::get(
            Type::getInt8Ty(M->getContext()), RequiredExtraBytes);
        std::vector<Type *> PaddedStructElements;
        for (size_t j = 0; j < OldStruct->getNumElements(); j++)
          PaddedStructElements.push_back(OldStruct->getElementType(j));
        PaddedStructElements.push_back(StructPadding);
        PaddingAdded = true;
        const ArrayRef<Type *> NewStructElements(PaddedStructElements);
        AllocType = StructType::get(OldStruct->getContext(), NewStructElements,
                                    OldStruct->isPacked());
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);
      }
    }
  } else {
    ElementType = Inst->getType();
    AllocType = ElementType;
  }

  llvm::AllocaInst *Alloca = nullptr;
  if (WGDynamicLocalSize) {
    GlobalVariable *LocalSize;
    LoadInst *LocalSizeLoad[3];
    for (int i = 0; i < 3; ++i) {
      std::string Name = LS_G_NAME(i);
      LocalSize = cast<GlobalVariable>(M->getOrInsertGlobal(Name, ST));
      LocalSizeLoad[i] = Builder.CreateLoad(ST, LocalSize);
    }

    Value *LocalXTimesY = Builder.CreateBinOp(
        Instruction::Mul, LocalSizeLoad[0], LocalSizeLoad[1], "tmp");
    Value *NumberOfWorkItems = Builder.CreateBinOp(
        Instruction::Mul, LocalXTimesY, LocalSizeLoad[2], "num_wi");

    Alloca = Builder.CreateAlloca(AllocType, NumberOfWorkItems, Name);
  } else {
    llvm::Type *ContextArrayType = ArrayType::get(
        ArrayType::get(ArrayType::get(AllocType, WGLocalSizeX), WGLocalSizeY),
        WGLocalSizeZ);
    Alloca = Builder.CreateAlloca(ContextArrayType, nullptr, Name);
  }

  // Generously align the context arrays to enable wide vector accesses to them.
  // Also at least LLVM 3.3 produced illegal code at least for a Core i5 when
  // aligned only at the element size.
  Alloca->setAlignment(llvm::Align(CONTEXT_ARRAY_ALIGN));

  if (DebugVal && DebugCall && !WGDynamicLocalSize) {

    llvm::SmallVector<llvm::Metadata *, 4> Subscripts;
    Subscripts.push_back(DB->getOrCreateSubrange(0, WGLocalSizeZ));
    Subscripts.push_back(DB->getOrCreateSubrange(0, WGLocalSizeY));
    Subscripts.push_back(DB->getOrCreateSubrange(0, WGLocalSizeX));
    llvm::DINodeArray SubscriptArray = DB->getOrCreateArray(Subscripts);

    size_t SizeBits;
    SizeBits = Alloca
                   ->getAllocationSizeInBits(M->getDataLayout())
                   .value_or(TypeSize(0, false))
                   .getFixedValue();

    assert(SizeBits != 0);

    // if (size == 0) WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ * 8 *
    // Alloca->getAllocatedType()->getScalarSizeInBits();
    size_t AlignBits = Alloca->getAlign().value() * 8;

    Metadata *VariableDebugMeta =
        cast<MetadataAsValue>(DebugCall->getOperand(1))->getMetadata();
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### VariableDebugMeta :  ";
    VariableDebugMeta->dump();
    std::cerr << "### sizeBits :  " << SizeBits << "  alignBits: " << AlignBits
              << "\n";
#endif

    DILocalVariable *LocalVar = dyn_cast<DILocalVariable>(VariableDebugMeta);
    assert(LocalVar);
    if (LocalVar) {

      DICompositeType *CT = DB->createArrayType(
          SizeBits, AlignBits, LocalVar->getType(), SubscriptArray);

#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### DICompositeType:\n";
      CT->dump();
#endif
      DILocalVariable *NewLocalVar = DB->createAutoVariable(
          LocalVar->getScope(), LocalVar->getName(), LocalVar->getFile(),
          LocalVar->getLine(), CT, false, LocalVar->getFlags());

      Metadata *NewMeta = ValueAsMetadata::get(Alloca);
      DebugCall->setOperand(0, MetadataAsValue::get(M->getContext(), NewMeta));

      MetadataAsValue *NewLV =
          MetadataAsValue::get(M->getContext(), NewLocalVar);
      DebugCall->setOperand(1, NewLV);

      DebugCall->removeFromParent();
      DebugCall->insertAfter(Alloca);
    }
  }
  return Alloca;
}

/// Creates a GEP to a context array in the currently handled parallel region.
///
/// \param CtxArrayAlloca the context array alloca to address.
/// \param Before the instruction in the parallel region to insert the GEP
/// before.
/// \param AlignPading If this is set to true, the CArrayAlloca's innermost
/// dimension has the alignment padding which should be taken in account in
/// addressing the array.
llvm::GetElementPtrInst *
WorkitemHandler::createContextArrayGEP(llvm::AllocaInst *CtxArrayAlloca,
                                       llvm::Instruction *Before,
                                       bool AlignPadding) {
  std::vector<llvm::Value *> GEPArgs;
  IRBuilder<> Builder(Before);

  if (WGDynamicLocalSize) {
    if (WIH == WorkitemHandlerType::FIBER)
      GEPArgs.push_back(getLinearWiIndex(Builder, M, nullptr, WIH));
    else
      GEPArgs.push_back(getLinearWIIndexInRegion(Before));
  } else {
    GEPArgs.push_back(ConstantInt::get(ST, 0));
    GEPArgs.push_back(getLocalIdInRegion(Before, 2));
    GEPArgs.push_back(getLocalIdInRegion(Before, 1));
    GEPArgs.push_back(getLocalIdInRegion(Before, 0));
  }

  if (AlignPadding)
    GEPArgs.push_back(
        ConstantInt::get(Type::getInt32Ty(CtxArrayAlloca->getContext()), 0));

  IRBuilder<> Builder(Before);
  llvm::GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Builder.CreateGEP(
      CtxArrayAlloca->getAllocatedType(), CtxArrayAlloca, GEPArgs));
  assert(GEP != nullptr);

  return GEP;
}

// TO CLEAN: Refactor into getLinearWIIndexInRegion.
llvm::Value *WorkitemHandler::getLinearWiIndex(llvm::IRBuilder<> &Builder,
                                               llvm::Module *M,
                                               ParallelRegion *Region,
                                               WorkitemHandlerType WIH) {
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
  Value *ZPart;
  Value *YPart;

  Value *LocalSizeXTimesY =
      Builder.CreateBinOp(Instruction::Mul, LoadX, LoadY, "ls_xy");

  Value *Result;
  if (WIH == WorkitemHandlerType::FIBER) {
    llvm::LoadInst *LoadXId = Builder.CreateLoad(ST, LocalIdGlobals[0], "id_x");
    llvm::LoadInst *LoadYId = Builder.CreateLoad(ST, LocalIdGlobals[1], "id_y");
    llvm::LoadInst *LoadZId = Builder.CreateLoad(ST, LocalIdGlobals[2], "id_z");
    ZPart =
        Builder.CreateBinOp(Instruction::Mul, LocalSizeXTimesY, LoadZId, "tmp");
    YPart = Builder.CreateBinOp(Instruction::Mul, LoadX, LoadYId, "ls_x_y");
    Value *ZYSum =
        Builder.CreateBinOp(Instruction::Add, ZPart, YPart, "zy_sum");
    Result =
        Builder.CreateBinOp(Instruction::Add, ZYSum, LoadXId, "linear_xyz_idx");
  } else {
    ZPart =
        Builder.CreateBinOp(Instruction::Mul, LocalSizeXTimesY,
                            Region->getOrCreateIDLoad(LID_G_NAME(2)), "tmp");
    YPart =
        Builder.CreateBinOp(Instruction::Mul, LoadX,
                            Region->getOrCreateIDLoad(LID_G_NAME(1)), "ls_x_y");
    Value *ZYSum =
        Builder.CreateBinOp(Instruction::Add, ZPart, YPart, "zy_sum");
    Result = Builder.CreateBinOp(Instruction::Add, ZYSum,
                                 Region->getOrCreateIDLoad(LID_G_NAME(0)),
                                 "linear_xyz_idx");
  }
  return Result;
}

/// Checks if it's OK to mark the work-item loops in the currently processed
/// kernel as parallel loops.
///
/// Currently the only known reason to not mark them is to workaround a VPlan
/// crash that occurs with volatile memory accesses inside the parallel
/// WI-loops. Thus, we return false only in case of using LLVM 17+,
/// where the issue is producible, and if the loop contains volatile accesses.
/// The PoCL issue: https://github.com/pocl/pocl/issues/1556
///
/// We could make this Loop/PRegion-specific, but it seems not worth the effort
/// at this point as WorkitemLoops doesn't have a ready loop at hand when it
/// needs to annotate it, and luckily volatile usage is not common and ruins
/// the perf anyhow.
///
/// \return False in case we should _not_ add the parallel loop metadata,
/// even though the loop is known to be parallel.
bool WorkitemHandler::canAnnotateParallelLoops() {
  for (auto &BB : *K) {
    for (auto &I : BB) {
      if (I.isVolatile())
        return false;
    }
  }
  return true;
}

/// Returns the instruction in the entry block of the currently handled kernel
/// which computes the total size of work-items in the work-group.
///
/// If it doesn't exist, creates and adds it to the end of the entry block.
llvm::Instruction *WorkitemHandler::getWorkGroupSizeInstr() {

  if (WGSizeInstr != nullptr)
    return WGSizeInstr;

  IRBuilder<> Builder(K->getEntryBlock().getTerminator());

  llvm::Module *M = K->getParent();
  GlobalVariable *GV = M->getGlobalVariable("_local_size_x");
  if (GV != NULL) {
    WGSizeInstr = Builder.CreateLoad(ST, GV);
  }

  GV = M->getGlobalVariable("_local_size_y");
  if (GV != NULL) {
    WGSizeInstr = cast<llvm::Instruction>(Builder.CreateBinOp(
        Instruction::Mul, Builder.CreateLoad(ST, GV), WGSizeInstr));
  }

  GV = M->getGlobalVariable("_local_size_z");
  if (GV != NULL) {
    WGSizeInstr = cast<llvm::Instruction>(Builder.CreateBinOp(
        Instruction::Mul, Builder.CreateLoad(ST, GV), WGSizeInstr));
  }

  return WGSizeInstr;
}

/// Converts calls to the __pocl_{work_group,local_mem}_alloca() pseudo
/// functions to allocas in the current kernel.
///
/// These compiler-expanded functions are used to allocate temporary
/// storage for (sub)group-level built-in implementation. Search for their
/// usage in the bitcode library for examples. They are converted to
/// allocas and placed in the entry of the function.
bool WorkitemHandler::handleLocalMemAllocas() {

  std::vector<CallInst *> InstructionsToFix;

  for (BasicBlock &BB : *K) {
    for (Instruction &I : BB) {

      if (!isa<CallInst>(I))
        continue;
      CallInst &Call = cast<CallInst>(I);

      if (Call.getCalledFunction() == nullptr ||
          (Call.getCalledFunction() != LocalMemAllocaFuncDecl &&
           Call.getCalledFunction() != WorkGroupAllocaFuncDecl))
        continue;
      InstructionsToFix.push_back(&Call);
    }
  }

  bool Changed = false;
  for (CallInst *Call : InstructionsToFix) {
    Value *Size = Call->getArgOperand(0);
    Align Alignment =
        cast<ConstantInt>(Call->getArgOperand(1))->getAlignValue();

    // Push the alloca to the pure uniform entry block so it's called only once
    // per WG launch.
    IRBuilder<> Builder(K->getEntryBlock().getTerminator());

    if (Call->getCalledFunction() == WorkGroupAllocaFuncDecl) {
      Value *ExtraSize = Call->getArgOperand(2);
      Instruction *WGSize = getWorkGroupSizeInstr();
      Size = Builder.CreateBinOp(Instruction::Mul, WGSize, Size);
      Size = Builder.CreateBinOp(Instruction::Add, Size, ExtraSize);
    }
    AllocaInst *WGAlloca = new AllocaInst(
        llvm::Type::getInt8Ty(Call->getContext()), 0, Size, Alignment,
        "__pocl_wg_alloca", Inst2InsertPt(K->getEntryBlock().getTerminator()));
    Call->replaceAllUsesWith(WGAlloca);
    Call->eraseFromParent();

    // Also move the variable the allocation result is saved to.
    for (Instruction::use_iterator UI = WGAlloca->use_begin(),
                                   UE = WGAlloca->use_end();
         UI != UE; ++UI) {
      llvm::StoreInst *InitializerStore =
          dyn_cast_or_null<StoreInst>(UI->getUser());
      if (InitializerStore == nullptr ||
          InitializerStore->getValueOperand() != WGAlloca)
        continue;

      llvm::AllocaInst *TempVariableAlloca =
          dyn_cast_or_null<AllocaInst>(InitializerStore->getPointerOperand());
      if (TempVariableAlloca == nullptr)
        continue;
      TempVariableAlloca->moveAfter(WGAlloca);
      InitializerStore->moveAfter(TempVariableAlloca);
    }

    Changed = true;
  }
  return Changed;
}

/// Converts some of the work-item function calls to loads from the pseudo
/// variables or precomputed values from within the kernel function.
///
/// Currently handles get_global_size(), get_local_id(), get_global_id(),
/// get_group_id() and get_global_offset() calls. Expands the calls next to
/// their users for easier analysis.
bool WorkitemHandler::handleWorkitemFunctions() {
  std::set<llvm::Instruction *> InstrsToDelete;

  for (Function::iterator BBI = K->begin(), BBE = K->end(); BBI != BBE; ++BBI) {
    llvm::BasicBlock &BB = *BBI;
    for (llvm::BasicBlock::iterator II = BB.begin(); II != BB.end(); ++II) {
      llvm::Instruction *Instr = &*II;
      llvm::CallInst *Call = dyn_cast<llvm::CallInst>(Instr);
      if (Call == nullptr)
        continue;

      if (isCompilerExpandableWIFunctionCall(*Call)) {
        auto Callee = Call->getCalledFunction();
        int Dim =
            cast<llvm::ConstantInt>(Call->getArgOperand(0))->getZExtValue();

        for (Instruction::use_iterator UI = Call->use_begin(),
                                       UE = Call->use_end();
             UI != UE;) {
          llvm::Instruction *User = cast<Instruction>(UI->getUser());
          llvm::Instruction *InsertBefore = User;
          if (isa<PHINode>(InsertBefore))
            InsertBefore = Call;
          IRBuilder<> Builder(InsertBefore);
          llvm::Value *Replacement = nullptr;
          if (Dim >= 3) {
            if (Callee->getName() == GID_BUILTIN_NAME ||
                Callee->getName() == GROUP_ID_BUILTIN_NAME ||
                Callee->getName() == LID_BUILTIN_NAME ||
                Callee->getName() == GOFF_BUILTIN_NAME ||
                Callee->getName() == GLID_BUILTIN_NAME ||
                Callee->getName() == LLID_BUILTIN_NAME)
              Replacement = ConstantInt::get(Call->getType(), 0);
            else
              Replacement = ConstantInt::get(Call->getType(), 1);
          } else if (Callee->getName() == GID_BUILTIN_NAME)
            Replacement = Builder.CreateLoad(ST, GlobalIdGlobals[Dim]);
          else if (Callee->getName() == GROUP_ID_BUILTIN_NAME)
            Replacement = Builder.CreateLoad(ST, GroupIdGlobals[Dim]);
          else if (Callee->getName() == NGROUPS_BUILTIN_NAME)
            Replacement = Builder.CreateLoad(ST, NumGroupsGlobals[Dim]);
          else if (Callee->getName() == LS_BUILTIN_NAME)
            Replacement = Builder.CreateLoad(ST, LocalSizeGlobals[Dim]);
          else if (Callee->getName() == LID_BUILTIN_NAME)
            Replacement = getLocalIdInRegion(InsertBefore, Dim);
          else if (Callee->getName() == GS_BUILTIN_NAME)
            Replacement = getGlobalSize(Dim);
          else if (Callee->getName() == GOFF_BUILTIN_NAME)
            Replacement = Builder.CreateLoad(ST, GlobalOffsetGlobals[Dim]);
          User->replaceUsesOfWith(Call, Replacement);
          UI = Call->use_begin();
          UE = Call->use_end();
        }
        InstrsToDelete.insert(Call);
        continue;
      }
    }
  }
  for (auto I : InstrsToDelete)
    I->eraseFromParent();

  return InstrsToDelete.size() > 0;
}

bool WorkitemHandler::removeBarrierCalls() {
  std::set<Instruction *> BarriersToRemove;
  for (Function::iterator I = K->begin(), E = K->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (llvm::isa<Barrier>(Instr)) {
        BarriersToRemove.insert(Instr);
      }
    }
  }

  bool Changed = !BarriersToRemove.empty();
  for (auto B : BarriersToRemove) {
    B->eraseFromParent();
  }

  return Changed;
}

} // namespace pocl
