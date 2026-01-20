//=- DeSPMD.cpp - Compile and optimize "GPU kernels" for "CPUs" --*- C++ -*-=//
//
// (To be) part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DeSPMD is an LLVM pass (to be upstreamed) that converts functions defined in
// "SPMD languages" such as OpenCL C or SPIR-V to a form that can be
// efficiently fine-grain parallelized on "MIMD machines" (typically "CPUs").
//
// The aim of the pass is to be able to output CPU-callable "work-group
// functions" directly from Clang when compiling such kernels.  It works by
// analyzing the kernel and deciding the best strategy to execute it on a
// non-SPMD device.  The following alternatives are considered by it:
//
// 1. WILoops, which creates parallel work-item loops around regions between
//    barriers which are delegated to LLVM vectorizers for vector mapping or
//    VLIW instruction shedulers for ILP extraction.
//
// 2. Fibers [TBD], which is a fallback for complex control flow corner cases
//    which cannot be (efficiently) handled by WILoops.
//
// 3. CBS [TBD] for cases X and Y.
//
//
// NOTE: This pass is being prepared for LLVM upstreaming, thus doesn't
// follow the PoCL namespaces and conventions etc., but strictly those of
// LLVM's. This header comment should be kept (nearly) identical to
// DeSPMD.h's. The model was LoopVectorize.{cpp,h}.
//===----------------------------------------------------------------------===//

#include "DeSPMD.h"

#include "llvm/Analysis/PostDominators.h"
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

// TODO: Move the needed definitions from these PoCL modules to this file.
// LLVM prefers self-contained files for passes (even if they grow large).
#include "CanonicalizeBarriers.h"
#include "DebugHelpers.h"
#include "Fiber.h"
#include "ImplicitConditionalBarriers.h"
#include "ImplicitLoopBarriers.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "LoopBarriers.h"
#include "PHIsToAllocas.h"
#include "SubgroupBarrier.h"
#include "WorkgroupBarrier.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
#include "pocl_llvm_api.h"

// TODO: recheck if we can reuse an existing analysis from LLVM:
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"

#include <iostream>

using namespace llvm;
using namespace pocl;

#define PASS_NAME "despmd"
#define DEBUG_TYPE PASS_NAME
#define PASS_CLASS llvm::DeSPMDPass
#define PASS_DESC "Convert SPMD kernel functions to CPU executable and optimizable ones."

namespace llvm {

#define REFRESH_LOOP_INFO()                                                    \
  do {                                                                         \
    if (Changed) {                                                             \
      DT.recalculate(F);                                                       \
      PDT.recalculate(F);                                                      \
      LI.releaseMemory();                                                      \
      LI.analyze(DT);                                                          \
      LI.verify(DT);                                                           \
    }                                                                          \
  } while (false)

DeSPMDPass::DeSPMDPass() {
}

static bool removeLifetimeMarkers(Function &F) {
  std::set<llvm::Instruction *> InstrsToDelete;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (I.isLifetimeStartOrEnd())
        InstrsToDelete.insert(&I);
    }
  }
  for (auto *I : InstrsToDelete)
    I->eraseFromParent();
  return InstrsToDelete.size() > 0;
}

/// Convert subgroup barriers to workgroup barriers when possible.
static bool convertSGBarriersToWGBarriers(llvm::Function &F,
                                          llvm::LoopInfo &LI) {

  for (llvm::Loop *OuterLoop : LI) {
    auto Loops = OuterLoop->getLoopsInPreorder();
    for (llvm::Loop *L : Loops) {

      // Loop with wg-barrier and sg-barrier implies non-divergence.
      // So, it should be safe to convert sg-barriers to wg-barriers in this
      // case. In nested loops, is it possible that outer loop syncs on wgs, and
      // inner on sgs?
      // TODO: In case of nested loops, use getSubLoops to identify if barriers
      // are on the same level.
      if (WorkgroupBarrier::isLoopWithWGBarrier(*L) &&
          SubgroupBarrier::isLoopWithSGBarrier(*L)) {
        for (llvm::BasicBlock *BB : L->getBlocks()) {
          if (SubgroupBarrier::hasSGBarrier(BB)) {
            switchBarrierGranularity(BB);
          }
        }
      }
    }
  }
  return true;
}

/// Insert and initialize magic global variables for work-item loop bounds if
/// applicable
///
/// These variables define the work-item ranges in each dimension the work-item
/// loops iterate over and the bound may be adjusted during the kernel
/// execution. The insertion is applicable for WorkitemHandlerType::LOOPS method
/// which doesn't require linear work-item loops. When the work-item loop bound
/// variables are present and invariant_wiloop_bounds metadata is not set, the
/// work-item loops must use them for correctness reasons.
///
/// For each dimension there are two bound variable: one for lower bound
/// (inclusive) and other for upper bound (exclusive).
static bool setupKernelEntryWILoopBounds(llvm::Function &F,
                                         WorkitemHandlerType WIH) {
  if (WIH != WorkitemHandlerType::LOOPS)
    return false;

  auto *M = F.getParent();
  auto *Entry = &F.getEntryBlock();
  if (!isPureUniformBlock(Entry)) {
    SplitBlock(Entry, Entry->getFirstInsertionPt());
    Entry = &F.getEntryBlock();
    markAsPureUniformBlock(Entry, "wg-function entry");
    Entry->setName("wg-func-entry");
  }

  for (unsigned Dim = 0; Dim < 3; Dim++) {
    auto *LowerBound = getOrCreateWILoopLowerBoundGV(M, Dim);
    auto *UpperBound = getOrCreateWILoopUpperBoundGV(M, Dim);
    IRBuilder B(Entry, Entry->getFirstInsertionPt());
    Type *BoundTy = LowerBound->getValueType();
    Value *UpperBoundValue = getWorkgroupLocalSize(M, Dim, B.GetInsertPoint());

    B.CreateStore(ConstantInt::get(BoundTy, 0), LowerBound);
    B.CreateStore(UpperBoundValue, UpperBound);
  }

  bool InvariantBounds = !hasCallTo(&F, "__pocl_probe_set_wiloop_bounds");

  // See hasInvariantWILoopBounds() definition for the meaning of the MD.
  setModuleBoolMetadata(M, "invariant_wiloop_bounds", InvariantBounds);

  return true;
}

PreservedAnalyses DeSPMDPass::run(Function &F,
                                  FunctionAnalysisManager &AM) {

  if (!pocl::isKernelToProcess(F))
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

  bool Changed = false;

  WorkitemHandlerType WIH = getWorkitemHandler(F, PDT, LI);

#ifdef RENAME_UNNAMED_BBS
  renameUnnamedBlocks(F);
#endif

  Changed = convertPHIsToAllocaAccesses(F, DT) || Changed;
  REFRESH_LOOP_INFO();

  Changed = setupKernelEntryWILoopBounds(F, WIH) || Changed;
  REFRESH_LOOP_INFO();

  pocl::VariableUniformityAnalysisResult VUA;
  VUA.runOnFunction(F, LI, PDT);

  Changed = canonicalizeBarriers(F, LI, DT) || Changed;
  REFRESH_LOOP_INFO();

  if (WIH != WorkitemHandlerType::FIBER) {
    Changed = convertSGBarriersToWGBarriers(F, LI) || Changed;
    Changed = enforceOuterLoopParIfBeneficial(F, LI, VUA) || Changed;
    Changed = canonicalizeBarriers(F, LI, DT) || Changed;
    Changed = addLoopConstructIsolationBarriers(F, LI, VUA, DT) || Changed;
    REFRESH_LOOP_INFO();

    DT.recalculate(F);
    PDT.recalculate(F);

    Changed = addImplicitBranchBarriers(F, LI, VUA, PDT, DT) || Changed;
    Changed = canonicalizeBarriers(F, LI, DT) || Changed;
    REFRESH_LOOP_INFO();

    DT.recalculate(F);
    PDT.recalculate(F);

    // Run implicit conditional barriers again, since new conditional
    // barriers may have been added during the first pass.
    Changed = addImplicitBranchBarriers(F, LI, VUA, PDT, DT) || Changed;
    REFRESH_LOOP_INFO();
    Changed = canonicalizeBarriers(F, LI, DT) || Changed;
    REFRESH_LOOP_INFO();

    Changed = removeLifetimeMarkers(F);

    // TODO: Run CBS if chosen.
    Changed = addWorkItemLoops(F, DT, PDT, LI, VUA) || Changed;
  } else {
    Changed = addFiberExecution(F, DT, PDT, LI, VUA) || Changed;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace llvm
