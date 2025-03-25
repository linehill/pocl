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

// TODO: Move the needed definitions from these PoCL modules to this file.
// LLVM prefers self-contained files for passes (even if they grow large).
#include "BarrierTailReplication.h"
#include "CanonicalizeBarriers.h"
#include "Fiber.h"
#include "ImplicitConditionalBarriers.h"
#include "ImplicitLoopBarriers.h"
#include "LLVMUtils.h"
#include "LoopBarriers.h"
#include "PHIsToAllocas.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"

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

PreservedAnalyses DeSPMDPass::run(Function &F,
                                  FunctionAnalysisManager &AM) {

  if (!pocl::isKernelToProcess(F))
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &PDT = AM.getResult<PostDominatorTreeAnalysis>(F);
  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);

  bool Changed = false;

  WorkitemHandlerType WIH = getWorkitemHandler();

  Changed = convertPHIsToAllocaAccesses(F) || Changed;
  REFRESH_LOOP_INFO();

  pocl::VariableUniformityAnalysisResult VUA;
  VUA.runOnFunction(F, LI, PDT);

  Changed = canonicalizeBarriers(F) || Changed;
  REFRESH_LOOP_INFO();

  if (WIH != WorkitemHandlerType::FIBER) {
    Changed = enforceOuterLoopParIfBeneficial(F, LI, VUA) || Changed;
    Changed = canonicalizeBarriers(F) || Changed;

    REFRESH_LOOP_INFO();
    Changed = addLoopConstructIsolationBarriers(F, LI, VUA, DT) || Changed;
    REFRESH_LOOP_INFO();

    Changed = addImplicitBranchBarriers(F, LI, VUA, PDT, DT) || Changed;
    Changed = canonicalizeBarriers(F) || Changed;
    REFRESH_LOOP_INFO();

    Changed = replicateBarrierPathTails(F, LI, DT, PDT, VUA) || Changed;
    REFRESH_LOOP_INFO();

    // Run implicit conditional barriers again since BTR might have added new
    // conditional barrier cases that must be handled.
    Changed = addImplicitBranchBarriers(F, LI, VUA, PDT, DT) || Changed;
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
