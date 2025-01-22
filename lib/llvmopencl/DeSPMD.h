//=- DeSPMD.h - Compile and optimize "GPU kernels" for "CPUs" --*- C++ -*-=//
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
// The pass is designed to work with minimal preprocessing, thus should work on
// -O0 input. It performs minimal unrelated changes to the input enable a
// better debugging experience with the CFG transformations applied.
//
// NOTE: This pass is being prepared for LLVM upstreaming, thus doesn't
// follow the PoCL namespaces and conventions etc., but adheres strictly to
// those of LLVM Project's.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_DESPMD_H
#define LLVM_TRANSFORMS_DESPMD_H

#include "llvm/IR/PassManager.h"
#include <llvm/Passes/PassBuilder.h>
#include "llvm/Support/CommandLine.h"

namespace llvm {

class Function;

/// The DeSPMD Pass.
struct DeSPMDPass : public PassInfoMixin<DeSPMDPass> {
private:
public:
  DeSPMDPass();

  static void registerWithPB(llvm::PassBuilder &B);
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif

