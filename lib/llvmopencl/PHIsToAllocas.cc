// LLVM function to convert all PHIs to alloca reads.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
//               2024-2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/IR/IRBuilder.h>

#include "Barrier.h"
#include "LLVMUtils.h"
#include "PHIsToAllocas.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemLoops.h"
POP_COMPILER_DIAGS

// #define DEBUG_PHIS_TO_ALLOCAS

// Skip PHIsToAllocas when we are not creating the work item loops,
// as it leads to worse code without benefits for the full replication method.
// Note: re-enabling this causes workgroup/cond_barriers_in_for_cbs to fail
//#define CBS_NO_PHIS_IN_SPLIT

#include <iostream>

namespace pocl {

using namespace llvm;

bool convertPHIsToAllocaAccesses(llvm::Function &F) {

  std::vector<PHINode *> PHIs;
  for (auto &BB : F) {
    for (auto &I : BB) {
      PHINode *PHI = dyn_cast_or_null<PHINode>(&I);
      if (PHI != nullptr)
        PHIs.push_back(PHI);
    }
  }

  for (PHINode *Phi : PHIs) {
    std::string AllocaName = std::string(Phi->getName().str()) + ".ex_phi";

    llvm::Function *Function = Phi->getParent()->getParent();

    IRBuilder<> Builder(&*(Function->getEntryBlock().getFirstInsertionPt()));

    llvm::Instruction *AllocaI =
        Builder.CreateAlloca(Phi->getType(), 0, AllocaName);

    for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues();
         ++Incoming) {
      Value *Val = Phi->getIncomingValue(Incoming);
      BasicBlock *IncomingBB = Phi->getIncomingBlock(Incoming);
      Builder.SetInsertPoint(IncomingBB->getTerminator());
      llvm::Instruction *Store = Builder.CreateStore(Val, AllocaI);
    }
    Builder.SetInsertPoint(Phi);

    llvm::Instruction *LoadedValue =
        Builder.CreateLoad(Phi->getType(), AllocaI);
    Phi->replaceAllUsesWith(LoadedValue);
    Phi->eraseFromParent();
  }
  return PHIs.size() > 0;
}

} // namespace pocl
