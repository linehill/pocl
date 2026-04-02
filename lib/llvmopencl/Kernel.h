// Class for kernels, a special kind of function.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
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

#ifndef POCL_KERNEL_H
#define POCL_KERNEL_H

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>
POP_COMPILER_DIAGS
#include "ParallelRegion.h"
#include "VariableUniformityAnalysisResult.hh"

// Barrier pairing for a parallel region.
struct BarrierPairing {
    // Entry barrier of parallel region.
    llvm::BasicBlock *EntryBB;
    // Exit barriers of parallel region.
    std::vector<llvm::BasicBlock *> ExitBBs;
};

namespace pocl {

  class Kernel : public llvm::Function {
  public:
    void getRegionExitBlocks(llvm::SmallVectorImpl<llvm::BasicBlock *> &B);

    /// Creates and returns a parallel region between barrier blocks start and
    /// end.
    ParallelRegion *CreateParallelRegionBetween(
        llvm::BasicBlock *start, llvm::BasicBlock *end,
        pocl::ParallelRegion::ParallelRegionVector *regions,
        VariableUniformityAnalysisResult &VUA);

    void
    getParallelRegions(llvm::LoopInfo &LI,
                       ParallelRegion::ParallelRegionVector *ParallelRegions,
                       VariableUniformityAnalysisResult &VUA);

    void addLocalSizeInitCode(size_t LocalSizeX, size_t LocalSizeY,
                              size_t LocalSizeZ);
    /// Finds and creates a parallel region mapping for the barriers. Stores
    /// barriers in a map such that each key value will act as an entry barrier
    /// for a parallel region. Each barrier block in the value will act as an
    /// exit barrier for a parallel region that has the key as an entry
    /// barrier. { A : [B,C], D : [E]}, means that parallel regions bounded by
    /// A->B, A->C, D->E will be formed.
    void getRegionBarrierMapping(
        std::vector<BarrierPairing> &BarrierPairing);

    static bool isKernel(const llvm::Function &F);

    static bool classof(const Kernel *) { return true; }
    // We assume any function can be a kernel. This could be used
    // to check for metadata but would need to be overrideable somehow
    // to honor the forced kernel name(s) parameter in command line.
    static bool classof(const llvm::Function *) { return true; }

  private:
    // Note: Since this class is only using llvm's inheritance mechanism for
    // creating a convenience class, we never allocate storage for it
    // separately, thus should not allocate any additional data here.
  };

}

#endif
