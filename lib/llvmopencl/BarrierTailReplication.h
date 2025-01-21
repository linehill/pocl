// LLVM function pass to replicate barrier tails (successors to barriers).
//
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012-2019 Pekka Jääskeläinen
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

#ifndef POCL_BARRIER_TAIL_REPLICATION
#define POCL_BARRIER_TAIL_REPLICATION

#include "config.h"

namespace llvm {
class Function;
class DominatorTree;
class LoopInfo;
class PostDominatorTree;
} // namespace llvm

namespace pocl {

class VariableUniformityAnalysisResult;

/// Ensure that the function has clean single-entry-single-exit parallel
/// regions in the paths towards exit nodes.
///
/// This is done by replicating the tails of control flow paths with side
/// entries in the middle of a parallel region.
bool replicateBarrierPathTails(llvm::Function &F, llvm::LoopInfo &LI,
                               llvm::DominatorTree &DT,
                               llvm::PostDominatorTree &PDT,
                               VariableUniformityAnalysisResult &VUA);

} // namespace pocl

#endif
