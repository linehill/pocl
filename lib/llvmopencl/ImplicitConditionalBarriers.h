// Adds implicit barriers to branches where required and seen beneficial.
//
// Copyright (c) 2013 Pekka Jääskeläinen / TUT
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

#ifndef POCL_IMPLICIT_CONDITIONAL_BARRIERS_H
#define POCL_IMPLICIT_CONDITIONAL_BARRIERS_H

#include "config.h"

namespace llvm {
class Function;
class LoopInfo;
class PostDominatorTree;
class DominatorTree;
} // namespace llvm

namespace pocl {

class VariableUniformityAnalysisResult;

/// Adds implicit barriers to branches leading to conditional barriers.
///
/// In essence, it converts the following control flow cases:
///
///      a
///      |
///    .[P].
///    |   |
///    b   c
///   [B]
///
///   to
///
///      a
///      |
///     [B]
///    .[P].
///   [B] [B]
///    |   |
///    a   b
///   [B]
///
/// Legend: [P] is a BB with the predicate.
///         [B] is a barrier.
///          a and b are regular basic blocks.
///
/// We can inject the barrier legally due to the barrier semantics:
/// 'a' or 'b' are entered by all or none of the work-items. Thus,
/// the additional barrier is legal there as well.
///
/// This isolates the branch leading to the barrier, which is then
/// solely controlling the flow leading to the conditional barrier's
/// parallel region.
///
/// \return True in case modified the function.
bool addImplicitBranchBarriers(llvm::Function &F, llvm::LoopInfo &LI,
                               VariableUniformityAnalysisResult &VUA,
                               llvm::PostDominatorTree &PDT,
                               llvm::DominatorTree &DT);
}

#endif
