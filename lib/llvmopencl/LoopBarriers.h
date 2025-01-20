// Addition of implicit barriers to isolate loops for clean and correct
// parallel regions.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2025 Pekka Jääskeläinen / Intel Finland Oy
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

#ifndef POCL_LOOP_BARRIERS_H
#define POCL_LOOP_BARRIERS_H

#include "config.h"

namespace llvm {
class Function;
class LoopInfo;
} // namespace llvm

namespace pocl {

class VariableUniformityAnalysisResult;

/// Adds implicit barriers to isolate loops to produce clean and semantically
/// correct parallel region control flow regions.
///
/// \return True in case modified the function.
bool addLoopConstructIsolationBarriers(llvm::Function &F, llvm::LoopInfo &LI,
                                       VariableUniformityAnalysisResult &VUA,
                                       llvm::DominatorTree &DT);

} // namespace pocl

#endif
