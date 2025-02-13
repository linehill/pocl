
// Header for fiber work-group method.
//
// Copyright (c) 2025 Tapio Nevalainen / Tampere University
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

#ifndef POCL_FIBER_H
#define POCL_FIBER_H

#include "config.h"

#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "llvm/Analysis/PostDominators.h"

namespace llvm {
class DominatorTreeAnalysis;
class Function;
class LoopAnalysis;
class PostDominatorTreeAnalysis;
class VariableUniformityAnalysisResult;
} // namespace llvm

namespace pocl {

/// Fallback method to handle all types of subgroup configurations and
/// barrier-usage corner cases in general.
///
/// Modifies the LLVM IR so that each workgroup/subgroup barrier call
/// is registered and the next workitem is scheduled. Registering barriers and
/// scheduling workitems are handled by separate scheduler functionality
/// in fiber_scheduler.c, which is exposed through the kernel library.
///
/// Each barrier call is followed by a jump to the 'Scheduler' block, which
/// determines the next block for the newly scheduled workitem, and jumps
/// there. The logic is very robust and does not require pre-modifications.
bool addFiberExecution(llvm::Function &F, llvm::DominatorTree &DT,
                       llvm::PostDominatorTree &PDT, llvm::LoopInfo &LI,
                       VariableUniformityAnalysisResult &VUA);

} // namespace pocl

#endif
