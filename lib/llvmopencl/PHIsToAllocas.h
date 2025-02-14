// Header for PHIsToAllocas functionality.
//
// Copyright (c) 2012 Pekka Jääskeläinen / TUT
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

#ifndef POCL_PHIS_TO_ALLOCAS_H
#define POCL_PHIS_TO_ALLOCAS_H

namespace llvm {
class Function;
}

namespace pocl {

class VariableUniformityAnalysisResult;

/// Converts PHIs to an alloca and the sources to writes.
///
/// The control flow transformations DeSPMD performs do not handle PHI nodes.
/// When we compile from sources, the input is unoptimized and not in SSA form,
/// thus there should not be PHI nodes either, but when the input originates
/// from SPIR-V there could be PHI nodes we should get rid of. Maintains
/// uniformity info in \p VUA that has been produced with the PHIs intact.
bool convertPHIsToAllocaAccesses(llvm::Function &F,
                                 VariableUniformityAnalysisResult &VUA);

} // namespace pocl

#endif
