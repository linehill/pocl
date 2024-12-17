// Implementation for VariableUniformityAnalysis function pass.
//
// Copyright (c) 2023 Michal Babej / Intel Finland Oy
//               2024 Pekka Jääskeläinen / Intel Finland Oy
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

#ifndef POCL_VUA_RESULT_HH
#define POCL_VUA_RESULT_HH

namespace llvm {
class Function;
class LoopInfo;
class PostDominatorTree;
class BasicBlock;
class Value;
class Module;
class Loop;

class PreservedAnalyses;
template <typename, typename...> class AnalysisManager;
} // namespace llvm

#include <map>

namespace pocl {
class VariableUniformityAnalysisResult {

public:
  bool runOnFunction(llvm::Function &F, llvm::LoopInfo &LI,
                     llvm::PostDominatorTree &PDT);
  bool isUniform(llvm::Function *F, llvm::Value *V);
  void setUniform(llvm::Function *F, llvm::Value *V, bool isUniform = true);
  void analyzeBBDivergence(llvm::Function *F, llvm::BasicBlock *BB,
                           llvm::BasicBlock *PreviousUniformBB,
                           llvm::PostDominatorTree &PDT);

  /// Returns true in case the value should be privatized, that is, a copy
  /// should be created for each parallel work-item.
  ///
  /// This is not the same as !isUniform() because of some of the allocas:
  /// The loop iteration variables are in some cases uniform. Each work item
  /// sees the same induction variable value at every iteration, but the
  /// variables should be still replicated to avoid multiple increments of
  /// the same induction variable by each work-item in a b-loop
  /// of which iterator cannot be merged cross WIs.
  bool shouldBePrivatized(llvm::Function *F, llvm::Value *Val);

  bool doFinalization(llvm::Module &M);
  void analyzeLoop(llvm::Function &F, llvm::Loop &L,
                   llvm::PostDominatorTree &PDT);
  bool isUniformLoop(llvm::Function &F, llvm::Loop &L);
  ~VariableUniformityAnalysisResult() { uniformityCache_.clear(); }

  bool invalidate(llvm::Function &F, const llvm::PreservedAnalyses PA,
                  llvm::AnalysisManager<llvm::Function>::Invalidator &Inv);

private:
  bool isUniformityAnalyzed(llvm::Function *F, llvm::Value *V) const;
  void removeUniformityData(llvm::Value &V, int Depth);
  void removeUniformityDatum(llvm::Function &F, llvm::Value &V);

  using UniformityIndex = std::map<llvm::Value *, bool>;
  using UniformityCache = std::map<llvm::Function *, UniformityIndex>;
  using LoopUniformityIndex = std::map<llvm::Loop *, bool>;
  using LoopUniformityCache = std::map<llvm::Function *, LoopUniformityIndex>;
  mutable UniformityCache uniformityCache_;
  mutable LoopUniformityCache LoopUniformityCache_;
};

} // namespace pocl
#endif
