// Provides transformation to improve vectorization on kernels with divergent
// exits.
//
// Copyright (c) 2026 Henry Linjamäki / Tampere University
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

// A transformation method in this file aims to improve auto-vectorization
// opportunities on kernels featuring "early-exits" (EE) - out-of-bounds checks
// over global or work-group dimensions at very beginning of the kernels - for
// example:
//
//    kernel void k(..., int n) {
//      int tid = get_global_id(0);
//      if (tid >= n) // "early-exit".
//        return;
//      some_work(...);
//    }
//
//  The exit idoms like this hinder the opportunity to form separate parallel
//  region over `some_work()` (= place implicit barriers around it) to aid in
//  the quality of auto-vectorization* because of the *divergent* exit
//  branch. Whenever possible and legal, the UniformizeDivergentExits
//  transformer attempts to transform the branch to be uniform by "shrinking"
//  the work-group to work-item set that countinues executing the kernel.  An
//  example in pseudo-code:
//
//    kernel void k(..., int n) {
//      // Count number of WIs that continues executing the kernel after the
//      // early-exit check.
//      int t = get_group_id(0) * get_local_size(0);
//      t = n - t;
//      t = min(max(t , 0), get_local_size(0));
//
//      // Set WI-loop bounds over WG dimension 0. This makes WI-loop over
//      // dimension 0 ahead the branch loop over [0, t) instead of
//      // [0, get_local_size(0)).
//      _wiloop_lower_bound_x = 0;
//      _wiloop_upper_bound_x = t;
//
//      // Uniformized branch.
//      if (t != 0)
//        return;
//
//      some_work(...);
//    }
//
// *: Not true in some cases in some cases. This transformer enables
//    enforceOuterLoopParIfBeneficial() to take action but leads to performance
//    regression in some cases it currently doesn't analysis its action is
//    benecifial. This transformer is disabled by defauls because of it (can be
//    enabled with POCL_UNIFORMIZE_DIVERGENT_EXITS=1 environment variable).
//
// Limitations:
//
// * Currently the transform can be applied on WI-loops that utilize
//   _wiloop_lower_bound_{x,y,z} and _wiloop_upper_bound_{x,y,z} variables -
//   e.g. this excludes linear WI-loops
//
// * Only the first branch from the entry of the kernel is considered as
//   candidate for divergent exit uniformization.

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "UniformizeDivergentExits.h"

#include "DebugHelpers.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "PoCLPatternMatch.h"
#include "SubgroupBarrier.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "pocl_llvm_api.h"

#include "llvm/IR/CFG.h"
#include "llvm/IR/ConstantRange.h"
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

POP_COMPILER_DIAGS

#include "pocl_runtime_config.h"

#include <string>
#include <vector>

#define DEBUG_TYPE "uniformize-divergent-exits"

//#define ENABLE_DEBUG
#include "TemporaryLLVMDebugMacros.hh"

namespace pocl {

using namespace llvm;

// Return true if the 'BB' potentially has user-visible effects beyond kernel.
static bool hasSideEffectsOutsideKernel(const BasicBlock *BB) {
  assert(BB);

  // The BB is really unreachable just because currently there is no builtins
  // with "noreturn" attributes in the input languages.
  if (isa<UnreachableInst>(BB->getTerminator()))
    return false;

  for (auto &I : *BB) {
    // TODO: private memory writes should not have visible effects beyond the
    //       kernel.

    if (I.mayHaveSideEffects()) {
      return true;
    }
  }

  return false;
}

using SideEffectMapT = std::map<const BasicBlock *, bool>;

// Return true if the 'BB' and its successors recursively potentially has
// user-visible effects.
static bool pathHasSideEffectsOutsideKernel(const BasicBlock *BB,
                                            SideEffectMapT &SideEffectMap) {
  // Caveat: this function currently doesn't detect (potential) infinite loops
  // and consider them as kernel side-effect. A silly example:
  //
  //   if (get_global_id(0) > n) while(1) {}; some_work();

  assert(BB);

  if (SideEffectMap.count(BB))
    return SideEffectMap[BB];

  bool HasSideEffects = hasSideEffectsOutsideKernel(BB);

  if (!HasSideEffects) {
    // Mark the BB visited to avoid infinite recursion.
    SideEffectMap[BB] = false;

    for (auto *Succ : successors(BB))
      HasSideEffects |= pathHasSideEffectsOutsideKernel(Succ, SideEffectMap);
  }

  SideEffectMap[BB] = HasSideEffects;
  return HasSideEffects;
}

static bool pathHasSideEffectsOutsideKernel(const BasicBlock *BB) {
  SideEffectMapT SideEffectMap;
  return pathHasSideEffectsOutsideKernel(BB, SideEffectMap);
}

struct DivergentExitCandidate {
  // A block with BranchInst with divergent condition.
  BasicBlock *Block;
  // Indicates which branch label leads to the kernel exit, without visible
  // effects outside the kernel if omitted.
  bool TrueLabelExits;
};

/// Returns divergent exit blocks
///
/// "Divergent exit block": an uniform block with a divergent branch whose other
/// edge leads to function returns without side-effects across all paths through
/// the edge.
static std::vector<DivergentExitCandidate>
getDivergentExits(BasicBlock &StartBlock, PostDominatorTree &PDT,
                  VariableUniformityAnalysisResult &VUA) {

  std::vector<DivergentExitCandidate> Result;

  auto *F = StartBlock.getParent();
  auto *BB = &StartBlock;
  while (auto *Succ = BB->getSingleSuccessor()) // Skip unconditional branches.
    BB = Succ;

  // Only consider the first conditional branch instruction, for now.
  auto *BI = dyn_cast<BranchInst>(BB->getTerminator());
  if (!BI)
    return Result;

  // For now, only consider the first divergent branch. It's possible to have
  // kernels with multiple divergent exit blocks through uniform branches.
  //
  // TODO: support cases where divergent branch appears at or after a join
  //       block. E.g:
  //
  //     Entry
  //       |\
  //       | B
  //       |/
  //       C
  //       |\
  //       | Exit
  //       D
  if (VUA.isUniform(F, BI))
    return Result;

  bool SideEffectsInSucc[2];
  SideEffectsInSucc[0] = pathHasSideEffectsOutsideKernel(BI->getSuccessor(0));
  SideEffectsInSucc[1] = pathHasSideEffectsOutsideKernel(BI->getSuccessor(1));

  if (SideEffectsInSucc[0] && SideEffectsInSucc[1])
    return Result;

  if (!SideEffectsInSucc[0] && !SideEffectsInSucc[1]) {
    // Unexpected case. Either pathHasSideEffectsOutsideKernel() calls have a
    // bug or there is really dead code ahead that haven't been
    // opportunistically eliminated.
    //
    // This assert here is for prompting investigation and can be removed if it
    // produces too much false alarms.
    assert(false && "Divergent exit block has side-effectless successors!");

    return Result;
  }

  Result.push_back(DivergentExitCandidate{BB, !SideEffectsInSucc[0]});
  return Result;
}

using WGMin = std::tuple<Value *, unsigned>;
static const WGMin NullWGMin = WGMin{nullptr, UINT_MAX};

/// If 'V' is known to equal to 'get_local_id(Dim) + U' (without overflows),
/// where 'U' and 'Dim' are some uniform values, for all WIs in WG, return {U,
/// Dim}. If not, return {nullptr, UINT_MAX}.
///
/// 'UsedAsSigned' indicates whether 'V' is interpreted as signed integer.
///
/// Code is possibly inserted before 'InsPt' for calculating the 'U' value
static std::tuple<Value *, unsigned>
getUniformBaseValue(Module *M, Value *V, bool UsedAsSigned,
                    BasicBlock::iterator InsPt) {
  using namespace llvm::PatternMatch;

  IRBuilder<> B(&*InsPt);

  uint64_t Dim;
  if (match(V, m_GlobalID(m_ConstantInt(Dim))) && Dim <= 3) {
    Value *BaseGID = createBaseGlobalID(M, Dim, InsPt);
    BaseGID = B.CreateZExtOrTrunc(BaseGID, V->getType());
    return {BaseGID, Dim};
  }

  if (match(V, m_Trunc(m_GlobalID(m_ConstantInt(Dim)))) && Dim <= 3) {
    auto *TI = cast<TruncInst>(V);

    // Check the GID doesn't get clipped or interpreted as negative.  Such cases
    // need two separate WI-loops per dimension which are not supported for
    // now. E.g.
    //
    //   %gid64 = call i64 @_Z12get_global_idj(i32 0)
    //   %gid32 = trunc %gid64 to i32
    //   %c = icmp sge i32 %gid32, %n
    //   br i1 %c, label %exit, label %continue
    //
    // Needs one WI-loop for WIs with GID0 in range [0, %n) and another for WIs
    // with GID0 in range (IIN32_MAX, GridSize).
    bool IsSupported = TI->hasNoUnsignedWrap() && TI->hasNoSignedWrap();

    auto BitWidth = TI->getType()->getIntegerBitWidth();
    auto GID = cast<Instruction>(TI->getOperand(0));
    auto *RangeMD = GID->getMetadata(LLVMContext::MD_range);
    if (!IsSupported && RangeMD) {
      auto CR = getConstantRangeFromMetadata(*RangeMD);
      if (BitWidth >= (CR.getActiveBits() + UsedAsSigned))
        IsSupported = true;
    }

    if (IsSupported) {
      Value *BaseGID = createBaseGlobalID(M, Dim, InsPt);
      BaseGID = B.CreateZExtOrTrunc(BaseGID, V->getType());
      return {BaseGID, Dim};
    }
  }

  return NullWGMin;
}

// Emit code to constraint 'ValueToClamp' between [LowerBound, UpperBound].
static Value *createClamp(IRBuilder<> &B, Value *LowerBound,
                          Value *ValueToClamp, Value *UpperBound, bool Signed) {

  auto MinOpcId = Signed ? Intrinsic::smin : Intrinsic::umin;
  auto MaxOpcId = Signed ? Intrinsic::smax : Intrinsic::umax;

  auto *V = B.CreateBinaryIntrinsic(MaxOpcId, ValueToClamp, LowerBound);
  return B.CreateBinaryIntrinsic(MinOpcId, V, UpperBound);
}

static bool tryUniformizeDivergentExit(DivergentExitCandidate &DEC,
                                       VariableUniformityAnalysisResult &VUA) {
  LLVM_DEBUG(dbgs() << "Try uniformize divergent exit branch in: "
                    << getNameOrAsOperand(DEC.Block) << "\n");

  auto *F = DEC.Block->getParent();
  auto *M = F->getParent();

  // Established by DivergentExitCandidate struct.
  auto *BI = cast<BranchInst>(DEC.Block->getTerminator());

  auto *ICmp = dyn_cast<ICmpInst>(BI->getCondition());
  if (!ICmp) {
    LLVM_DEBUG(dbgs() << "  bail out: exit condition is not an icmp.\n");
    return false;
  }

  auto *Lhs = ICmp->getOperand(0);
  auto *Rhs = ICmp->getOperand(1);
  auto Pred = ICmp->getPredicate();
  bool RhsIsUniform = VUA.isUniform(F, Rhs);
  bool LhsIsUniform = VUA.isUniform(F, Lhs);

  if (!is_contained({CmpInst::ICMP_UGT, CmpInst::ICMP_UGE, CmpInst::ICMP_ULT,
                     CmpInst::ICMP_ULE, CmpInst::ICMP_SGT, CmpInst::ICMP_SGE,
                     CmpInst::ICMP_SLT, CmpInst::ICMP_SLE},
                    Pred)) {
    LLVM_DEBUG(dbgs() << "  bail out: unsuitable compare predicate\n");
    return false;
  }

  if (RhsIsUniform && LhsIsUniform) {
    LLVM_DEBUG(dbgs() << "  bail out: both the compare operands are uniform\n");
    return false;
  }

  if (!RhsIsUniform && !LhsIsUniform) {
    LLVM_DEBUG(
        dbgs() << "  bail out: both the compare operands are divergent\n");
    return false;
  }

  auto *ExitSuccessor = BI->getSuccessor(!DEC.TrueLabelExits);
  auto *ContinueSuccessor = BI->getSuccessor(DEC.TrueLabelExits);

  // Canonicalize the compare:

  // - make non-exiting path to be taken on true predicate.
  if (DEC.TrueLabelExits)
    Pred = CmpInst::getInversePredicate(Pred);

  // - put the uniform compare operand on the right side.
  if (LhsIsUniform) {
    std::swap(RhsIsUniform, LhsIsUniform);
    std::swap(Rhs, Lhs);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  assert(RhsIsUniform);

  // Done: compare expression is now 'X (pred) U' where 'X' is a diverging
  // value, 'U' is an uniform value and '(pred)' is one of {<, <=, >, >=} and a
  // WI exits the kernel when the predicate is false.

  LLVM_DEBUG(dbgs() << "  canonicalized exit branch condition: '%"
                    << getNameOrAsOperand(Lhs) << " "
                    << CmpInst::getPredicateName(Pred) << " %"
                    << getNameOrAsOperand(Rhs) << "' (false -> exit)\n");

  // For now, restrict EE handling for > and >= predicates this means we only
  // need to define upper bound for the work-item loops.
  if (!is_contained({CmpInst::ICMP_ULT, CmpInst::ICMP_ULE, CmpInst::ICMP_SLT,
                     CmpInst::ICMP_SLE},
                    Pred)) {
    LLVM_DEBUG(dbgs() << "  bail out: unimplemented support for "
                      << CmpInst::getPredicateName(Pred) << "\n");
    return false;
  }

  auto [XBase, Dim] =
      getUniformBaseValue(M, Lhs, ICmp->isSigned(), BasicBlock::iterator(ICmp));
  if (!XBase) {
    LLVM_DEBUG(dbgs() << "  bail out: couldn't determine uniform base value\n");
    return false;
  }

  // All's set - create new kernel exit block.

  auto *NewExitBlock = SplitBlock(DEC.Block, BI);
  markAsPureUniformBlock(NewExitBlock, "uniformized exit");
  NewExitBlock->setName("uniformized.exit");

  // Emit code for calculating number of WI across the 'Dim' that continues
  // executing the kernel.

  IRBuilder<> B(NewExitBlock, BasicBlock::iterator(BI));
  Value *NewWICount = B.CreateSub(Rhs, XBase);
  if (Pred == CmpInst::ICMP_UGE)
    NewWICount = B.CreateAdd(NewWICount, ConstantInt::get(Lhs->getType(), 1));

  // Calculate new WI-loop bounds. Note, that if we were to support kernels with
  // chained divergent-exits - e.g.
  //
  //   if (get_global_id(0) >= A) return;
  //   some_work(...);
  //   if (get_global_id(0) >= B) return;
  //   more_work(...);
  //
  // We would need to merge the new bounds with the previous ones.

  Value *LocalSize = getWorkgroupLocalSize(M, Dim, B.GetInsertPoint());
  LocalSize = B.CreateZExtOrTrunc(LocalSize, Lhs->getType());

  auto *WILowerBound = ConstantInt::get(Lhs->getType(), 0);
  auto *WIUpperBound = createClamp(B, ConstantInt::get(Lhs->getType(), 0),
                                   NewWICount, LocalSize, ICmp->isSigned());

  B.CreateStore(WILowerBound, getOrCreateWILoopLowerBoundGV(M, Dim));
  B.CreateStore(WIUpperBound, getOrCreateWILoopUpperBoundGV(M, Dim));

  // Fix the original branch. It's possible, the calculated bounds yield [N, N)
  // range, meaning all WIs would exit the kernel at this point in the original
  // code. In princible, we can continue, as the parallel region loops should
  // not process any work-items. However, we might execute code in the
  // pure-uniform blocks with unintended side-effects - a such case is case8 in
  // test_divergent_exits.cpp where the kernel enters into an infinite loop
  // unexpectedly.
  const bool ExitForEmptyWGs = true;

  if (ExitForEmptyWGs) {
    auto *Continue = B.CreateICmp(CmpInst::ICMP_NE, WILowerBound, WIUpperBound,
                                  "continue_cond");
    BI->setCondition(Continue);
    BI->setSuccessor(0, ContinueSuccessor);
    BI->setSuccessor(1, ExitSuccessor);
  } else {
    auto *NewBI = BranchInst::Create(ContinueSuccessor);
    ReplaceInstWithInst(BI, NewBI);
    BI = NewBI;
  }

  markAsPureUniformBlock(NewExitBlock, "uniformized exit");
  NewExitBlock->setName("uniformized.exit");

  LLVM_DEBUG(dbgs() << "  success\n");
  return true;
}

bool UniformizeDivergentExits(Function &F, PostDominatorTree &PDT,
                              VariableUniformityAnalysisResult &VUA) {
  if (!isKernelToProcess(F))
    return false;

  // This transformer defaults off because of performance regressions in some
  // kernels. An example of this are loops that vectorize better
  // horizontally. For default enablement we need to come up with heurestics in
  // outerLoopIsLikelyBeneficial() to avoid the regression or improve
  // loop-interchange pass in the upstream.
  if (!pocl_get_bool_option("POCL_UNIFORMIZE_DIVERGENT_EXITS", false))
    return false;

  bool Changed = false;

  LLVM_DEBUG(dbgs() << "process: " << F.getName() << "\n");

  bool WGDynamicLocalSize = true;
  if (getModuleBoolMetadata(*F.getParent(), "WGDynamicLocalSize",
                            WGDynamicLocalSize) &&
      WGDynamicLocalSize) {
    // Bail out for now. Dynamic local sizes can be supported but WI-loop bounds
    // setup needs tweaking first.
    LLVM_DEBUG(dbgs() << "  bail out: support for dynamic local sizes is not"
                         " implemented. ");

    return false;
  }

  // Bail out if kernel may require linear WI-loop strategy which doesn't
  // support dynamic WI-loop bounds.
  if (SubgroupBarrier::hasSGBarriers(&F) ||
      F.hasMetadata("intel_reqd_sub_group_size")) {
    LLVM_DEBUG(dbgs() << "  bail out: kernel may require linear WI-loops.");
    return false;
  }

  for (auto &DEC : getDivergentExits(F.getEntryBlock(), PDT, VUA))
    Changed |= tryUniformizeDivergentExit(DEC, VUA);

  if (Changed) {
    EliminateUnreachableBlocks(F);

    setModuleBoolMetadata(F.getParent(), "invariant_wiloop_bounds", false);

    // This metadata is for internal regression tests.
    setModuleBoolMetadata(F.getParent(), "has_uniformized_exit", true);
  }

  return Changed;
}

} // namespace pocl
