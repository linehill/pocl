// Class for subgroup barrier instructions, modelled as a CallInstr.
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

#ifndef POCL_SGBARRIER_H
#define POCL_SGBARRIER_H

#include "Barrier.h"

namespace pocl {
/// Class for the subgroup barrier instruction, inherits from the 'Barrier'
/// class.
///
/// Enables the distinction between subgroup barriers and workgroup barriers.
/// It is used by the 'Fiber' method to handle the control flow of diverging
/// sub-groups. WILoops does not currently utilize this distinction.
/// Other work-group methods (for now) identify 'SubgroupBarrier' as
/// a 'Barrier' and treat it as a work-group barrier.
class SubgroupBarrier : public Barrier {
private:
#if LLVM_MAJOR < 20
  static SubgroupBarrier *create(llvm::Instruction *InsertBefore) {
    if (InsertBefore != &InsertBefore->getParent()->front() &&
#else
  static SubgroupBarrier *create(InstListType::iterator InsertBefore) {
    if (InsertBefore != InsertBefore->getParent()->begin() &&
#endif
        llvm::isa<SubgroupBarrier>(InsertBefore->getPrevNode()))
      return llvm::cast<SubgroupBarrier>(InsertBefore->getPrevNode());

    llvm::Module *M = InsertBefore->getModule();
    llvm::FunctionCallee FC = M->getOrInsertFunction(
        SGBARRIER_FUNCTION_NAME, llvm::Type::getVoidTy(M->getContext()));
    llvm::Function *F = llvm::cast<llvm::Function>(FC.getCallee());
    F->addFnAttr(llvm::Attribute::Convergent);
    return llvm::cast<pocl::SubgroupBarrier>(
        llvm::CallInst::Create(F, "", InsertBefore));
  }

public:
  static bool isLoopWithSGBarrier(llvm::Loop &L) {
    for (llvm::BasicBlock *BB : L.blocks())
      for (llvm::Instruction &I : *BB)
        if (llvm::isa<SubgroupBarrier>(&I))
          return true;

    return false;
  }

  static bool classof(const SubgroupBarrier *S) { return true; }

  static bool classof(const llvm::CallInst *C) {
    return C->getCalledFunction() != nullptr &&
           C->getCalledFunction()->getName() == SGBARRIER_FUNCTION_NAME;
  }
  static bool classof(const llvm::Instruction *I) {
    return (llvm::isa<llvm::CallInst>(I) &&
            classof(llvm::cast<llvm::CallInst>(I)));
  }
  static bool classof(const User *U) {
    return (llvm::isa<Instruction>(U) &&
            classof(llvm::cast<llvm::Instruction>(U)));
  }
  static bool classof(const Value *V) {
    return (llvm::isa<User>(V) && classof(llvm::cast<llvm::User>(V)));
  }
  static bool hasSGBarrier(const llvm::BasicBlock *BB) {
    for (const llvm::Instruction &I : *BB)
      if (llvm::isa<SubgroupBarrier>(&I))
        return true;
    return false;
  }

  static bool hasSGBarriers(const llvm::Function *F) {
    if (!F->getParent()->getFunction(SGBARRIER_FUNCTION_NAME))
      return false;

    for (auto &BB : *F)
      if (hasSGBarrier(&BB))
        return true;
    return false;
  }

#if LLVM_MAJOR < 20
  static SubgroupBarrier *createAtEnd(llvm::BasicBlock *BB) {
    return create(BB->getTerminator());
  }
#else
  static SubgroupBarrier *createAtEnd(llvm::BasicBlock *BB) {
    return create(BB->getTerminator()->getIterator());
  }
#endif

#if LLVM_MAJOR < 20
  static SubgroupBarrier *createAtStart(llvm::BasicBlock *BB) {
    return create(BB->getFirstNonPHI());
  }
#else
  static SubgroupBarrier *createAtStart(llvm::BasicBlock *BB) {
    return create(BB->getFirstInsertionPt());
  }
#endif

  // Returns true in case the given basic block ends with a subgroup barrier,
  // that is, contains only a branch instruction after a subgroup barrier call.
  static bool endsWithSGBarrier(const llvm::BasicBlock *BB) {
    const llvm::Instruction *Inst = BB->getTerminator();
    if (Inst == NULL)
      return false;
    return BB->size() > 1 && Inst->getPrevNode() != NULL &&
           llvm::isa<SubgroupBarrier>(Inst->getPrevNode());
  }
};

} // namespace pocl

#endif
