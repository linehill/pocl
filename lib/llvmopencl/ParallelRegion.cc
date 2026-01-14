// Class definition for parallel regions, a group of BasicBlocks that
// each kernel should run in parallel.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012-2019 Pekka Jääskeläinen
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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "Barrier.h"
#include "DebugHelpers.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "ParallelRegion.h"

POP_COMPILER_DIAGS

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

#include "pocl_llvm_api.h"

using namespace std;
using namespace llvm;
using namespace pocl;

#include <iostream>

#define DEBUG_TYPE "ParallelRegion"

#ifdef LLVM_DEBUG
#undef LLVM_DEBUG
#endif

#ifdef DEBUG_PR_CREATION
#define LLVM_DEBUG(X) X
#define dbgs() std::cerr << DEBUG_TYPE << ": "
#else
#define LLVM_DEBUG(X)
#endif

int ParallelRegion::idGen = 0;

ParallelRegion::ParallelRegion(int forcedRegionId)
    : exitIndex_(0), entryIndex_(0), pRegionId(forcedRegionId) {
  if (forcedRegionId == -1)
    pRegionId = idGen++;
}

/**
 * Ensure all variables are named so they will be replicated and renamed
 * correctly.
 */
void ParallelRegion::GenerateTempNames(llvm::BasicBlock *BB) {
  for (llvm::Instruction &Instr : *BB) {

    if (Instr.hasName() || !Instr.isUsedOutsideOfBlock(BB))
      continue;
    int TempCounter = 0;
    std::string TempName = "";
    do {
      std::ostringstream Name;
      Name << ".pocl_temp." << TempCounter;
      ++TempCounter;
      TempName = Name.str();
    } while (BB->getParent()->getValueSymbolTable()->lookup(TempName) != NULL);
    Instr.setName(TempName);
  }
}

void ParallelRegion::remap(ValueToValueMapTy &Map) {
  for (iterator i = begin(), e = end(); i != e; ++i) {
    LLVM_DEBUG(dbgs() << "### block before remap: \n");
    LLVM_DEBUG((*i)->dump());

    for (BasicBlock::iterator ii = (*i)->begin(), ee = (*i)->end();
         ii != ee; ++ii)
      RemapInstruction(&*ii, Map,
                       RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);

    LLVM_DEBUG(dbgs() << "### block after remap: \n");
    LLVM_DEBUG((*i)->dump());
  }
}

void ParallelRegion::chainAfter(ParallelRegion *Region) {
  /* If we are replicating a conditional barrier region, the last block can be
     an unreachable block to mark the impossible path. Skip it and choose the
     correct branch instead.

     TODO: why have the unreachable block there the first place? Could we just
     not add it and fix the branch? */
  BasicBlock *Tail = Region->exitBB();
  auto Term = Tail->getTerminator();
  if (isa<UnreachableInst>(Term)) {
    Tail = Region->at(Region->size() - 2);
    Term = Tail->getTerminator();
  }
#ifdef LLVM_BUILD_MODE_DEBUG
    if (Term->getNumSuccessors() != 1) {
      std::cout << "!!! trying to chain region" << std::endl;
      this->dumpNames();
      std::cout << "!!! after region" << std::endl;
      Region->dumpNames();
      Term->getParent()->dump();

      assert (Term->getNumSuccessors() == 1);
    }
#endif

    BasicBlock *Successor = Term->getSuccessor(0);
    Function *F = Successor->getParent();

  for (iterator i = begin(), e = end(); i != e; ++i)
    F->insert(Tail->getIterator(), *i);

  Term->setSuccessor(0, entryBB());

  Term = exitBB()->getTerminator();
  assert(Term->getNumSuccessors() == 1);
  Term->setSuccessor(0, Successor);
}

/**
 * Removes known dead side exits from parallel regions.
 *
 * These occur with conditional barriers. The head of the path
 * leading to the conditional barrier is shared by two PRs. The
 * first work-item defines which path is taken (by definition the
 * barrier is taken by all or none of the work-items). The blocks
 * in the branches are in different regions which can contain branches
 * to blocks that are in known non-taken path. This method replaces
 * the targets of such branches with undefined BBs so they will be cleaned
 * up by the optimizer.
 */
void
ParallelRegion::purge()
{
  SmallVector<BasicBlock *, 4> NewBlocks;

  // Go through all the BBs in the region and check their branch
  // targets, looking for destinations that are outside the region.
  // Only the last block in the PR can now contain such branches.
  for (iterator i = begin(), e = end(); i != e; ++i) {

    // Exit block has a successor out of the region.
    if (*i == exitBB())
      continue;
    LLVM_DEBUG(dbgs() << "### block before purge: \n");
    LLVM_DEBUG((*i)->dump());

    auto Terminator = (*i)->getTerminator();
    for (unsigned ii = 0, ee = Terminator->getNumSuccessors(); ii != ee; ++ii) {
      BasicBlock *Successor = Terminator->getSuccessor(ii);
      if (count(begin(), end(), Successor) == 0) {
        // This successor is not on the parallel region, purge.
        LLVM_DEBUG(dbgs() << "purging a branch to a block "
                          << Successor->getName().str()
                          << " outside the region\n");

        BasicBlock *Unreachable = BasicBlock::Create(
            (*i)->getContext(), (*i)->getName() + ".unreachable",
            (*i)->getParent(), back());
        new UnreachableInst(Unreachable->getContext(), Unreachable);
        Terminator->setSuccessor(ii, Unreachable);
        NewBlocks.push_back(Unreachable);
      }
    }
    LLVM_DEBUG(dbgs() << "### block after purge: \n");
    LLVM_DEBUG((*i)->dump());
  }

  // Add the new "unreachable" blocks to the
  // region. We cannot do in the loop as it
  // corrupts iterators.
  insert(end(), NewBlocks.begin(), NewBlocks.end());
}

void
ParallelRegion::insertLocalIdInit(llvm::BasicBlock* Entry,
                                  unsigned X, unsigned Y, unsigned Z) {

  IRBuilder<> Builder(Entry, Entry->getFirstInsertionPt());

  Module *M = Entry->getParent()->getParent();

  GlobalVariable *GVX = M->getGlobalVariable(LID_G_NAME(0));
  if (GVX != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT(M), X), GVX);

  GlobalVariable *GVY = M->getGlobalVariable(LID_G_NAME(1));
  if (GVY != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT(M), Y), GVY);

  GlobalVariable *GVZ = M->getGlobalVariable(LID_G_NAME(2));
  if (GVZ != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT(M), Z), GVZ);
}

void
ParallelRegion::insertPrologue(unsigned x,
                               unsigned y,
                               unsigned z)
{
  BasicBlock *entry = entryBB();
  ParallelRegion::insertLocalIdInit(entry, x, y, z);
}

void
ParallelRegion::dump()
{
#ifdef LLVM_BUILD_MODE_DEBUG
  for (iterator i = begin(), e = end(); i != e; ++i)
    (*i)->dump();
#endif
}

void
ParallelRegion::dumpNames()
{
  for (iterator i = begin(), e = end(); i != e; ++i)
    {
    std::cerr << (*i)->getName().str();
    if (entryBB() == (*i))
      std::cerr << "(EN)";
    if (exitBB() == (*i))
      std::cerr << "(EX)";
    std::cerr << " ";
    }
    std::cerr << std::endl;
}

// Recursive function to check if instruction depends on the 3D local OR global
// ids.
bool traceOperands(Value *V, std::set<Value *> &visited, int indent = 0) {
  if (!V || visited.count(V))
    return false;

  visited.insert(V);

  if (auto *CI = dyn_cast<CallInst>(V)) {
    Function *CalledFunc = CI->getCalledFunction();
    if (CalledFunc && ((CalledFunc->getName() == LID_BUILTIN_NAME) ||
                       (CalledFunc->getName() == GID_BUILTIN_NAME)))
      return true;
  }

  if (Instruction *I = dyn_cast<Instruction>(V)) {
    for (Value *Op : I->operands()) {
      if (traceOperands(Op, visited, indent + 2))
        return true;
    }
  }

  return false;
}

ParallelRegion *ParallelRegion::Create(const SmallPtrSet<BasicBlock *, 8> &BBs,
                                       BasicBlock *Entry, BasicBlock *Exit,
                                       bool IsSGRegion) {
  ParallelRegion *NewRegion = new ParallelRegion();

  assert(Entry != NULL);
  assert(Exit != NULL);

  bool NoIDRefs = true;

  // For subgroup regions, mark region as 'linearizable' IF there are NO
  // references to 3D ids (local OR global). This allows looping over PR in
  // a single loop (over local linear id).
  if (IsSGRegion) {
    NewRegion->setSGRegion();
    for (BasicBlock *BB : BBs) {
      for (Instruction &I : *BB) {
        std::set<Value *> visited;
        bool FoundRef = traceOperands(&I, visited);
        if (FoundRef) {
          NoIDRefs = false;
          break;
        }
      }
      if (!NoIDRefs)
        break;
    }
    if (NoIDRefs)
      NewRegion->markNoLocalIDReferences();
  }

  // This is done in two steps so the order of the vector is the same as
  // original function order.
  Function *F = Entry->getParent();
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    BasicBlock *B = &*i;
    for (SmallPtrSetIterator<BasicBlock *> j = BBs.begin(); j != BBs.end();
         ++j) {
      if (*j == B) {

#ifdef EXPLICIT_PR_NAMING
        std::string OldName = (*j)->getName().str();
        std::string GroupName = IsSGRegion ? "SGR_" : "WGR_";
        (*j)->setName(GroupName + OldName);
#endif

        NewRegion->push_back(&*i);
        if (Entry == *j)
          NewRegion->setEntryBBIndex(NewRegion->size() - 1);
        else if (Exit == *j)
          NewRegion->setExitBBIndex(NewRegion->size() - 1);
        break;
      }
    }
  }

  if (NewRegion->HasLocalIDReferences())
    NewRegion->localizeIDLoads();

  LLVM_DEBUG(assert(NewRegion->Verify()););

  return NewRegion;
}

/// Returns true if the paraller region is well-defined.
bool
ParallelRegion::verify([[maybe_unused]] bool AbortOnFailure)
{
  // Parallel region conditions:
  // 1) Single entry, in entry block.
  // 2) Single outgoing edge from exit block
  //    (other outgoing edges allowed, will be purged in replicas).
  // 3) No barriers inside the region.

  int entry_edges = 0;

  for (iterator i = begin(), e = end(); i != e; ++i) {
    for (pred_iterator ii(*i), ee(*i, true); ii != ee; ++ii) {
      if (count(begin(), end(), *ii) == 0) {
        if ((*i) != entryBB()) {
          dumpNames();
          std::cerr << "suspicious block: " << (*i)->getName().str() << std::endl;
          std::cerr << "the entry is: " << entryBB()->getName().str() << std::endl;

          ParallelRegion::ParallelRegionVector prvec;
          prvec.push_back(this);
          std::set<llvm::BasicBlock*> highlights;
          highlights.insert(entryBB());
          highlights.insert(*i);
          dumpCFG(*(*i)->getParent(), "ParllelRegion_verify.dot", nullptr,
                  &prvec, &highlights);
          assert(!AbortOnFailure && "Incoming edges to non-entry block!");
          return false;
        } else if (!Barrier::hasBarrier(*ii)) {
          (*i)->getParent()->viewCFG();
          assert(!AbortOnFailure && "Entry has edges from non-barrier blocks!");
          return false;
        }
        ++entry_edges;
      }
    }

    // if (entry_edges != 1) {
    //   assert(!AbortOnFailure && "Parallel regions must be single entry!");
    //   return false;
    // }
    if (exitBB()->getTerminator()->getNumSuccessors() != 1) {
      ParallelRegion::ParallelRegionVector regions;
      regions.push_back(this);

#ifdef LLVM_BUILD_MODE_DEBUG
      std::set<llvm::BasicBlock*> highlights;
      highlights.insert((*i));
      highlights.insert(exitBB());
      exitBB()->dump();
      dumpNames();
      dumpCFG(*(*i)->getParent(), "ParallelRegion_verify_broken.dot", nullptr,
              &regions, &highlights);
#endif

      assert(!AbortOnFailure && "Multiple outgoing edges from exit block!");
      return false;
    }

    for (BasicBlock::iterator ii = (*i)->begin(), ee = (*i)->end();
           ii != ee; ++ii) {
      if (isa<Barrier> (ii)) {
        assert(!AbortOnFailure && "Barrier found inside parallel region!");
        return false;
      }
    }
  }

  return true;
}

#define PARALLEL_MD_NAME "llvm.access.group"

/**
 * Adds metadata to all the memory instructions to denote
 * they originate from a parallel loop.
 *
 * Due to nested parallel loops, there can be multiple loop
 * references.
 *
 * Format (LLVM 8+):
 *
 *     !llvm.access.group !0
 *
 *     !0 distinct !{}
 *
 * In a 2-nested loop:
 *
 *     !llvm.access.group !0
 *
 *     !0 { !1, !2 }
 *     !1 distinct !{}
 *     !2 distinct !{}
 *
 * Parallel loop metadata prior to LLVM 12.0.1 on memory reads also implies that
 * if-conversion (i.e., speculative execution within a loop iteration) is safe.
 * Given an instruction reading from memory, IsLoadUnconditionallySafe should
 * return whether it is safe under (unconditional, unpredicated) speculative
 * execution. See https://bugs.llvm.org/show_bug.cgi?id=46666 and
 * https://github.com/pocl/pocl/issues/757.
 *
 * From LLVM 12.0.1 onward parallel loop metadata does not imply if-conversion
 * safety anymore. This got fixed by this change:
 * https://reviews.llvm.org/D103907 for LLVM 13 which also got backported to
 * LLVM 12.0.1. In other words this means that before the fix, the loop
 * vectorizer was not able to vectorize some kernels because they would required
 * a huge runtime memory check code insertion. Leading to vectorizer to give up.
 * With above fix, we can add metadata to every load.  This will cause
 * vectorizer to skip runtime memory check code insertion part because it
 * indicates that iterations do not depend on each other. Which in turn makes
 * vectorization easier. In this case using of IsLoadUnconditionallySafe
 * parameter will be skipped.
 */
void ParallelRegion::addParallelLoopMetadata(
    llvm::MDNode *Identifier,
    std::function<bool(llvm::Instruction *)> IsLoadUnconditionallySafe) {
  for (iterator i = begin(), e = end(); i != e; ++i) {
    BasicBlock *BB = *i;
    for (BasicBlock::iterator ii = BB->begin(), ee = BB->end(); ii != ee;
         ii++) {
      if (!ii->mayReadOrWriteMemory()) {
        continue;
      }

      MDNode *NewMD = MDNode::get(BB->getContext(), Identifier);
      MDNode *OldMD = ii->getMetadata(PARALLEL_MD_NAME);
      if (OldMD != nullptr) {
        NewMD = llvm::MDNode::concatenate(OldMD, NewMD);
      }
      ii->setMetadata(PARALLEL_MD_NAME, NewMD);
    }
  }
}

/**
 * Inserts a new basic block to the region, before an old basic block in
 * the region.
 *
 * Assumes the inserted block to be before the other block in control
 * flow, that is, there should be direct CFG edge from the block to the
 * other.
 */
void ParallelRegion::AddBlockBefore(llvm::BasicBlock *Block,
                                    llvm::BasicBlock *Before) {
  llvm::BasicBlock *OldExit = exitBB();
  ParallelRegion::iterator BeforePos = find(begin(), end(), Before);
  ParallelRegion::iterator OldExitPos = find(begin(), end(), OldExit);
  assert(BeforePos != end());

  /* The old exit node might is now pushed further, at most one position.
     Whether this is the case, depends if the node was inserted before or
     after that node in the vector. That is, if indexof(before) <
     indexof(oldExit). */
  if (BeforePos < OldExitPos)
    ++exitIndex_;

  insert(BeforePos, Block);
  /* The entryIndex_ should be still correct. In case the 'before' block
     was an old entry node, the new one replaces it as an entry node at
     the same index and the old one gets pushed forward. */
}

void ParallelRegion::AddBlockAfter(llvm::BasicBlock *Block,
                                   llvm::BasicBlock *After) {
  llvm::BasicBlock *OldExit = exitBB();
  ParallelRegion::iterator AfterPos = find(begin(), end(), After);
  ParallelRegion::iterator OldExitPos = find(begin(), end(), OldExit);
  assert(AfterPos != end());

  /* The old exit node might be pushed further, at most one position.
     Whether this is the case, depends if the node was inserted before or
     after that node in the vector. That is, if indexof(before) <
     indexof(oldExit). */
  if (AfterPos < OldExitPos)
    ++exitIndex_;
  AfterPos++;
  insert(AfterPos, Block);
}

bool ParallelRegion::hasBlock(llvm::BasicBlock *Block) {
  return find(begin(), end(), Block) != end();
}

/// Finds the instruction that loads an id of the work item in the
/// beginning of the parallel region, if not found, creates it.
///
/// \param IDGlobalName The name of the (magic) GlobalVariable temporally
/// representing the id.
/// \param Before If given, finds one in the basic block of the given
/// instruction, or creates one just before it.
/// \returns The instruction loading the id.
llvm::Instruction *
ParallelRegion::getOrCreateIDLoad(std::string IDGlobalName,
                                  llvm::Instruction *Before) {

  Module *M = entryBB()->getParent()->getParent();

  llvm::Type *ST = SizeT(M);
  GlobalVariable *IDGlobal =
      cast<GlobalVariable>(M->getOrInsertGlobal(IDGlobalName, ST));

  if (Before != nullptr) {
    // Try to find one in the same BB.
    BasicBlock *BB = Before->getParent();
    for (auto &I : *BB) {
      Instruction *BBInst = &I;

      if (BBInst == Before) {
        // Didn't find one before it. Create one.
        IRBuilder<> Builder(Before);
        return Builder.CreateLoad(ST, IDGlobal);
      }

      LoadInst *Load = dyn_cast<LoadInst>(BBInst);
      if (Load == nullptr)
        continue;
      GlobalVariable *Global =
          dyn_cast<GlobalVariable>(Load->getPointerOperand());
      if (Global == IDGlobal)
        return Load;
    }
  }

  // Otherwise, create one to the parallel region entry.
  if (IDLoadInstrs.find(IDGlobalName) != IDLoadInstrs.end())
    return IDLoadInstrs[IDGlobalName];

  GlobalVariable *Ptr =
      cast<GlobalVariable>(M->getOrInsertGlobal(IDGlobalName, ST));

  llvm::BasicBlock &BB = *entryBB();
  CreateBuilder(Builder, BB);

  Instruction *IDLoad = Builder.CreateLoad(ST, IDGlobal);
  IDLoadInstrs[IDGlobalName] = IDLoad;
  return IDLoad;
}

void ParallelRegion::InjectPrintF(llvm::Instruction *Before,
                                  std::string FormatStr,
                                  std::vector<Value *> &Params) {
  IRBuilder<> Builder(Before);
  llvm::Module *M = Before->getParent()->getParent()->getParent();

  llvm::Value *StringArg = Builder.CreateGlobalString(FormatStr);

  /* generated with help from https://llvm.org/demo/index.cgi */
  Function *PrintfFunc = M->getFunction("printf");
  if (PrintfFunc == nullptr) {
    PointerType *PointerTy4 =
        PointerType::get(IntegerType::get(M->getContext(), 8), 0);

    std::vector<Type *> FuncTy6Args;
    FuncTy6Args.push_back(PointerTy4);

    FunctionType *FuncTy6 =
        FunctionType::get(/*Result=*/IntegerType::get(M->getContext(), 32),
                          /*Params=*/FuncTy6Args,
                          /*isVarArg=*/true);

    PrintfFunc = Function::Create(/*Type=*/FuncTy6,
                                  /*Linkage=*/GlobalValue::ExternalLinkage,
                                  /*Name=*/"printf", M);
    PrintfFunc->setCallingConv(CallingConv::C);

    AttributeList FuncPrintfPAL =
        AttributeList()
#if LLVM_MAJOR < 21
            .addAttributeAtIndex(M->getContext(), 1U, Attribute::NoCapture)
#endif
            .addAttributeAtIndex(M->getContext(), 4294967295U,
                                 Attribute::NoUnwind);

    PrintfFunc->setAttributes(FuncPrintfPAL);
  }

  std::vector<Constant *> ConstPtr8Indices;

  ConstantInt *ConstInt64_9 =
      ConstantInt::get(M->getContext(), APInt(64, StringRef("0"), 10));
  ConstPtr8Indices.push_back(ConstInt64_9);
  ConstPtr8Indices.push_back(ConstInt64_9);
  assert(isa<Constant>(StringArg));
  Constant *ConstPtr8 = ConstantExpr::getGetElementPtr(
      PointerType::getUnqual(Type::getInt8Ty(M->getContext())),
      cast<Constant>(StringArg), ConstPtr8Indices);

  std::vector<Value *> Args;
  Args.push_back(ConstPtr8);
  Args.insert(Args.end(), Params.begin(), Params.end());

  CallInst::Create(PrintfFunc, Args, "", Inst2InsertPt(Before));
}

void ParallelRegion::SetExitBB(llvm::BasicBlock *Block) {
  for (size_t i = 0; i < size(); ++i)
    {
    if (at(i) == Block) {
      setExitBBIndex(i);
      return;
    }
    }
  assert (false && "The block was not found in the PRegion!");
}

/**
 * Adds a printf to the end of the parallel region that prints the
 * region ID and the work item ID.
 *
 * Useful for debugging control flow bugs.
 */
void
ParallelRegion::InjectRegionPrintF()
{
  llvm::Module *M = entryBB()->getParent()->getParent();

#if 0
  // it should reuse equal strings anyways
  const char* FORMAT_STR_VAR = ".pocl.pRegion_debug_str";
  llvm::Value *StringArg = M->getGlobalVariable(FORMAT_STR_VAR);
  if (StringArg == nullptr)
    {
      IRBuilder<> builder(entryBB());
      StringArg = builder.CreateGlobalString("PR %d WI %u %u %u\n", FORMAT_STR_VAR);
    }
#endif

  ConstantInt *pRID =
      ConstantInt::get(M->getContext(), APInt(32, (uint64_t)pRegionId));
  std::vector<Value *> Params;
  Params.push_back(pRID);
  Params.push_back(getOrCreateIDLoad(LID_G_NAME(0)));
  Params.push_back(getOrCreateIDLoad(LID_G_NAME(1)));
  Params.push_back(getOrCreateIDLoad(LID_G_NAME(2)));

  InjectPrintF(exitBB()->getTerminator(), "PR %d WI %u %u %u\n", Params);
}

/// Adds a printf to the end of the parallel region that prints the
/// hex contents of all named non-pointer variables.
///
/// Useful for debugging data flow bugs.
///
void ParallelRegion::InjectVariablePrintouts() {
  for (ParallelRegion::iterator i = begin(); i != end(); ++i) {
    llvm::BasicBlock *BB = *i;
    for (llvm::BasicBlock::iterator Instr = BB->begin(); Instr != BB->end();
         ++Instr) {
      llvm::Instruction *Instruction = &*Instr;
      if (isa<PointerType>(Instruction->getType()) || !Instruction->hasName())
        continue;
      std::string Name = Instruction->getName().str();
      std::vector<Value *> Args;
      IRBuilder<> Builder(exitBB()->getTerminator());
      Args.push_back(Builder.CreateGlobalString(Name));
      Args.push_back(Instruction);
      InjectPrintF(Instruction->getParent()->getTerminator(),
                   "variable %s == %x\n", Args);
    }
  }
}

/// Localizes all the loads to the the work-item identifiers.
///
/// In case the code inside the region queries the WI id, it should not (re)use
/// one that is loaded in another region, but one that is loaded in the same
/// region. Otherwise, it ends up using the last id the previous PR work-item
/// loop got. This caused problems in cases where the local id was stored to a
/// temporary variable in an earlier region and that temp was reused later.
///
/// The function scans for all accesses to the local and global ids and converts
/// them to loads inside the parallel region.
void ParallelRegion::localizeIDLoads() {
  // The id loads inside the parallel region.
  std::array<llvm::Instruction *, 6> RegionIDLoads = {
      getOrCreateIDLoad(LID_G_NAME(0)), getOrCreateIDLoad(LID_G_NAME(1)),
      getOrCreateIDLoad(LID_G_NAME(2)), getOrCreateIDLoad(GID_G_NAME(0)),
      getOrCreateIDLoad(GID_G_NAME(1)), getOrCreateIDLoad(GID_G_NAME(2))};

  llvm::Module *M = RegionIDLoads[0]->getParent()->getParent()->getParent();

  std::array<llvm::Value *, 6> Globals = {
      M->getNamedGlobal(LID_G_NAME(0)), M->getNamedGlobal(LID_G_NAME(1)),
      M->getNamedGlobal(LID_G_NAME(2)), M->getNamedGlobal(GID_G_NAME(0)),
      M->getNamedGlobal(GID_G_NAME(1)), M->getNamedGlobal(GID_G_NAME(2))};

  for (ParallelRegion::iterator BBI = begin(); BBI != end(); ++BBI) {
    llvm::BasicBlock *BB = *BBI;
    for (llvm::BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II) {
      llvm::Instruction *Instr = &*II;

      // If any of the operands is using an id, replace it with the
      // intra-PR load from the parallel region specific id variable.
      for (unsigned Opr = 0; Opr < Instr->getNumOperands(); ++Opr) {
        llvm::LoadInst *Load = dyn_cast<llvm::LoadInst>(Instr->getOperand(Opr));
        if (Load == NULL)
          continue;

        if (std::find(RegionIDLoads.begin(), RegionIDLoads.end(), Load) !=
            RegionIDLoads.end())
          continue; // Already converted.

        auto Pos = std::find(Globals.begin(), Globals.end(),
                             Load->getPointerOperand());
        if (Pos == Globals.end())
          continue;

        Instr->setOperand(Opr, RegionIDLoads[Pos - Globals.begin()]);
      }
    }
  }
}

bool ParallelRegion::shouldBeSerialized() const {
  return false;
  // We need to scan the instructions as the contents might be changed during
  // transformations.
  for (auto *BB : BBs_) {
    for (auto &I : *BB) {
      if (llvm::StoreInst *Store = dyn_cast_or_null<llvm::StoreInst>(&I)) {
        if (llvm::AllocaInst *Alloca = dyn_cast_or_null<llvm::AllocaInst>(
                Store->getPointerOperand())) {
          if (Alloca->getParent() == &BB->getParent()->getEntryBlock()) {
            LLVM_DEBUG(
                dbgs()
                << "#### serializing a region due to use of wg variables:\n");
            LLVM_DEBUG(Store->dump());
            return true;
          }
        }
      }
    }
  }
  return false;
}
