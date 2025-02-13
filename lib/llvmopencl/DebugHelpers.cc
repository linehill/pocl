// Helpers for debugging the kernel compiler.
//
// Copyright (c) 2013-2019 Pekka Jääskeläinen
//               2024-2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "pocl.h"

#include <llvm/Analysis/RegionInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/GraphWriter.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"
#include "DebugHelpers.h"
#include "SubgroupBarrier.h"
#include "WorkgroupBarrier.h"

#ifdef dumpCFG
#undef dumpCFG
#endif

#include "LLVMUtils.h"
#include "Workgroup.h"
#include "pocl_file_util.h"
#include "pocl_runtime_config.h"

POP_COMPILER_DIAGS

using namespace llvm;

namespace pocl {

static std::string getDotBasicBlockID(llvm::BasicBlock* bb) {
  std::ostringstream namess;
  namess << "BB" << std::hex << bb;
  return namess.str();
}

static void printBranches(
  llvm::BasicBlock* b, std::ostream& s, bool /*highlighted*/) {

  auto term = b->getTerminator();
  for (unsigned i = 0; i < term->getNumSuccessors(); ++i)
    {
      BasicBlock *succ = term->getSuccessor(i);
      s << getDotBasicBlockID(b) << " -> " << getDotBasicBlockID(succ) << ";"
        << std::endl;
    }
  s << std::endl;
}

static void printBasicBlock(
  llvm::BasicBlock* B, std::ostream& S, bool Highlighted) {

  S << getDotBasicBlockID(B);
  S << "[shape=rect,style=";
  if (WorkgroupBarrier::hasWGBarrier(B) || isPureUniformBlock(B))
      S << "dotted";
  else if (SubgroupBarrier::hasSGBarrier(B) || isPureUniformBlock(B))
      S << "dashed";
  else
    S << "solid";

  if (WorkgroupBarrier::hasWGBarrier(B)) {
    S << ",fillcolor=red,style=filled";
  } else if (SubgroupBarrier::hasSGBarrier(B)) {
    S << ", fillcolor=blue,style=filled";
  } else if (isPureUniformBlock(B)) {
    S << ",fillcolor=grey,style=filled";
  }
  if (Highlighted) {
    // Highlight the wanted nodes with a thick blue border.
    S << ",color=blue,penwidth=3";
  }
  S << ",label=\"" << B->getName().str() << ":\\n";

  // The work-item loop control structures.
  if (B->getName().starts_with("pregion_for_cond")) {
    S << "wi-loop branch\\n";
  } else if (B->getName().starts_with("pregion_for_inc")) {
    S << "local_id_* increment\\n";
  } else if (B->getName().starts_with("pregion_for_init")) {
    S << "wi-loop init\\n";
  } else if (B->getName().starts_with("pregion_for_end")) {
    S << "wi-loop exit\\n";
  } else {
    // analyze the contents of the BB
    int PreviousNonHighlighted = 0;
    for (llvm::BasicBlock::iterator Instr = B->begin(); Instr != B->end();
         ++Instr) {

      llvm::CallInst *Call = dyn_cast<CallInst>(Instr);
      if (Call != nullptr && Call->getCalledFunction() != nullptr &&
          Call->getCalledFunction()->getName().str() == "printf") {
        // Highlight printfs and try to print their format string
        // for spotting basic blocks with printfs of interest.
        std::string FmtString = "...";

        int MaxCharsToPrint = 20;
        llvm::GlobalVariable *FmtStrArg =
            dyn_cast<llvm::GlobalVariable>(Call->getArgOperand(0));
        S << "printf(\\\"";
        if (FmtStrArg != nullptr) {
          Constant *Initializer = FmtStrArg->getInitializer();
          int CharI = 0;
          while (Constant *Char = Initializer->getAggregateElement(CharI++)) {
            if (--MaxCharsToPrint == 0)
              break;
            char CharVal =
                (char)Char->getUniqueInteger().extractBitsAsZExtValue(8, 0);
            if (CharVal == 0)
              break;
            if (CharVal == '\n')
              S << "\\\\n";
            else
              S << CharVal;
          }
          if (MaxCharsToPrint == 0)
            S << "...";
        } else {
          S << "...";
        }
        S << "\\\", ...)\\n";
        PreviousNonHighlighted = 0;
      } else if (isa<WorkgroupBarrier>(Instr)) {
        S << "WG-BARRIER\\n";
        PreviousNonHighlighted = 0;
      } else if (isa<SubgroupBarrier>(Instr)) {
        S << "SG-BARRIER\\n";
        PreviousNonHighlighted = 0;
      } else if (isa<BranchInst>(Instr)) {
        S << "branch\\n";
        PreviousNonHighlighted = 0;
      } else if (isa<PHINode>(Instr)) {
        S << "PHI\\n";
        PreviousNonHighlighted = 0;
      } else if (isa<ReturnInst>(Instr)) {
        S << "RETURN\\n";
        PreviousNonHighlighted = 0;
      } else if (isa<UnreachableInst>(Instr)) {
        S << "UNREACHABLE\\n";
        PreviousNonHighlighted = 0;
      } else {
        if (PreviousNonHighlighted == 0)
          S << "...program instructions...\\n";
        PreviousNonHighlighted++;
      }
    }
  }
  S << "\"";
  S << "]";
  S << ";" << std::endl << std::endl;
}

void dumpCFG(llvm::Function &F, std::string FileName,
             const std::vector<llvm::Region *> *Regions,
             const ParallelRegion::ParallelRegionVector *ParRegions,
             const std::set<llvm::BasicBlock *> *Highlights) {
  unsigned LastRegID = 0;

  // By default dump the dots to the kernel compiler cache/tmp directory,
  // if set explicitly to avoid polluting the CWD.
  const char *TmpPath = pocl_get_string_option("POCL_CACHE_DIR", nullptr);
  if (TmpPath != nullptr)
    FileName = std::string(TmpPath) + "/" + FileName;

  std::string OrigName = FileName;

  int Counter = 0;
  while (pocl_exists(FileName.c_str())) {
    std::ostringstream SS;
    SS << OrigName << "." << Counter;
    FileName = SS.str();
    ++Counter;
  }

  std::ofstream S;
  S.open(FileName.c_str(), std::ios::trunc);
  S << "digraph " << F.getName().str() << " {" << std::endl;

  std::set<BasicBlock*> RegionBBs;

  if (Regions != nullptr && Regions->size()) {
    for (const Region *R : *Regions) {
      unsigned RegID = ++LastRegID;
      S << "\tsubgraph cluster" << RegID << " {" << std::endl;
      for (Region::const_block_iterator RI = R->block_begin(),
                                        RE = R->block_end();
           RI != RE; ++RI) {
        BasicBlock *BB = *RI;
        printBasicBlock(
            BB, S,
            (Highlights != NULL && Highlights->find(BB) != Highlights->end()));
        RegionBBs.insert(BB);
      }
      S << "label=\"Parallel region #" << RegID << "\";" << std::endl;
      S << "}" << std::endl;
    }
  }

  if (ParRegions != nullptr) {
    for (ParallelRegion::ParallelRegionVector::const_iterator
             RI = ParRegions->begin(),
             RE = ParRegions->end();
         RI != RE; ++RI) {
      ParallelRegion *PR = *RI;
      S << "\tsubgraph cluster" << PR->getID() << " {" << std::endl;
      for (ParallelRegion::iterator It = PR->begin(), E = PR->end(); It != E;
           ++It) {
        BasicBlock *BB = *It;
        printBasicBlock(
            BB, S,
            (Highlights != NULL && Highlights->find(BB) != Highlights->end()));
        RegionBBs.insert(BB);
      }
      S << "label=\"Parallel region #" << PR->getID() << "\";" << std::endl;
      S << "}" << std::endl;
    }
  }
  for (Function::iterator FI = F.begin(), e = F.end(); FI != e; ++FI) {
    BasicBlock *BB = &*FI;
    if (RegionBBs.find(BB) != RegionBBs.end())
      continue;
    printBasicBlock(
        BB, S, Highlights != NULL && Highlights->find(BB) != Highlights->end());
  }

  for (Function::iterator FI = F.begin(), e = F.end(); FI != e; ++FI) {
    BasicBlock *BB = &*FI;
    printBranches(
        BB, S, Highlights != NULL && Highlights->find(BB) != Highlights->end());
  }

  S << "}" << std::endl;
  S.close();
#if 0
  std::cout << "### dumped CFG to " << fname << std::endl;
#endif
}

void dumpCFG(llvm::Function &F, const char *Filename) {
  dumpCFG(F, std::string(Filename));
}

void dumpCFG(llvm::Function &F) {
  dumpCFG(F, "");
}

void viewCFG(llvm::Function &F) {
  std::string Filename = std::string("pocl_cfg.") + F.getName().str() + ".dot";
  dumpCFG(F, Filename);
  llvm::DisplayGraph(Filename, /*wait=*/false, GraphProgram::DOT);
}

bool chopBBs(llvm::Function &F, llvm::Pass &) {
  bool fchanged = false;
  const int MAX_INSTRUCTIONS_PER_BB = 70;
  do {
    fchanged = false;
    for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
      BasicBlock *b = &*i;
      
      if (b->size() > MAX_INSTRUCTIONS_PER_BB + 1)
        {
          int count = 0;
          BasicBlock::iterator splitPoint = b->begin();
          while (count < MAX_INSTRUCTIONS_PER_BB || isa<PHINode>(splitPoint))
            {
              ++splitPoint;
              ++count;
            }
          SplitBlock(b, &*splitPoint);
          fchanged = true;
          break;
        }
    }  

  } while (fchanged);
  return fchanged;
}

void PoCLCFGPrinter::dumpModule(llvm::Module &M) {
  for (llvm::Function &F : M) {
    std::string Name;
    if (F.hasName())
      Name += F.getName();
    else
      Name += "anonymous_func";
    Name = Prefix + Name + ".dot";
    // TODO somehow supply regions/highlights
    dumpCFG(F, Name, nullptr, nullptr);
  }
}

llvm::PreservedAnalyses PoCLCFGPrinter::run(llvm::Module &M,
                                            llvm::ModuleAnalysisManager &AM) {
  dumpModule(M);
  return PreservedAnalyses::all();
}

void PoCLCFGPrinter::registerWithPB(llvm::PassBuilder &PB) {
  PB.registerPipelineParsingCallback(
      [](::llvm::StringRef Name, ::llvm::ModulePassManager &MPM,
         llvm::ArrayRef<::llvm::PassBuilder::PipelineElement>) {
        if (Name == "print<pocl-cfg>") {
          MPM.addPass(PoCLCFGPrinter(llvm::errs()));
          return true;
        }
        // the string X in "print<pocl-cfg;X>" will be passed to constructor;
        // this can be used to run multiple times and dump to different files
        if (Name.consume_front("print<pocl-cfg;") && Name.consume_back(">")) {
          MPM.addPass(PoCLCFGPrinter(llvm::errs(), Name));
          return true;
        }

        return false;
      });
}

} // namespace pocl
