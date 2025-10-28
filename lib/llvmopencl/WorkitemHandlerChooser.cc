// Header for work-item handler choosing.
//
// Copyright (c) 2012 Pekka Jääskeläinen / TUT
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "LLVMUtils.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
POP_COMPILER_DIAGS

#include <iostream>

namespace pocl {

using namespace llvm;

static llvm::StringRef asString(WorkitemHandlerType WIH) {
  switch (WIH) {
  default:
    assert(!"Missing WIH type to string mapping!");
    LLVM_FALLTHROUGH;
  case WorkitemHandlerType::INVALID:
    return "invalid";

  case WorkitemHandlerType::LOOPS:
    return "loopvec";
  case WorkitemHandlerType::CBS:
    return "cbs";
  case WorkitemHandlerType::FIBER:
    return "fiber";
  }
}

// This is a counterpart for asString(WorkitemHandlerType). The behavior is
// undefined if 'Value' is not something returned by it.
WorkitemHandlerType parseFromString(llvm::StringRef Value) {
  static const llvm::StringMap<WorkitemHandlerType> Map({
      {"invalid", WorkitemHandlerType::INVALID},
      {"loopvec", WorkitemHandlerType::LOOPS},
      {"cbs", WorkitemHandlerType::CBS},
      {"fiber", WorkitemHandlerType::FIBER},
  });

  assert(Map.contains(Value));
  return Map.at(Value);
}

static const char *PoclWGMethodAttrName = "pocl-wg-method";

WorkitemHandlerType
getWorkitemHandler(Function &F, llvm::PostDominatorTree &PDT, LoopInfo &LI) {

  // The work-group method decision is made sticky because PoCL passes may make
  // transfomations that are incompatible with other WG methods.
  if (F.hasFnAttribute(PoclWGMethodAttrName))
    return parseFromString(
        F.getFnAttribute(PoclWGMethodAttrName).getValueAsString());

  WorkitemHandlerType Result = WorkitemHandlerType::INVALID;

  std::string method = "auto";
  if (getenv("POCL_WORK_GROUP_METHOD") != NULL) {
    method = getenv("POCL_WORK_GROUP_METHOD");
    if (method == "loops" || method == "workitemloops" || method == "loopvec")
      Result = WorkitemHandlerType::LOOPS;
    else if (method == "cbs")
      // CBS is deprecated for now at least until it's fixed.
      Result = WorkitemHandlerType::FIBER;
    else if (method == "fiber")
      Result = WorkitemHandlerType::FIBER;
    else if (method != "auto") {
      method = "auto";
    }
  }

  if (method == "auto") {
    // To be replaced with heuristics in DeSPMD.
    Result = WorkitemHandlerType::LOOPS;
  }

  if (Result == WorkitemHandlerType::LOOPS &&
      !wiloops::canHandleKernel(F, PDT, LI)) {
    Result = WorkitemHandlerType::FIBER;
  }

  auto WGMethodAttr =
      Attribute::get(F.getContext(), PoclWGMethodAttrName, asString(Result));

  F.setAttributes(
      F.getAttributes().addFnAttribute(F.getContext(), WGMethodAttr));

  return Result;
}

} // namespace pocl
