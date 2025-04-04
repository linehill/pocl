/* pocl-level0.c - driver for LevelZero Compute API devices.

   Copyright (c) 2022-2023 Michal Babej / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/


#include "common.h"
#include "common_driver.h"
#include "common_utils.h"
#include "devices.h"
#include "pocl_cl.h"
#include "utlist.h"

#include "pocl-level0.h"
#include "pocl_builtin_kernels.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_hash.h"
#include "pocl_llvm.h"
#include "pocl_local_size.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "pocl_spirv_utils.hh"

#include <ze_api.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

#ifdef ENABLE_LEVEL0_OCLOC
#include <ocloc_api.h>
#endif

#include "spirv_parser.hh"

using namespace SPIRVParser;

#include "level0-compilation.hh"
#include "level0-driver.hh"

// Disabled for now
#undef ENABLE_NPU

using namespace pocl;

static std::vector<Level0DriverUPtr> L0DriverInstances;
static unsigned TotalL0Devices;

static void pocl_level0_local_size_optimizer(cl_device_id Dev, cl_kernel Ker,
                                             unsigned DeviceI,
                                             size_t MaxGroupSize,
                                             size_t GlobalX, size_t GlobalY,
                                             size_t GlobalZ, size_t *LocalX,
                                             size_t *LocalY, size_t *LocalZ) {
  // for NPU, return all 1s
  if (Dev->type != CL_DEVICE_TYPE_GPU) {
    *LocalX = *LocalY = *LocalZ = 1;
    return;
  }
  if (GlobalX == 1 && GlobalY == 1 && GlobalZ == 1) {
    *LocalX = *LocalY = *LocalZ = 1;
    return;
  }

  assert(Ker->data[DeviceI] != nullptr);
  cl_program Prog = Ker->program;
  assert(Prog->data[DeviceI] != nullptr);

  Level0KernelBase *L0Kernel = (Level0KernelBase *)Ker->data[DeviceI];
  ze_kernel_handle_t KernelH = L0Kernel->getAnyCreated();

  uint32_t SuggestedX = 0;
  uint32_t SuggestedY = 0;
  uint32_t SuggestedZ = 0;

  if (KernelH) {
    ze_result_t Res =
        zeKernelSuggestGroupSize(KernelH, GlobalX, GlobalY, GlobalZ,
                                 &SuggestedX, &SuggestedY, &SuggestedZ);
    if (Res == ZE_RESULT_SUCCESS) {
      *LocalX = SuggestedX;
      *LocalY = SuggestedY;
      *LocalZ = SuggestedZ;
      return;
    } else {
      POCL_MSG_PRINT_LEVEL0("zeKernelSuggestGroupSize FAILED: %0x\n",
                            (unsigned)Res);
    }
  } else {
    POCL_MSG_PRINT_LEVEL0(
        "pocl_level0_local_size_optimizer : HKernel == nullptr\n");
  }

FINISH:
  pocl_default_local_size_optimizer(Dev, Ker, DeviceI, MaxGroupSize, GlobalX,
                                    GlobalY, GlobalZ, LocalX, LocalY, LocalZ);
}

static int pocl_level0_verify_ndrange_sizes(const size_t *GlobalOffsets,
                                            const size_t *GlobalWsize,
                                            const size_t *LocalWsize) {
  size_t GlobalMax = GlobalWsize[0] | GlobalWsize[1] | GlobalWsize[2];
  POCL_RETURN_ERROR_ON((GlobalMax > UINT32_MAX), CL_INVALID_GLOBAL_WORK_SIZE,
                       "Level0 driver does not support "
                       "Global Work Sizes >32bit \n");

  size_t OffsetMax = GlobalOffsets[0] | GlobalOffsets[1] | GlobalOffsets[2];
  POCL_RETURN_ERROR_ON((OffsetMax > UINT32_MAX), CL_INVALID_GLOBAL_OFFSET,
                       "Level0 driver does not support "
                       "Global Offset Sizes >32bit \n");

  return CL_SUCCESS;
}

static cl_int pocl_level0_post_init(struct pocl_device_ops *ops) {

#ifdef USE_LLVM_SPIRV_TARGET
  pocl_llvm_initialize_spirv_ext_option();
#endif

  // TODO currently only works with two drivers
  if (L0DriverInstances.size() != 2)
    return CL_SUCCESS;

  Level0Device *ExportDevice = nullptr;
  std::vector<Level0Device *> ImportDevices;
  Level0Driver *D0 = L0DriverInstances[0].get();
  Level0Driver *D1 = L0DriverInstances[1].get();
  Level0Driver *ExDrv = nullptr, *ImDrv = nullptr;
  if (D0->getNumDevices() == 0 || D1->getNumDevices() == 0) {
    return CL_SUCCESS;
  }

  if ((ExportDevice = D0->getExportDevice()) &&
      D0->getImportDevices(ImportDevices, ExportDevice) &&
      D1->getImportDevices(ImportDevices, nullptr) &&
      TotalL0Devices == (ImportDevices.size() + 1)) {
    ExDrv = D0;
    ImDrv = D1;
  } else if ((ExportDevice = D1->getExportDevice()) &&
             D1->getImportDevices(ImportDevices, ExportDevice) &&
             D0->getImportDevices(ImportDevices, nullptr) &&
             TotalL0Devices == (ImportDevices.size() + 1)) {
    ExDrv = D1;
    ImDrv = D0;
  } else {
    return CL_SUCCESS;
  }
  POCL_MSG_PRINT_LEVEL0("Found both Import&Export devices, "
                        "creating shared memory allocator\n");

  assert(ExportDevice != nullptr);
  assert(!ImportDevices.empty());

  auto ExClDev = ExportDevice->getClDev();
  auto ImClDev = ImportDevices[0]->getClDev();
  POCL_MSG_PRINT_LEVEL0("ExportDev: %s\nImportDevs[0]: %s\n",
                        ExClDev->short_name, ImClDev->short_name);
  Level0DMABufAllocatorSPtr SharedAlloc{
      new Level0DMABufAllocator(ExportDevice, ImportDevices)};
  ExportDevice->assignAllocator(SharedAlloc);
  for (auto &Dev : ImportDevices) {
    Dev->assignAllocator(SharedAlloc);
  }
  return CL_SUCCESS;
}

cl_int
pocl_level0_create_finalized_command_buffer(cl_device_id Dev,
                                            cl_command_buffer_khr CmdBuf);
cl_int pocl_level0_free_command_buffer(cl_device_id Dev,
                                       cl_command_buffer_khr CmdBuf);

void pocl_level0_init_device_ops(struct pocl_device_ops *Ops) {
  Ops->device_name = "level0";

  Ops->probe = pocl_level0_probe;
  Ops->post_init = pocl_level0_post_init;
  Ops->init = pocl_level0_init;
  Ops->uninit = pocl_level0_uninit;
  Ops->reinit = pocl_level0_reinit;

  // used by the NPU device to execute all commands except NDRangeKernel
  // GPU devices don't use these, they use L0 API zeCmdListAppend...
  Ops->read = pocl_driver_read;
  Ops->read_rect = pocl_driver_read_rect;
  Ops->write = pocl_driver_write;
  Ops->write_rect = pocl_driver_write_rect;
  Ops->copy = pocl_driver_copy;
  Ops->copy_with_size = pocl_driver_copy_with_size;
  Ops->copy_rect = pocl_driver_copy_rect;
  Ops->memfill = pocl_driver_memfill;
  Ops->map_mem = pocl_driver_map_mem;
  Ops->unmap_mem = pocl_driver_unmap_mem;

  Ops->can_migrate_d2d = nullptr;
  Ops->migrate_d2d = nullptr;

  Ops->run = nullptr;
  Ops->run_native = nullptr;

  Ops->get_mapping_ptr = pocl_level0_get_mapping_ptr;
  Ops->free_mapping_ptr = pocl_level0_free_mapping_ptr;

  Ops->compute_local_size = pocl_level0_local_size_optimizer;
  Ops->verify_ndrange_sizes = pocl_level0_verify_ndrange_sizes;

  Ops->alloc_mem_obj = pocl_level0_alloc_mem_obj;
  Ops->free = pocl_level0_free;
  Ops->svm_free = pocl_level0_svm_free;
  Ops->svm_alloc = pocl_level0_svm_alloc;
  Ops->usm_alloc = pocl_level0_usm_alloc;
  Ops->usm_free = pocl_level0_usm_free;
  Ops->usm_free_blocking = pocl_level0_usm_free_blocking;

  Ops->build_source = pocl_level0_build_source;
  Ops->build_binary = pocl_level0_build_binary;
  Ops->link_program = pocl_level0_link_program;
  Ops->free_program = pocl_level0_free_program;
  Ops->setup_metadata = pocl_level0_setup_metadata;
  Ops->supports_binary = pocl_level0_supports_binary;
  Ops->build_poclbinary = pocl_level0_build_poclbinary;
  Ops->build_builtin = pocl_level0_build_builtin;
  Ops->compile_kernel = nullptr;
  Ops->supports_dbk = pocl_level0_supports_dbk;
  Ops->build_defined_builtin = pocl_level0_build_builtin;
  Ops->create_kernel = pocl_level0_create_kernel;
  Ops->free_kernel = pocl_level0_free_kernel;
  Ops->init_build = pocl_level0_init_build;

  Ops->join = pocl_level0_join;
  Ops->submit = pocl_level0_submit;
  Ops->broadcast = pocl_broadcast;
  Ops->notify = pocl_level0_notify;
  Ops->flush = pocl_level0_flush;
  Ops->build_hash = pocl_level0_build_hash;

  /* TODO get timing data from level0 API */
  /* ops->get_timer_value = pocl_level0_get_timer_value; */

  Ops->wait_event = pocl_level0_wait_event;
  Ops->notify_event_finished = pocl_level0_notify_event_finished;
  Ops->notify_cmdq_finished = pocl_level0_notify_cmdq_finished;
  Ops->free_event_data = pocl_level0_free_event_data;
  Ops->update_event = pocl_level0_update_event;

  Ops->init_queue = pocl_level0_init_queue;
  Ops->free_queue = pocl_level0_free_queue;

  Ops->create_sampler = pocl_level0_create_sampler;
  Ops->free_sampler = pocl_level0_free_sampler;

  Ops->get_device_info_ext = pocl_level0_get_device_info_ext;
  Ops->get_subgroup_info_ext = pocl_level0_get_subgroup_info_ext;
  Ops->set_kernel_exec_info_ext = pocl_level0_set_kernel_exec_info_ext;
  Ops->get_synchronized_timestamps = pocl_driver_get_synchronized_timestamps;

  // Ops->create_finalized_command_buffer =
  //     pocl_level0_create_finalized_command_buffer;
  // Ops->free_command_buffer = pocl_level0_free_command_buffer;
}


static int readProgramSpv(cl_program Program, cl_uint DeviceI,
                          const char *ProgramSpvPath) {
  /* Read binaries from program.spv to memory */
  if (Program->program_il_size == 0) {
    assert(ProgramSpvPath);
    assert(Program->program_il == nullptr);
    uint64_t Size = 0;
    char *Binary = nullptr;
    int Res = pocl_read_file(ProgramSpvPath, &Binary, &Size);
    POCL_RETURN_ERROR_ON((Res != 0), CL_BUILD_PROGRAM_FAILURE,
                         "Failed to read binaries from program.spv to "
                         "memory: %s\n",
                         ProgramSpvPath);
    Program->program_il = Binary;
    Program->program_il_size = Size;
  }
  return CL_SUCCESS;
}

char *pocl_level0_build_hash(cl_device_id ClDevice) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
  char *Res = (char *)malloc(32);
  if (ClDevice->type == CL_DEVICE_TYPE_GPU ||
      ClDevice->type == CL_DEVICE_TYPE_CPU) {
      snprintf(Res, 32, "pocl-level0-spirv-%x", Device->getIPVersion());
    // Intel FPGA Emulation uses device type Accelerator
  } else if (ClDevice->type == CL_DEVICE_TYPE_ACCELERATOR) {
    snprintf(Res, 32, "pocl-level0-fpga");
  } else if (ClDevice->type == CL_DEVICE_TYPE_CUSTOM) {
    snprintf(Res, 32, "pocl-level0-vpu");
  } else {
    snprintf(Res, 32, "pocl-level0-unknown");
  }
  return Res;
}

unsigned int pocl_level0_probe(struct pocl_device_ops *Ops) {
  int EnvCount = pocl_device_get_env_count(Ops->device_name);

  // POCL_DEVICES is set but doesn't contain level0 -> return 0
  // POCL_DEVICES is unset -> continue
  if (EnvCount == 0) {
    return 0;
  }

  ze_result_t Res = zeInit(0);
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("zeInit FAILED\n");
    return 0;
  }

  uint32_t DriverCount = 64;
  ze_driver_handle_t DrvHandles[64];
  Res = zeDriverGet(&DriverCount, DrvHandles);
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_ERR("zeDriverGet FAILED\n");
    return 0;
  }

  for (unsigned I = 0; I < DriverCount; ++I) {
    // workaround for what appears to be a bug
    if (DrvHandles[I] == nullptr)
      continue;
    L0DriverInstances.emplace_back(new Level0Driver(DrvHandles[I]));
  }
  for (auto &Level0Driver : L0DriverInstances) {
    TotalL0Devices += Level0Driver->getNumDevices();
  }
  if (EnvCount > 0 && EnvCount < TotalL0Devices) {
    TotalL0Devices = EnvCount;
    POCL_MSG_PRINT_LEVEL0("LevelZero Probe: limiting devices to %u\n",
                          TotalL0Devices);
  }

  POCL_MSG_PRINT_LEVEL0("LevelZero Probe devices: %u\n", TotalL0Devices);
  return TotalL0Devices;
}

cl_int pocl_level0_init(unsigned J, cl_device_id ClDevice,
                        const char *Parameters) {
  POCL_MSG_PRINT_LEVEL0("Initializing device %u\n", J);
  assert(J < TotalL0Devices);

  Level0Device *Device = nullptr;
  unsigned TempJ = J;
  Level0Driver *Drv = nullptr;
  for (unsigned I = 0; I < L0DriverInstances.size(); ++I) {
    unsigned Num = L0DriverInstances[I]->getNumDevices();
    if (TempJ < Num) {
      Drv = L0DriverInstances[I].get();
      break;
    } else {
      TempJ -= Num;
    }
  }
  assert(Drv != nullptr);
  Device = Drv->createDevice(TempJ, ClDevice, Parameters);

  if (Device == nullptr) {
    POCL_MSG_ERR("createdevice failed\n");
    return CL_FAILED;
  }

  ClDevice->data = (void *)Device;

  return CL_SUCCESS;
}

cl_int pocl_level0_uninit(unsigned J, cl_device_id ClDevice) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
  return CL_SUCCESS;
}

cl_int pocl_level0_reinit(unsigned J, cl_device_id ClDevice, const char *parameters) {
  return CL_SUCCESS;
}

static void convertProgramBcPathToSpv(char *ProgramBcPath,
                                      char *ProgramSpvPath) {
  strncpy(ProgramSpvPath, ProgramBcPath, POCL_MAX_PATHNAME_LENGTH);
  size_t Len = strlen(ProgramBcPath);
  assert(Len > 3);
  Len -= 2;
  ProgramSpvPath[Len] = 0;
  strncat(ProgramSpvPath, "spv", POCL_MAX_PATHNAME_LENGTH);
}

static constexpr unsigned DefaultCaptureSize = 128 * 1024;

// TODO move to pocl_run_command.c
static int runAndAppendOutputToBuildLog(cl_program Program, unsigned DeviceI,
                                        const char **Args) {
  int Errcode = CL_SUCCESS;

  char *CapturedOutput = nullptr;
  size_t CaptureCapacity = 0;

  CapturedOutput = (char *)malloc(DefaultCaptureSize);
  POCL_RETURN_ERROR_ON((CapturedOutput == nullptr), CL_OUT_OF_HOST_MEMORY,
                       "Error while allocating temporary memory\n");
  CaptureCapacity = (DefaultCaptureSize) - 1;
  CapturedOutput[0] = 0;
  char *SavedCapturedOutput = CapturedOutput;

  std::string CommandLine;
  unsigned I = 0;
  while (Args[I] != nullptr) {
    CommandLine += " ";
    CommandLine += Args[I];
    ++I;
  }
  POCL_MSG_PRINT_LEVEL0("launching command: \n#### %s\n", CommandLine.c_str());

  std::string LaunchMsg;
  LaunchMsg.append("Output of ");
  LaunchMsg.append(Args[0]);
  LaunchMsg.append(":\n");
  if (LaunchMsg.size() < CaptureCapacity) {
    strncat(CapturedOutput, LaunchMsg.c_str(), CaptureCapacity);
    CapturedOutput += LaunchMsg.size();
    CaptureCapacity -= LaunchMsg.size();
  }

#ifdef HAVE_FORK
  Errcode =
      pocl_run_command_capture_output(CapturedOutput, &CaptureCapacity, Args);
  if (CaptureCapacity > 0) {
    CapturedOutput[CaptureCapacity] = 0;
  }

  pocl_append_to_buildlog(Program, DeviceI, SavedCapturedOutput,
                          strlen(SavedCapturedOutput));
#else
  POCL_MSG_WARN("Running a command with output capture is requested which"
                " is not implemented on this platform. Will run the command"
                " without capture.");
  Errcode = pocl_run_command(Args);
  char Msg[] = "UNIMPLEMENTED: pocl_run_command_capture_output for this"
               " platform.";
  pocl_append_to_buildlog(Program, DeviceI, Msg, strlen(Msg));
#endif
  return Errcode;
}

#ifdef HAVE_SPIRV_LINK
static int linkWithSpirvLink(cl_program Program, cl_uint DeviceI,
                             char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH],
                             std::vector<std::string> &SpvBinaryPaths,
                             int CreateLibrary) {
  std::vector<std::string> CompilationArgs;
  std::vector<const char *> CompilationArgs2;

  CompilationArgs.push_back(pocl_get_path("SPIRV_LINK", SPIRV_LINK));
  if (CreateLibrary != 0) {
    CompilationArgs.push_back("--create-library");
  }
  // allow linking of SPIR-V modules with different version
#ifdef SPIRV_LINK_HAS_USE_HIGHEST_VERSION
  CompilationArgs.push_back("--use-highest-version");
#endif
  CompilationArgs.push_back("-o");
  CompilationArgs.push_back(ProgramSpvPathTemp);
  for (auto &Path : SpvBinaryPaths) {
    CompilationArgs.push_back(Path);
  }
  CompilationArgs2.resize(CompilationArgs.size() + 1);
  for (unsigned i = 0; i < CompilationArgs.size(); ++i)
    CompilationArgs2[i] = (char *)CompilationArgs[i].data();
  CompilationArgs2[CompilationArgs.size()] = nullptr;

  int Err =
      runAndAppendOutputToBuildLog(Program, DeviceI, CompilationArgs2.data());
  POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                       "spirv-link exited with nonzero code\n");
  POCL_RETURN_ERROR_ON(!pocl_exists(ProgramSpvPathTemp),
                       CL_LINK_PROGRAM_FAILURE,
                       "spirv-link failed (output file does not exist)\n");
  return CL_SUCCESS;
}
#endif

static int runLLVMOpt(cl_program Program, cl_uint DeviceI,
                      char ProgramBcPathTemp[POCL_MAX_PATHNAME_LENGTH]) {
#ifdef HAVE_LLVM_OPT
  const char *L0passes =
      pocl_get_string_option("POCL_LEVEL0_LINK_OPT", nullptr);
  if (L0passes == nullptr)
    return CL_SUCCESS;

  char ProgramBcOldPathTemp[POCL_MAX_PATHNAME_LENGTH];

  std::vector<std::string> CompilationArgs;
  std::vector<const char *> CompilationArgs2;

  CompilationArgs.push_back(pocl_get_path("LLVM_OPT", LLVM_OPT));
  CompilationArgs.push_back(L0passes);
  strcpy(ProgramBcOldPathTemp, ProgramBcPathTemp);
  strncat(ProgramBcPathTemp, ".opt.bc", POCL_MAX_PATHNAME_LENGTH);
  CompilationArgs.push_back("-o");
  CompilationArgs.push_back(ProgramBcPathTemp);
  CompilationArgs.push_back(ProgramBcOldPathTemp);

  CompilationArgs2.resize(CompilationArgs.size() + 1);
  for (unsigned i = 0; i < CompilationArgs.size(); ++i)
    CompilationArgs2[i] = (char *)CompilationArgs[i].data();
  CompilationArgs2[CompilationArgs.size()] = nullptr;

  int Err =
      runAndAppendOutputToBuildLog(Program, DeviceI, CompilationArgs2.data());
  POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                       "llvm-opt exited with nonzero code\n");
  POCL_RETURN_ERROR_ON(!pocl_exists(ProgramBcPathTemp),
                       CL_BUILD_PROGRAM_FAILURE,
                       "llvm-opt failed to produce output file\n");
#endif
  return CL_SUCCESS;
}

static int linkWithLLVMLink(cl_program Program, cl_uint DeviceI,
                            char ProgramBcPathTemp[POCL_MAX_PATHNAME_LENGTH],
                            char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH],
                            std::vector<void *> &LLVMIRBinaries,
                            int CreateLibrary) {

  cl_device_id Dev = Program->devices[DeviceI];
  Level0Device *Device = static_cast<Level0Device *>(Dev->data);

  void *DestLLVMIR = nullptr;
  int Err = pocl_llvm_link_multiple_modules(Program, DeviceI, ProgramBcPathTemp,
                                            LLVMIRBinaries.data(),
                                            LLVMIRBinaries.size());

  POCL_RETURN_ERROR_ON((Err != CL_SUCCESS), CL_LINK_PROGRAM_FAILURE,
                       "llvm::link failed to link all modules\n");
  POCL_RETURN_ERROR_ON(!pocl_exists(ProgramBcPathTemp), CL_LINK_PROGRAM_FAILURE,
                       "llvm::link failed to "
                       "produce output file\n");

  Err = runLLVMOpt(Program, DeviceI, ProgramBcPathTemp);
  if (Err != CL_SUCCESS)
    return Err;

  Err = pocl_convert_bitcode_to_spirv(ProgramBcPathTemp, nullptr, 0, Program,
                                      DeviceI, Dev->supported_spirv_extensions,
                                      ProgramSpvPathTemp, nullptr, nullptr,
                                      Device->getSupportedSpvVersion());

  POCL_RETURN_ERROR_ON((Err != 0), CL_LINK_PROGRAM_FAILURE,
                       "llvm IR -> SPIRV conversion failed\n");

  return CL_SUCCESS;
}

int pocl_level0_build_source(cl_program Program, cl_uint DeviceI,
                             cl_uint NumInputHeaders,
                             const cl_program *InputHeaders,
                             const char **HeaderIncludeNames, int LinkProgram) {
  cl_device_id Dev = Program->devices[DeviceI];
  Level0Device *Device = (Level0Device *)Dev->data;
  int Err = CL_SUCCESS;

  if (Dev->compiler_available == CL_FALSE ||
      Dev->linker_available == CL_FALSE) {
    POCL_RETURN_ERROR_ON(1, CL_BUILD_PROGRAM_FAILURE,
                         "This device cannot build from sources\n");
  }

#ifdef ENABLE_LEVEL0_OCLOC
    POCL_MSG_PRINT_LLVM("OCLOC: building from sources for device %d\n", DeviceI);

    uint32_t numSources = 1;
    const uint8_t *dataSources = reinterpret_cast<const uint8_t *>(Program->source);
    uint64_t lenSources = strlen(Program->source) + 1;
    const char *nameSources = "main.cl";

    uint32_t numInputHeaders = NumInputHeaders;
    std::vector<const uint8_t *> dataInputHeaders;
    std::vector<uint64_t> lenInputHeaders;
    std::vector<std::string> inputHeaderNames;
    std::vector<const char *> nameInputHeaders;
    for (cl_uint  i = 0; i < NumInputHeaders; ++i) {
        dataInputHeaders.push_back((const uint8_t *)InputHeaders[i]->source);
        lenInputHeaders.push_back(strlen(InputHeaders[i]->source));
        std::string Name = "header_" + std::to_string(i) + ".h";
        inputHeaderNames.push_back(Name);
        nameInputHeaders.push_back(inputHeaderNames[i].c_str());
    }
    uint32_t numOutputs = 0;
    uint8_t **dataOutputs = nullptr;
    uint64_t *lenOutputs = nullptr;
    char **nameOutputs = nullptr;

    //std::stringstream ConvStream;
    //ConvStream << std::setfill('0') << std::setw(sizeof(uint32_t)*2) << std::hex << Device->getIPVersion();
    //std::string DevIParg(ConvStream.str());
    std::string DevIParg = std::to_string(Device->getIPVersion());
    POCL_MSG_PRINT_LLVM("OCLOC: Using device IP arg: %s\n", DevIParg.c_str());

    char OclocCacheDir[POCL_MAX_PATHNAME_LENGTH] = {0};
    pocl_cache_get_ocloc_cache_dir(OclocCacheDir);
    pocl_mkdir_p(OclocCacheDir);

    // for cache hash. TODO: this should be done via preprocessing
    std::string SourcePlusOpts(Program->source);
    if (Program->compiler_options)
        SourcePlusOpts.append(Program->compiler_options);
    for (cl_uint  i = 0; i < NumInputHeaders; ++i) {
        SourcePlusOpts.append(InputHeaders[i]->source);
    }

    char ProgramBcPathTemp[POCL_MAX_PATHNAME_LENGTH] = {0};
    char ProgramBcPath[POCL_MAX_PATHNAME_LENGTH] = {0};
    char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH] = {0};
    char ProgramSpvPath[POCL_MAX_PATHNAME_LENGTH] = {0};
    pocl_cache_create_program_cachedir(Program, DeviceI,
                                       SourcePlusOpts.c_str(),
                                       SourcePlusOpts.size(),
                                       ProgramBcPath);
    convertProgramBcPathToSpv(ProgramBcPath, ProgramSpvPath);

    if (pocl_exists(ProgramBcPath) && pocl_exists(ProgramSpvPath)) {
        Err = readProgramSpv(Program, DeviceI, ProgramSpvPath);
        if (Err)
            return Err;
        char *Content = nullptr;
        uint64_t ContentSize = 0;
        Err = pocl_read_file(ProgramBcPath, &Content, &ContentSize);
        POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                             "Failed to read program.bc in %s\n", ProgramBcPath);
        Program->binaries[DeviceI] = (unsigned char *)Content;
        Program->binary_sizes[DeviceI] = ContentSize;

    } else {

    const char *Args[] = {
        "compile",
        "-q",
        "-spv_only",
        "-device",
        DevIParg.c_str(),
        "-file",
        nameSources,
        "-allow_caching",
        "-cache_dir",
        OclocCacheDir,
        nullptr, // "-options",
        nullptr // Program->compiler_options,
    };
    unsigned int numArgs = 10;
    if (Program->compiler_options) {
        Args[10] = "-options";
        Args[11] = Program->compiler_options;
        numArgs = 12;
    }

    int Errcode = oclocInvoke(numArgs, Args,
                              numSources, &dataSources, &lenSources, &nameSources,
                              numInputHeaders, dataInputHeaders.data(), lenInputHeaders.data(), nameInputHeaders.data(),
                              &numOutputs, &dataOutputs, &lenOutputs, &nameOutputs);

    unsigned char *spirV = nullptr;
    size_t spirVlen = 0;
    const char *log = nullptr;
    size_t logLen = 0;
    for (unsigned int i = 0; i < numOutputs; ++i) {
        std::string spvExtension = ".spv";
        std::string logFileName = "stdout.log";
        auto nameLen = strlen(nameOutputs[i]);
        if ((nameLen > spvExtension.size()) && (strstr(&nameOutputs[i][nameLen - spvExtension.size()], spvExtension.c_str()) != nullptr)) {
            spirV = dataOutputs[i];
            spirVlen = lenOutputs[i];
            dataOutputs[i] = nullptr;
            lenOutputs[i] = 0;
        } else if ((nameLen >= logFileName.size()) && (strstr(nameOutputs[i], logFileName.c_str()) != nullptr)) {
            log = reinterpret_cast<const char *>(dataOutputs[i]);
            logLen = lenOutputs[i];
            dataOutputs[i] = nullptr;
            lenOutputs[i] = 0;
            break;
        }
    }
    // TODO check if this is OK
    oclocFreeOutput(&numOutputs, &dataOutputs, &lenOutputs, &nameOutputs);

    if (logLen > 0) {
        strncpy(Program->main_build_log, log, MAIN_PROGRAM_LOG_SIZE);
    }
    if (Errcode != 0) {
        if (logLen == 0) {
            std::string Log = "Ocloc compilation failed: " + std::to_string(Errcode) + "\n";
            strncpy(Program->main_build_log, Log.c_str(), MAIN_PROGRAM_LOG_SIZE);
        }
        return CL_BUILD_PROGRAM_FAILURE;
    }
    Program->program_il = reinterpret_cast<char *>(spirV);
    Program->program_il_size = spirVlen;
    assert(Program->program_il != nullptr);
    assert(Program->program_il_size > 0);

    char *OutputBinary = nullptr;
    uint64_t OutputBinarySize = 0;
    Err = pocl_convert_spirv_to_bitcode(
        ProgramSpvPathTemp, Program->program_il, Program->program_il_size,
        Program, DeviceI, Dev->supported_spirv_extensions, ProgramBcPathTemp,
        &OutputBinary, &OutputBinarySize);
    POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                         "failed to convert SPIRV -> BC\n");
    Program->binaries[DeviceI] = (unsigned char *)OutputBinary;
    Program->binary_sizes[DeviceI] = OutputBinarySize;

    pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
    pocl_rename(ProgramBcPathTemp, ProgramBcPath);
    }

    assert(Program->binaries[DeviceI] != nullptr);
    assert(Program->binary_sizes[DeviceI] != 0);
    pocl_llvm_read_program_llvm_irs(Program, DeviceI, nullptr);

    if (LinkProgram != 0) {
        pocl_llvm_recalculate_gvar_sizes(Program, DeviceI);
        return Device->createSpirvProgram(Program, DeviceI);
    } else {
        // only final (linked) programs have  ZE module
        assert(Program->data[DeviceI] == nullptr);
        return CL_SUCCESS;
    }

#elif defined(ENABLE_LLVM)
  POCL_MSG_PRINT_LLVM("building from sources for device %d\n", DeviceI);

  // last arg is 0 because we never link with Clang, let the spirv-link and
  // level0 do the linking
  int Errcode = pocl_llvm_build_program(Program, DeviceI, NumInputHeaders,
                                        InputHeaders, HeaderIncludeNames, 0);
  POCL_RETURN_ERROR_ON((Errcode != CL_SUCCESS), CL_BUILD_PROGRAM_FAILURE,
                       "Failed to build program from source\n");

  char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH] = {0};
  char ProgramBcPath[POCL_MAX_PATHNAME_LENGTH];
  char ProgramSpvPath[POCL_MAX_PATHNAME_LENGTH];

  pocl_cache_program_bc_path(ProgramBcPath, Program, DeviceI);
  pocl_cache_program_spv_path(ProgramSpvPath, Program, DeviceI);

  // result of pocl_llvm_build_program
  assert(pocl_exists(ProgramBcPath));

  if (pocl_exists(ProgramSpvPath) != 0) {
    POCL_MSG_PRINT_LEVEL0("Found compiled SPIR-V in cache\n");
    readProgramSpv(Program, DeviceI, ProgramSpvPath);
  } else {
    char *OutputBinary = nullptr;
    uint64_t OutputBinarySize = 0;
    assert(Program->binaries[DeviceI] != nullptr);
    assert(Program->binary_sizes[DeviceI] > 0);

    Err = runLLVMOpt(Program, DeviceI, ProgramBcPath);
    if (Err != CL_SUCCESS)
      return Err;
    Err = pocl_convert_bitcode_to_spirv(
        nullptr, (char *)Program->binaries[DeviceI],
        Program->binary_sizes[DeviceI], Program, DeviceI,
        Dev->supported_spirv_extensions, ProgramSpvPathTemp, &OutputBinary,
        &OutputBinarySize, Device->getSupportedSpvVersion());
    POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                         "llvm-spirv exited with nonzero code\n");

    Program->program_il = OutputBinary;
    Program->program_il_size = OutputBinarySize;
    pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
  }

  assert(Program->program_il != nullptr);
  assert(Program->program_il_size > 0);
  assert(Program->binaries[DeviceI] != nullptr);
  assert(Program->binary_sizes[DeviceI] > 0);
  assert(Program->llvm_irs[DeviceI] != nullptr);

  if (LinkProgram) {
    pocl_llvm_recalculate_gvar_sizes(Program, DeviceI);
    return Device->createSpirvProgram(Program, DeviceI);
  } else {
    // only final (linked) programs have  ZE module
    assert(Program->data[DeviceI] == nullptr);
    return CL_SUCCESS;
  }

#else
  POCL_RETURN_ERROR_ON(1, CL_BUILD_PROGRAM_FAILURE,
                       "This device requires LLVM to build from sources\n");
#endif
}

/*
static bool isFileELF(const char *Binary, size_t Length) {
    return Length > 4 &&
           Binary[0] == 0x7f && Binary[1] == 0x45
           && Binary[2] == 0x4C && Binary[3] == 0x46;
}
*/

int pocl_level0_supports_binary(cl_device_id ClDev, size_t Length,
                                const char *Binary) {
  if (pocl_bitcode_is_spirv_execmodel_kernel(Binary, Length,
                                             ClDev->address_bits)) {
    // currently necessary to have LLVM for parsing the metadata
    // return 1 when SPIR-V parser is finished
    return ClDev->compiler_available == CL_TRUE &&
           ClDev->linker_available == CL_TRUE &&
           ClDev->num_ils_with_version > 0;
  }

  Level0Device *Device = (Level0Device *)ClDev->data;
  //if (isFileELF(Binary, Length)) {
  // TODO is this safe ?
  if (Device->supportsBinary(Binary, Length)) {
    POCL_MSG_WARN("Detected GPU binary\n");
    return 1;
  } else {
    POCL_MSG_WARN("Unknown binary - not ELF & not SPIRV\n");
    return 0;
  }
}

char *pocl_level0_init_build(void *Data) {
  // the -O0 helps to avoid a bunch of issues created by Clang's optimization
  // (issues for llvm-spirv translator)
  // * the freeze instruction
  // * the vector instructions (llvm.vector.reduce.add.v4i32)
  // "InvalidBitWidth: Invalid bit width in input: 63" - happens with
  // test_convert_type_X
  return strdup("-O0");
}

int pocl_level0_build_binary(cl_program Program, cl_uint DeviceI,
                             int LinkProgram, int SpirBuild) {
  cl_device_id Dev = Program->devices[DeviceI];
  Level0Device *Device = (Level0Device *)Dev->data;

  if (Dev->compiler_available == CL_FALSE ||
      Dev->linker_available == CL_FALSE) {
    POCL_RETURN_ERROR_ON(1, CL_BUILD_PROGRAM_FAILURE,
                         "This device cannot build binaries\n");
  }

  char ProgramBcPath[POCL_MAX_PATHNAME_LENGTH];
  char ProgramSpvPath[POCL_MAX_PATHNAME_LENGTH];
  char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH];
  ProgramSpvPathTemp[0] = 0;
  char ProgramBcPathTemp[POCL_MAX_PATHNAME_LENGTH];
  ProgramBcPathTemp[0] = 0;
  int Err = 0;

  if (Program->pocl_binaries[DeviceI] != nullptr) {
    /* we have pocl_binaries with BOTH SPIRV and IR Bitcode */

    pocl_cache_program_spv_path(ProgramSpvPath, Program, DeviceI);

    POCL_RETURN_ERROR_ON(
        (readProgramSpv(Program, DeviceI, ProgramSpvPath) != CL_SUCCESS),
        CL_BUILD_PROGRAM_FAILURE, "Could not read compiled program.spv at %s\n",
        ProgramSpvPath);

    // TODO is this really LLVM IR
    // program.bc should be read by clCreateProgramWithBinary
    assert(Program->binaries[DeviceI] != nullptr);
    assert(Program->binary_sizes[DeviceI] != 0);

  } else {
    // assert(Program->pocl_binaries[DeviceI] == nullptr);
    char *OutputBinary = nullptr;
    uint64_t OutputBinarySize = 0;

    if (Program->binaries[DeviceI] == nullptr) {
      /* we have only program_il, which is SPIR-V*/
      assert(Program->program_il != nullptr);
      assert(Program->program_il_size > 0);

#ifdef USE_LLVM_SPIRV_TARGET
      POCL_MSG_WARN("Level0: not converting SPIRV -> LLVM IR "
                    "with SPIRV backend\n");
#else
      Err = pocl_convert_spirv_to_bitcode(
          ProgramSpvPathTemp, Program->program_il, Program->program_il_size,
          Program, DeviceI, Dev->supported_spirv_extensions, ProgramBcPathTemp,
          &OutputBinary, &OutputBinarySize);
      POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                           "failed to compile SPV -> BC\n");
      Program->binaries[DeviceI] = (unsigned char *)OutputBinary;
      Program->binary_sizes[DeviceI] = OutputBinarySize;

      assert(Program->binaries[DeviceI] != nullptr);
      assert(Program->binary_sizes[DeviceI] != 0);
#endif
    } else {
      /* we have program->binaries[] which could be LLVM IR SPIR, or GPU binary */
      assert(Program->binaries[DeviceI] != nullptr);
      assert(Program->binary_sizes[DeviceI] > 0);

      bool TripleIsSpir64 =
          pocl_bitcode_is_triple((char *)Program->binaries[DeviceI],
                            Program->binary_sizes[DeviceI], "spir64-unknown");
      // bool IsELF = isFileELF((char *)Program->binaries[DeviceI],
      //                        Program->binary_sizes[DeviceI]);
      // POCL_RETURN_ERROR_ON((!TripleIsSpir64 && !IsELF), CL_BUILD_PROGRAM_FAILURE,
      //                      "the binary supplied to level0 driver is "
      //                      "not a recognized binary type\n");

      if (TripleIsSpir64) {
        // SPIRV
        Err = pocl_convert_bitcode_to_spirv(
          ProgramBcPathTemp, (char *)Program->binaries[DeviceI],
          Program->binary_sizes[DeviceI], Program, DeviceI,
          Dev->supported_spirv_extensions, ProgramSpvPathTemp, &OutputBinary,
          &OutputBinarySize, Device->getSupportedSpvVersion());
        POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                           "failed to compile BC -> SPV\n");
        Program->program_il = OutputBinary;
        Program->program_il_size = OutputBinarySize;
      } else {
        // GPU binary
        POCL_RETURN_ERROR_ON((LinkProgram == 0), CL_BUILD_PROGRAM_FAILURE,
                              "creating multi-part programs via GPU binaries"
                              " is not supported\n");
        Program->pocl_binaries_prefer_native[DeviceI] = CL_TRUE;
        // don't skip creating cache dir
        pocl_cache_create_program_cachedir(Program, DeviceI, (char *)Program->binaries[DeviceI],
                                           Program->binary_sizes[DeviceI], ProgramBcPath);
        convertProgramBcPathToSpv(ProgramBcPath, ProgramSpvPath);
        assert(Program->build_hash[DeviceI][0] != 0);

        return Device->createGPUBinaryProgram(Program, DeviceI);
      }
    }

    pocl_cache_create_program_cachedir(Program, DeviceI, Program->program_il,
                                       Program->program_il_size, ProgramBcPath);
    convertProgramBcPathToSpv(ProgramBcPath, ProgramSpvPath);
    pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
    pocl_rename(ProgramBcPathTemp, ProgramBcPath);

    assert(Program->program_il != nullptr);
    assert(Program->program_il_size > 0);
    assert(Program->binaries[DeviceI] != nullptr);
    assert(Program->binary_sizes[DeviceI] != 0);
  }

  if (LinkProgram != 0) {
    // for Metadata, read the Bitcode into LLVM::Module
    pocl_llvm_read_program_llvm_irs(Program, DeviceI, ProgramBcPath);
    pocl_llvm_recalculate_gvar_sizes(Program, DeviceI);
    return Device->createSpirvProgram(Program, DeviceI);
  } else {
    // only final (linked) programs have ZE module
    assert(Program->data[DeviceI] == nullptr);
    return CL_SUCCESS;
  }
}

int pocl_level0_link_program(cl_program Program, cl_uint DeviceI,
                             cl_uint NumInputPrograms,
                             const cl_program *InputPrograms,
                             int CreateLibrary) {
  cl_device_id Dev = Program->devices[DeviceI];
  Level0Device *Device = (Level0Device *)Dev->data;

  if (!Dev->compiler_available || !Dev->linker_available) {
    POCL_RETURN_ERROR_ON(1, CL_BUILD_PROGRAM_FAILURE,
                         "This device cannot link binaries\n");
  }

  char ProgramBcPath[POCL_MAX_PATHNAME_LENGTH];
  char ProgramSpvPath[POCL_MAX_PATHNAME_LENGTH];

  assert(Program->pocl_binaries[DeviceI] == nullptr);
  assert(Program->binaries[DeviceI] == nullptr);
  assert(Program->binary_sizes[DeviceI] == 0);

  std::vector<std::string> SpvBinaryPaths;
  std::vector<void *> LLVMIRBinaries;
  std::vector<char> SpvConcatBinary;

  cl_uint I;
  cl_uint ProgsHaveProgramLLVMIR = 0;
  for (I = 0; I < NumInputPrograms; I++) {
    assert(Dev == InputPrograms[I]->devices[DeviceI]);
    POCL_LOCK_OBJ(InputPrograms[I]);

    pocl_cache_program_spv_path(ProgramSpvPath, InputPrograms[I], DeviceI);
    assert(pocl_exists(ProgramSpvPath));
    SpvBinaryPaths.push_back(ProgramSpvPath);

    char *Spv = (char *)InputPrograms[I]->program_il;
    if (Spv == nullptr) {
      readProgramSpv(InputPrograms[I], DeviceI, ProgramSpvPath);
    }
    Spv = (char *)InputPrograms[I]->program_il;
    assert(Spv);
    size_t Size = InputPrograms[I]->program_il_size;
    assert(Size);
    SpvConcatBinary.insert(SpvConcatBinary.end(), Spv, Spv + Size);

    if (InputPrograms[I]->binary_sizes[DeviceI] > 0) {
      pocl_cache_program_bc_path(ProgramBcPath, InputPrograms[I], DeviceI);
      assert(pocl_exists(ProgramBcPath));
      if (InputPrograms[I]->llvm_irs[DeviceI] == nullptr) {
        pocl_llvm_read_program_llvm_irs(InputPrograms[I], DeviceI, nullptr);
      }
    }

    if (InputPrograms[I]->llvm_irs[DeviceI] != nullptr) {
      ++ProgsHaveProgramLLVMIR;
      LLVMIRBinaries.push_back(InputPrograms[I]->llvm_irs[DeviceI]);
    }

    POCL_UNLOCK_OBJ(InputPrograms[I]);
  }

  if (ProgsHaveProgramLLVMIR != NumInputPrograms) {
    POCL_MSG_ERR("LevelZero: not all programs have program.bc\n");
    return CL_LINK_PROGRAM_FAILURE;
  }

  pocl_cache_create_program_cachedir(Program, DeviceI, SpvConcatBinary.data(),
                                     SpvConcatBinary.size(), ProgramBcPath);
  convertProgramBcPathToSpv(ProgramBcPath, ProgramSpvPath);

  if (pocl_exists(ProgramSpvPath) && pocl_exists(ProgramBcPath)) {
    POCL_MSG_PRINT_LEVEL0("Found linked SPIR-V in cache\n");
  } else {

    char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH];
    pocl_cache_tempname(ProgramSpvPathTemp, ".spv", NULL);
    char ProgramBcPathTemp[POCL_MAX_PATHNAME_LENGTH];
    pocl_cache_tempname(ProgramBcPathTemp, ".bc", NULL);

#if 0
    // this can be re-enabled when we get rid of dependency on LLVM IR
    // in the compilation chain
    if (linkWithSpirvLink(Program, DeviceI, ProgramSpvPathTemp, SpvBinaryPaths,
                          CreateLibrary) != CL_SUCCESS) {
      POCL_MSG_WARN("LevelZero : failed to link using spirv-link,"
                    "retrying with llvm-link\n");
    }
#endif
    if (linkWithLLVMLink(Program, DeviceI, ProgramBcPathTemp,
                         ProgramSpvPathTemp, LLVMIRBinaries, 0) != CL_SUCCESS) {
      POCL_MSG_ERR("LevelZero: failed to link "
                   "with both spirv-link and llvm-link\n");
      return CL_LINK_PROGRAM_FAILURE;
    }
    pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
    pocl_rename(ProgramBcPathTemp, ProgramBcPath);
  }

  readProgramSpv(Program, DeviceI, ProgramSpvPath);
  assert(Program->program_il != nullptr);
  assert(Program->program_il_size > 0);
  if (Program->binary_sizes[DeviceI] == 0) {
    char *OutputBinary = nullptr;
    uint64_t OutputBinarySize = 0;
    int Err = pocl_read_file(ProgramBcPath, &OutputBinary, &OutputBinarySize);
    POCL_RETURN_ERROR_ON((Err != 0), CL_LINK_PROGRAM_FAILURE,
                         "failed to read BC file from cache\n");
    Program->binaries[DeviceI] = (unsigned char *)OutputBinary;
    Program->binary_sizes[DeviceI] = OutputBinarySize;
  }
  assert(Program->binaries[DeviceI] != nullptr);
  assert(Program->binary_sizes[DeviceI] > 0);

  if (CreateLibrary == 0) {
    // for Metadata, read the Bitcode into LLVM::Module
    pocl_llvm_read_program_llvm_irs(Program, DeviceI, ProgramBcPath);
    pocl_llvm_recalculate_gvar_sizes(Program, DeviceI);
    return Device->createSpirvProgram(Program, DeviceI);
  } else {
    // only final (linked) programs have  ZE module
    assert(Program->data[DeviceI] == nullptr);
    return CL_SUCCESS;
  }
}

int pocl_level0_free_program(cl_device_id ClDevice, cl_program Program,
                             unsigned ProgramDeviceI) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
#ifdef ENABLE_LLVM
  pocl_llvm_free_llvm_irs(Program, ProgramDeviceI);
#endif
  Device->freeProgram(Program, ProgramDeviceI);
  return CL_SUCCESS;
}

static int pocl_level0_setup_spirv_metadata(cl_device_id Device,
                                            cl_program Program,
                                            unsigned ProgramDeviceI) {
  assert(Program->data[ProgramDeviceI] != nullptr);

  int32_t *Stream = (int32_t *)Program->program_il;
  size_t StreamSize = Program->program_il_size / 4;
  OpenCLFunctionInfoMap KernelInfoMap;
  if (!parseSPIRV(Stream, StreamSize, KernelInfoMap)) {
    POCL_MSG_ERR("Unable to parse SPIR-V module of the program\n");
    return 0;
  }

  Program->num_kernels = KernelInfoMap.size();
  if (Program->num_kernels == 0) {
    POCL_MSG_WARN("No kernels found in program.\n");
    return 1;
  }

  Program->kernel_meta = (pocl_kernel_metadata_t *)calloc(
      Program->num_kernels, sizeof(pocl_kernel_metadata_t));

  uint32_t Idx = 0;
  for (auto &I : KernelInfoMap) {
    mapToPoCLMetadata(I, Program->num_devices,
                      &Program->kernel_meta[Idx]);
    ++Idx;
  }

  return 1;
}

static int pocl_level0_setup_lz_metadata(cl_device_id Device,
                                         cl_program Program,
                                         unsigned ProgramDeviceI) {
  assert(Program->data[ProgramDeviceI] != nullptr);
  Level0NativeProgram *L0Prog = (Level0NativeProgram *)Program->data[ProgramDeviceI];

  POCL_MSG_ERR("pocl_level0_setup_lz_metadata : START\n");

  const std::vector<std::string> &KernelNames = L0Prog->getKernelNames();
  Program->num_kernels = KernelNames.size();

  if (Program->num_kernels == 0) {
    POCL_MSG_WARN("No kernels found in program.\n");
    return 1;
  }

  Program->kernel_meta = (pocl_kernel_metadata_t *)calloc(
    Program->num_kernels, sizeof(pocl_kernel_metadata_t));

  auto releaseK = [=](Level0NativeKernel *K) {
    L0Prog->releaseKernel(K);
  };

  for (uint32_t Idx = 0; Idx < Program->num_kernels; ++Idx) {

    std::unique_ptr<Level0NativeKernel, decltype(releaseK)>
          K{L0Prog->createKernel(KernelNames[Idx]), releaseK};
    if (!K)
      return 0;
    pocl_kernel_metadata_t *Meta = &Program->kernel_meta[Idx];

    ze_kernel_properties_t KernelProps;
    ze_kernel_preferred_group_size_properties_t PrefGroupSize;
    std::string KernelAttrs;
    if (!K->getProperties(KernelProps, PrefGroupSize, KernelAttrs))
      return 0;

    Meta->name = strdup(KernelNames[Idx].c_str());
    Meta->num_args = KernelProps.numKernelArgs;
    //Meta->num_locals = 0;
    //Meta->local_sizes = nullptr;
    if (!KernelAttrs.empty())
        Meta->attributes = strdup(KernelAttrs.c_str());
    Meta->has_arg_metadata = POCL_HAS_KERNEL_ARG_TYPE_NAME |
                             POCL_HAS_KERNEL_ARG_NAME |
                             POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER |
                             POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER;
    Meta->reqd_wg_size[0] = KernelProps.requiredGroupSizeX;
    Meta->reqd_wg_size[1] = KernelProps.requiredGroupSizeY;
    Meta->reqd_wg_size[2] = KernelProps.requiredGroupSizeZ;
    // TODO: setup of these attributes is missing
    // meta->vec_type_hint
    // meta->wg_size_hint


    if (0) {
    Meta->max_subgroups[ProgramDeviceI] = KernelProps.maxSubgroupSize;
    Meta->compile_subgroups[ProgramDeviceI] =
        KernelProps.requiredSubgroupSize;
    Meta->max_workgroup_size[ProgramDeviceI] = 0; // TODO
    Meta->preferred_wg_multiple[ProgramDeviceI] =
          PrefGroupSize.preferredMultiple;
    Meta->local_mem_size[ProgramDeviceI] = KernelProps.localMemSize;
    Meta->private_mem_size[ProgramDeviceI] = KernelProps.privateMemSize;
    Meta->spill_mem_size[ProgramDeviceI] = KernelProps.spillMemSize;
    }

    if (Meta->num_args == 0)
      continue;
    Meta->arg_info = (pocl_argument_info *)calloc(Meta->num_args,
                                           sizeof(pocl_argument_info));

    for (unsigned ArgI = 0; ArgI < KernelProps.numKernelArgs; ++ArgI) {
      pocl_argument_info *Arg = &Meta->arg_info[ArgI];
      uint32_t ArgSize = 0;
      std::string ArgType;
      if (!K->getKernelArgProperties(ArgI, ArgSize, ArgType))
        return 0;
      assert(!ArgType.empty());
      if (ArgType.back() == 0)
        ArgType.pop_back();
      while (isspace(ArgType.back()))
        ArgType.pop_back();

      std::string Name = "Arg" + std::to_string(ArgI);
      Arg->name = strdup(Name.c_str());
      Arg->type_name = strdup(ArgType.c_str());
      Arg->type = POCL_ARG_TYPE_NONE;

      // determine type
      Arg->type_size = ArgSize;
      if (ArgType.back() == '*') {
        Arg->type = POCL_ARG_TYPE_POINTER;
        assert(ArgSize == 0 || ArgSize == sizeof(cl_mem));
      } else {
        if (ArgType == "sampler_t")
          Arg->type = POCL_ARG_TYPE_SAMPLER;
        if (ArgType == "pipe_t")
          Arg->type = POCL_ARG_TYPE_PIPE;
        if (ArgType.size() > 7 &&
            ArgType.compare(0,5,"image",5) == 0 &&
            ArgType.compare(ArgType.size()-2,2,"_t",2) == 0)
          Arg->type = POCL_ARG_TYPE_IMAGE;
      }

      // const, volatile etc
      Arg->type_qualifier = CL_KERNEL_ARG_TYPE_NONE;

      // read-only, read-write etc, images only
      if (Arg->type == POCL_ARG_TYPE_IMAGE)
        Arg->access_qualifier = CL_KERNEL_ARG_ACCESS_READ_WRITE;
      else
        Arg->access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;

      // default private
      Arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
      if (Arg->type == POCL_ARG_TYPE_IMAGE)
        Arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
      if (Arg->type == POCL_ARG_TYPE_POINTER) {
        if (Arg->type_size == 0)
          Arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
        else
          Arg->address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
      }

      POCL_MSG_WARN("Arg %u: NAME %s TYPE_NAME %s TYPE %u "
                    "TYPE_SIZE %u ADDR_Q: %x\n", ArgI,
                    Arg->name, Arg->type_name,
                    (unsigned)Arg->type, Arg->type_size,
                    (unsigned)Arg->address_qualifier);
    }
#if 0 \
    /// TODO: \
    /// required number of subgroups per thread group, \
    /// or zero if there is no required number of subgroups
      uint32_t requiredNumSubGroups;

      /// [out] required subgroup size,
      /// or zero if there is no required subgroup size
      uint32_t requiredSubgroupSize;
#endif
  }

  POCL_MSG_ERR("pocl_level0_setup_lz_metadata : SUCCESS\n");
  return 1;
}


#ifdef ENABLE_NPU

static bool pocl_npu_is_layout_gemm(cl_uint Rank, const void *Layout) {
  const cl_tensor_layout_ml_exp *Ptr = (cl_tensor_layout_ml_exp *)Layout;

  // supported layouts from openvino compiler plugin /
  // "rankToLegacyLayoutString": C, NC, CHW, NCHW, NCDHW
  if (Ptr->ml_type == CL_TENSOR_LAYOUT_ML_NC_EXP && Rank == 2)
    return true;
  if (Ptr->ml_type == CL_TENSOR_LAYOUT_ML_CHW_EXP && Rank == 3)
    return true;
  return false;
}

int pocl_npu_validate_khr_gemm(cl_bool TransA, cl_bool TransB,
                               const cl_tensor_desc_exp *TenA,
                               const cl_tensor_desc_exp *TenB,
                               const cl_tensor_desc_exp *TenCIOpt,
                               const cl_tensor_desc_exp *TenCOut,
                               const cl_tensor_datatype_value_exp *Alpha,
                               const cl_tensor_datatype_value_exp *Beta) {

  // datatype match between A&B and CIopt&COut already checked in initial
  // validation (pocl_validate_khr_gemm)

  // currently FP 16-64 and INT 8-64 are supported.
  POCL_RETURN_ERROR_ON((TenA->dtype == CL_TENSOR_DTYPE_FP8E4M3_EXP ||
                        TenA->dtype == CL_TENSOR_DTYPE_FP8E5M2_EXP ||
                        TenA->dtype == CL_TENSOR_DTYPE_INT4_EXP ||
                        TenCOut->dtype == CL_TENSOR_DTYPE_FP8E4M3_EXP ||
                        TenCOut->dtype == CL_TENSOR_DTYPE_FP8E5M2_EXP ||
                        TenCOut->dtype == CL_TENSOR_DTYPE_INT4_EXP),
                       CL_INVALID_TENSOR_DATATYPE_EXP,
                       "Datatype support not yet implemented. NPU supports "
                       "only FP16/32/64 and INT8/16/32/64 currently\n");

  // type mixing check.
  POCL_RETURN_ERROR_ON((pocl_tensor_type_is_int(TenA->dtype) !=
                        pocl_tensor_type_is_int(TenCOut->dtype)),
                       CL_INVALID_TENSOR_DATATYPE_EXP,
                       "Datatype mixing (INT & FP) not supported\n");

  POCL_RETURN_ERROR_ON((pocl_tensor_type_size(TenA->dtype) >
                        pocl_tensor_type_size(TenCOut->dtype)),
                       CL_INVALID_TENSOR_DATATYPE_EXP,
                       "Datatype of C is smaller than A\n");

  // check validity of data layouts of the tensors.
  POCL_RETURN_ERROR_ON(
      (TenA->layout_type != CL_TENSOR_LAYOUT_ML_EXP ||
       TenB->layout_type != CL_TENSOR_LAYOUT_ML_EXP ||
       TenCOut->layout_type != CL_TENSOR_LAYOUT_ML_EXP ||
       (TenCIOpt && TenCIOpt->layout_type != CL_TENSOR_LAYOUT_ML_EXP)),
      CL_INVALID_TENSOR_LAYOUT_EXP,
      "GEMM on NPU device only supports ML layouts\n");

  POCL_RETURN_ERROR_ON(
      (!pocl_npu_is_layout_gemm(TenA->rank, TenA->layout) ||
       !pocl_npu_is_layout_gemm(TenB->rank, TenB->layout) ||
       !pocl_npu_is_layout_gemm(TenCOut->rank, TenCOut->layout) ||
       (TenCIOpt &&
        !pocl_npu_is_layout_gemm(TenCIOpt->rank, TenCIOpt->layout))),
      CL_INVALID_TENSOR_LAYOUT_EXP,
      "GEMM on NPU device only supports C, NC, CHW, NCHW, NCDHW layouts\n");

  return CL_SUCCESS;
}
#endif

int pocl_level0_supports_dbk(cl_device_id device, cl_dbk_id_exp kernel_id,
                             const void *kernel_attributes) {
#ifdef ENABLE_NPU
  // check for NPU specific requirements on Tensors.
  return pocl_validate_dbk_attributes(kernel_id, kernel_attributes,
                                      pocl_npu_validate_khr_gemm);

#else
  POCL_RETURN_ERROR_ON(1, CL_DBK_UNSUPPORTED_EXP,
                       "The LevelZero driver must be compiled with enabled "
                       "NPU to support tensor DBKs\n");
#endif
}

int pocl_level0_setup_metadata(cl_device_id Dev, cl_program Program,
                               unsigned ProgramDeviceI) {
  if (Program->num_builtin_kernels) {
    return pocl_driver_setup_metadata(Dev, Program, ProgramDeviceI);
  }
  // using the LLVM::Module as source for metadata gets more reliable info
  // than SPIR-V parsing. TODO make the SPIR-V parsing work, so we don't have
  // to use LLVM::Module.
  if (Program->llvm_irs[ProgramDeviceI] != nullptr) {
    return pocl_driver_setup_metadata(Dev, Program, ProgramDeviceI);
  }
  if (Program->program_il && Program->program_il_size) {
    return pocl_level0_setup_spirv_metadata(Dev, Program, ProgramDeviceI);
  }
  if (Program->binaries[ProgramDeviceI] && Program->binary_sizes[ProgramDeviceI]) {
    return pocl_level0_setup_lz_metadata(Dev, Program, ProgramDeviceI);
  }
  POCL_MSG_ERR("LevelZero: Don't know how to setup metadata\n");
  return 0;
}

int pocl_level0_create_kernel(cl_device_id Dev, cl_program Program,
                              cl_kernel Kernel, unsigned ProgramDeviceI) {
  assert(Program->data[ProgramDeviceI] != nullptr);
  Level0Device *Device = (Level0Device *)Dev->data;
  return Device->createKernel(Program, Kernel, ProgramDeviceI);
}

int pocl_level0_free_kernel(cl_device_id Dev, cl_program Program,
                            cl_kernel Kernel, unsigned ProgramDeviceI) {
  assert(Program->data[ProgramDeviceI] != nullptr);
  assert(Kernel->data[ProgramDeviceI] != nullptr);
  Level0Device *Device = (Level0Device *)Dev->data;
  return Device->freeKernel(Program, Kernel, ProgramDeviceI);
}

int pocl_level0_build_poclbinary(cl_program Program, cl_uint DeviceI) {

  // TODO deal with GPU binary somehow
  assert(Program->build_status == CL_BUILD_SUCCESS);

  if (Program->num_kernels == 0) {
    return CL_SUCCESS;
  }

  /* For binaries of other than Executable type (libraries, compiled but
   * not linked programs, etc), do not attempt to compile the kernels. */
  if (Program->binary_type != CL_PROGRAM_BINARY_TYPE_EXECUTABLE) {
    return CL_SUCCESS;
  }

  return (Program->binaries[DeviceI] ? CL_SUCCESS : CL_BUILD_PROGRAM_FAILURE);
}

int pocl_level0_build_builtin(cl_program Program, cl_uint DeviceI) {
  cl_device_id Dev = Program->devices[DeviceI];
  Level0Device *Device = (Level0Device *)Dev->data;

  return Device->createBuiltinProgram(Program, DeviceI);
}

int pocl_level0_init_queue(cl_device_id Dev, cl_command_queue Queue) {
  Level0Device *Device = (Level0Device *)Dev->data;
  assert(Device);
  bool HiddenDefaultQ = (Queue->properties & CL_QUEUE_HIDDEN);
  if (HiddenDefaultQ)
    POCL_MSG_WARN("Creating hidden default queue: %p  |  %zu\n", Queue, Queue->id);
  bool Inorder = (Queue->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) == 0;
  Level0CmdList *CList = Device->createCmdList(Queue->priority,
                                               false, // prefer throughput
                                               Inorder);
  POCL_RETURN_ERROR_COND((CList == nullptr), CL_OUT_OF_RESOURCES);
  PoclL0QueueData *QD = new PoclL0QueueData;
  POCL_RETURN_ERROR_COND((QD == nullptr), CL_OUT_OF_HOST_MEMORY);
  Queue->data = QD;
  QD->CmdList = CList;
  POCL_INIT_COND(QD->Cond);
  return CL_SUCCESS;
}

int pocl_level0_free_queue(cl_device_id Dev, cl_command_queue Queue) {
  Level0Device *Device = (Level0Device *)Dev->data;
  assert(Device);
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;
  if (QD == nullptr)
    return CL_SUCCESS;

  if (QD->CmdList)
    Device->destroyCmdList(QD->CmdList);
  POCL_DESTROY_COND(QD->Cond);
  delete QD;
  Queue->data = nullptr;
  return CL_SUCCESS;
}

void pocl_level0_notify_cmdq_finished(cl_command_queue Queue) {
  // must be called with CQ already locked.
  // this must be a broadcast since there could be multiple
  // user threads waiting on the same command queue
  // in pthread_scheduler_wait_cq().
    POCL_MSG_WARN ("@@@ LEVEL0 : NOTIFY CMDQ FINISHED\n");
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;
  POCL_BROADCAST_COND(QD->Cond);
}

void pocl_level0_notify_event_finished(cl_event Event) {
  PoclL0EventData *EvData = (PoclL0EventData *)Event->data;
  POCL_BROADCAST_COND(EvData->Cond);
}

void pocl_level0_free_event_data(cl_event Event) {
  if (Event->data == nullptr) {
    return;
  }
  PoclL0EventData *EvData = (PoclL0EventData *)Event->data;
  POCL_DESTROY_COND(EvData->Cond);
  POCL_MEM_FREE(Event->data);
}

void pocl_level0_join(cl_device_id Device, cl_command_queue Queue) {
  POCL_LOCK_OBJ(Queue);
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;
  assert(QD);

  while (true) {
      if (Queue->command_count == 0) {
          POCL_UNLOCK_OBJ(Queue);
          return;
      } else {
          POCL_WAIT_COND(QD->Cond, Queue->pocl_lock);
      }
  }
  POCL_UNLOCK_OBJ(Queue);
}

void pocl_level0_wait_event(cl_device_id ClDevice, cl_event Event) {
    POCL_MSG_PRINT_LEVEL0("device->wait_event on event %zu\n", Event->id);
    assert(Event->data);
    PoclL0EventData *EvData = (PoclL0EventData *)Event->data;

    POCL_LOCK_OBJ(Event);
    while (Event->status > CL_COMPLETE) {
        POCL_WAIT_COND(EvData->Cond, Event->pocl_lock);
    }
    POCL_UNLOCK_OBJ(Event);
}

void pocl_level0_flush(cl_device_id ClDev, cl_command_queue Queue) {
  POCL_MSG_WARN("Flush called on Queue: %p | ID %zu\n", Queue, Queue->id);
  Level0Device *Device = (Level0Device *)ClDev->data;
  assert(Device);
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;
  assert(QD);
  int R = QD->CmdList->flush();
  assert(R == 0);
}

void pocl_level0_submit(_cl_command_node *Node, cl_command_queue Queue) {
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;
  cl_event Ev = Node->sync.event.event;

  Node->ready = CL_TRUE;
  assert(Ev->data);
  PoclL0EventData *EvData = (PoclL0EventData *)Ev->data;

  std::vector<cl_event> WaitExtEvents;
  std::vector<cl_event> WaitIntEvents;
  event_node *WaitNode = nullptr;
  LL_FOREACH(Ev->wait_list, WaitNode) {
      cl_event WaitEv = WaitNode->event;
      if (WaitEv->queue != Queue)
          WaitExtEvents.push_back(WaitEv);
      else {
          WaitIntEvents.push_back(WaitEv);
      }
  }
  POCL_MSG_WARN ("LZ : submitting command %zu to Queue %p %zu | CmdList %p \n",
                  Ev->id, Queue, Queue->id, QD->CmdList);
  pocl_update_event_submitted(Ev);
  int Res = QD->CmdList->appendEventToList(Ev, WaitIntEvents, WaitExtEvents);
  assert(Res == ZE_RESULT_SUCCESS);
  POCL_UNLOCK_OBJ(Ev);
}

void pocl_level0_notify(cl_device_id ClDev, cl_event Event, cl_event Finished) {
  _cl_command_node *Node = Event->command;
  Level0Device *Device = (Level0Device *)ClDev->data;
  assert(Device);

  // TODO handling this is not really solved..
  assert(Finished->status >= CL_COMPLETE);
/*
  if (Finished->status < CL_COMPLETE) {
    // remove the Event from unsubmitted list
    PoclL0QueueData *QD = (PoclL0QueueData *)Event->queue->data;
    pocl_update_event_failed_locked(Event);
    // TODO
    //Device->failAndFreeLzEvForClEv(Event);
    return;
  }
*/
  // node is ready to execute
  POCL_MSG_PRINT_LEVEL0("notify on event %zu | READY %i\n", Event->id,
                        Node->ready);

  Device->notifyAndFreeLzEvForClEv(Event);
}

void pocl_level0_update_event(cl_device_id ClDevice, cl_event Event) {
  if (Event->data == nullptr) {
    PoclL0EventData *EvData = (PoclL0EventData *)malloc(sizeof(PoclL0EventData));
    assert(EvData);
    POCL_INIT_COND(EvData->Cond);
    Event->data = (void *)EvData;
  }
  if (Event->status == CL_QUEUED) {
    Event->time_queue = pocl_gettimemono_ns();
  }
  if (Event->status == CL_SUBMITTED) {
    Event->time_submit = pocl_gettimemono_ns();
  }
}

/***********************************************************************************************************/
/***********************************************************************************************************/
/***********************************************************************************************************/
/***********************************************************************************************************/

int pocl_level0_alloc_mem_obj(cl_device_id ClDevice, cl_mem Mem, void *HostPtr) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
  pocl_mem_identifier *P = &Mem->device_ptrs[ClDevice->global_mem_id];

  assert(P->mem_ptr == NULL);

  P->extra = 0;
  /* for Images, ze_image_handler_t */
  P->extra_ptr = NULL;

  /* won't preallocate host-visible memory for images,
   * only for buffers, because images are not set up enough at this point. */
  if ((Mem->is_image != 0u) && (Mem->image_channels == 0)) {
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  void *Allocation = nullptr;
  bool IsAllocHostAccessible = false;
  // special handling for clCreateBuffer called on SVM or USM pointer
  if (((Mem->flags & CL_MEM_USE_HOST_PTR) != 0u) &&
      (Mem->mem_host_ptr_is_svm != 0)) {
    P->mem_ptr = Mem->mem_host_ptr;
    P->version = Mem->mem_host_ptr_version;
    IsAllocHostAccessible = true;
  } else {
    // handle all other cases here.
    // CL_MEM_USE_HOST_PTR without SVM
    //   handled by normal memory + memcpy at sync points
    // CL_MEM_DEVICE_{PRIVATE,SHARED}_ADDRESS_EXT:
    //   Treat cl_ext_buffer_device_address identically as USM Device.
    //   If we passed an SVM/USM address, we can use it directly in the
    //   previous branch. That should be at least a USM Device allocation.
    bool Compress = false;
    if (pocl_get_bool_option("POCL_LEVEL0_COMPRESS", 0)) {
      Compress = (Mem->flags & CL_MEM_READ_ONLY) > 0;
    }

    ze_device_mem_alloc_flags_t DevFlags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;

    ze_host_mem_alloc_flags_t HostFlags =
        ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED |
        ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
    if (Mem->flags & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR |
                      CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL))
      HostFlags |= ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;
    else
      DevFlags |= ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;

    Allocation = Device->allocBuffer((uintptr_t)Mem, DevFlags, HostFlags,
                                     Mem->size, IsAllocHostAccessible);
    if (Allocation == nullptr) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }
    P->mem_ptr = Allocation;
    P->version = 0;
  }

  if (Mem->is_image != 0u) {
    // image attributes must be already set up
    assert(Mem->image_channel_data_type != 0);
    assert(Mem->image_channel_order != 0);
    ze_image_handle_t Image = Device->allocImage(
        Mem->image_channel_data_type, Mem->image_channel_order, Mem->type,
        Mem->flags, Mem->image_width, Mem->image_height, Mem->image_depth,
        Mem->image_array_size);
    if (Image == nullptr) {
      if (Allocation != nullptr) {
        Device->freeBuffer((uintptr_t)Mem, Allocation);
      }
      P->mem_ptr = nullptr;
      P->version = 0;
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    } else {
      P->extra_ptr = (void *)Image;
    }
  }

  if (Mem->mem_host_ptr == nullptr) {
    if (IsAllocHostAccessible) {
      // since we allocate shared memory, use it for mem_host_ptr
      assert((Mem->flags & CL_MEM_USE_HOST_PTR) == 0);
      Mem->mem_host_ptr = Allocation;
      Mem->mem_host_ptr_version = 0;
      ++Mem->mem_host_ptr_refcount;
    } else {
      pocl_alloc_or_retain_mem_host_ptr(Mem);
    }
  }

  POCL_MSG_PRINT_MEMORY("level0 ALLOCATED | MEM_HOST_PTR %p SIZE %zu | "
                        "level0 DEV BUF %p | STA BUF %p | EXTRA_PTR %p \n",
                        Mem->mem_host_ptr, Mem->size, P->mem_ptr,
                        (void *)P->extra, P->extra_ptr);

  return CL_SUCCESS;
}

void pocl_level0_free(cl_device_id ClDevice, cl_mem Mem) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
  pocl_mem_identifier *P = &Mem->device_ptrs[ClDevice->global_mem_id];

  POCL_MSG_PRINT_MEMORY("level0 DEVICE FREE | PTR %p SIZE %zu \n", P->mem_ptr,
                        Mem->size);

  if (Mem->is_image != 0u) {
    assert(P->extra_ptr != nullptr);
    ze_image_handle_t Image = (ze_image_handle_t)P->extra_ptr;
    pocl::Level0Device::freeImage(Image);
  }

  // special handling for clCreateBuffer called on SVM pointer
  if (((Mem->flags & CL_MEM_USE_HOST_PTR) != 0u) &&
      (Mem->mem_host_ptr_is_svm != 0)) {
    P->mem_ptr = nullptr;
    P->version = 0;
  } else {
    Device->freeBuffer((uintptr_t)Mem, P->mem_ptr);
    assert(Mem->mem_host_ptr != nullptr);
  }

  if (Mem->mem_host_ptr != nullptr && Mem->mem_host_ptr == P->mem_ptr) {
    assert((Mem->flags & CL_MEM_USE_HOST_PTR) == 0);
    Mem->mem_host_ptr = nullptr;
    Mem->mem_host_ptr_version = 0;
    --Mem->mem_host_ptr_refcount;
  }

  P->mem_ptr = nullptr;
  P->version = 0;
  P->extra_ptr = nullptr;
  P->extra = 0;
}

cl_int pocl_level0_get_mapping_ptr(void *Data, pocl_mem_identifier *MemId,
                                   cl_mem Mem, mem_mapping_t *Map) {
  /* assume buffer is allocated */
  assert(MemId->mem_ptr);
  assert(Mem->mem_host_ptr);

  assert(Mem->mem_host_ptr);
  Map->host_ptr = (char *)Mem->mem_host_ptr + Map->offset;

  /* POCL_MSG_ERR ("map HOST_PTR: %p | SIZE %zu | OFFS %zu | DEV PTR: %p \n",
                  map->host_ptr, map->size, map->offset, mem_id->mem_ptr); */
  assert(Map->host_ptr);
  return CL_SUCCESS;
}

cl_int pocl_level0_free_mapping_ptr(void *Data, pocl_mem_identifier *MemId,
                                    cl_mem Mem, mem_mapping_t *Map) {
  Map->host_ptr = NULL;
  return CL_SUCCESS;
}

int pocl_level0_create_sampler(cl_device_id ClDevice, cl_sampler Samp,
                               unsigned ContextDeviceI) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
  ze_sampler_handle_t HSampler = Device->allocSampler(
      Samp->addressing_mode, Samp->filter_mode, Samp->normalized_coords);
  if (HSampler == nullptr) {
    POCL_MSG_ERR("Failed to create sampler\n");
    return CL_FAILED;
  }
  Samp->device_data[ClDevice->dev_id] = HSampler;
  return CL_SUCCESS;
}

int pocl_level0_free_sampler(cl_device_id ClDevice, cl_sampler Samp,
                             unsigned ContextDeviceI) {

  ze_sampler_handle_t HSampler =
      (ze_sampler_handle_t)Samp->device_data[ClDevice->dev_id];
  if (HSampler != nullptr) {
    pocl::Level0Device::freeSampler(HSampler);
  }
  return CL_SUCCESS;
}

void *pocl_level0_svm_alloc(cl_device_id Dev, cl_svm_mem_flags Flags,
                            size_t Size) {
  Level0Device *Device = (Level0Device *)Dev->data;
  bool Compress = false;
  if (pocl_get_bool_option("POCL_LEVEL0_COMPRESS", 0)) {
    Compress = (Flags & CL_MEM_READ_ONLY) > 0;
  }
  return Device->allocUSMSharedMem(Size, Compress);
}

void pocl_level0_svm_free(cl_device_id Dev, void *SvmPtr) {
  Level0Device *Device = (Level0Device *)Dev->data;
  Device->freeUSMMem(SvmPtr);
}

void *pocl_level0_usm_alloc(cl_device_id Dev, unsigned AllocType,
                            cl_mem_alloc_flags_intel Flags, size_t Size,
                            cl_int *ErrCode) {
  Level0Device *Device = (Level0Device *)Dev->data;
  int errcode = CL_SUCCESS;
  void *Ptr = nullptr;
  ze_host_mem_alloc_flags_t HostZeFlags = 0;
  ze_device_mem_alloc_flags_t DevZeFlags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
  if (Flags & CL_MEM_ALLOC_WRITE_COMBINED_INTEL)
    HostZeFlags |= ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
  if (Flags & CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL)
    DevZeFlags |= ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;
  if (Flags & CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL)
    HostZeFlags |= ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;

  switch (AllocType) {
  case CL_MEM_TYPE_HOST_INTEL:
    POCL_GOTO_ERROR_ON(!Device->supportsHostUSM(), CL_INVALID_OPERATION,
                       "Device does not support Host USM allocations\n");
    Ptr = Device->allocUSMHostMem(Size, HostZeFlags);
    break;
  case CL_MEM_TYPE_DEVICE_INTEL:
    POCL_GOTO_ERROR_ON(!Device->supportsDeviceUSM(), CL_INVALID_OPERATION,
                       "Device does not support Device USM allocations\n");
    Ptr = Device->allocUSMDeviceMem(Size, DevZeFlags);
    break;
  case CL_MEM_TYPE_SHARED_INTEL:
    POCL_GOTO_ERROR_ON(!Device->supportsSingleSharedUSM(), CL_INVALID_OPERATION,
                       "Device does not support Shared USM allocations\n");
    Ptr = Device->allocUSMSharedMem(Size, false, DevZeFlags, HostZeFlags);
    break;
  default:
    POCL_MSG_ERR("Unknown USM AllocType requested\n");
    errcode = CL_INVALID_PROPERTY;
  }
ERROR:
  if (ErrCode)
    *ErrCode = errcode;
  return Ptr;
}

void pocl_level0_usm_free(cl_device_id Dev, void *SvmPtr) {
  Level0Device *Device = (Level0Device *)Dev->data;
  Device->freeUSMMem(SvmPtr);
}

void pocl_level0_usm_free_blocking(cl_device_id Dev, void *SvmPtr) {
  Level0Device *Device = (Level0Device *)Dev->data;
  Device->freeUSMMemBlocking(SvmPtr);
}

cl_int pocl_level0_get_device_info_ext(cl_device_id Dev,
                                       cl_device_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  Level0Device *Device = (Level0Device *)Dev->data;

  switch (param_name) {

  case CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL:
  case CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL:
  case CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
  case CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
  case CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL: {
    cl_bitfield Caps = Device->getMemCaps(param_name);
    POCL_RETURN_GETINFO(cl_bitfield, Caps);
  }

  case CL_DEVICE_SUB_GROUP_SIZES_INTEL: {
    const std::vector<size_t> &SupportedSGSizes =
        Device->getSupportedSubgroupSizes();
    if (!SupportedSGSizes.empty()) {
      POCL_RETURN_GETINFO_ARRAY(size_t, SupportedSGSizes.size(),
                                SupportedSGSizes.data());
    } else {
      POCL_RETURN_GETINFO(size_t, 0);
    }
  }

  case CL_DEVICE_IP_VERSION_INTEL: {
    uint32_t Res = Device->getIPVersion();
    POCL_RETURN_GETINFO(uint32_t, Res);
  }
  case CL_DEVICE_ID_INTEL: {
    uint32_t Res = Device->getProperties().deviceId;
    POCL_RETURN_GETINFO(uint32_t, Res);
  }
  case CL_DEVICE_NUM_SLICES_INTEL: {
    uint32_t Res = Device->getProperties().numSlices;
    POCL_RETURN_GETINFO(uint32_t, Res);
  }
  case CL_DEVICE_NUM_SUB_SLICES_PER_SLICE_INTEL: {
    uint32_t Res = Device->getProperties().numSubslicesPerSlice;
    POCL_RETURN_GETINFO(uint32_t, Res);
  }
  case CL_DEVICE_NUM_EUS_PER_SUB_SLICE_INTEL: {
    uint32_t Res = Device->getProperties().numEUsPerSubslice;
    POCL_RETURN_GETINFO(uint32_t, Res);
  }
  case CL_DEVICE_NUM_THREADS_PER_EU_INTEL: {
    uint32_t Res = Device->getProperties().numThreadsPerEU;
    POCL_RETURN_GETINFO(uint32_t, Res);
  }
  case CL_DEVICE_FEATURE_CAPABILITIES_INTEL: {
    cl_device_feature_capabilities_intel Res = Device->getFeatureCaps();
    POCL_RETURN_GETINFO(cl_device_feature_capabilities_intel, Res);
  }

  default:
    return CL_INVALID_VALUE;
  }
}

cl_int pocl_level0_get_subgroup_info_ext(
    cl_device_id Dev, cl_kernel Kernel, unsigned ProgramDeviceI,
    cl_kernel_sub_group_info param_name, size_t input_value_size,
    const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {

  Level0Device *Device = (Level0Device *)Dev->data;
  Level0SpecKernel *L0Kernel = (Level0SpecKernel *)Kernel->data[ProgramDeviceI];
  size_t SgSize = Device->getMaxSGSizeForKernel(L0Kernel);

  switch (param_name) {
  case CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE: {
    POCL_RETURN_GETINFO(size_t, SgSize);
  }
  case CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE: {
    POCL_RETURN_GETINFO(
        size_t, std::min((size_t)Dev->max_num_sub_groups,
                         (size_t)((input_value_size > sizeof(size_t)
                                       ? ((size_t *)input_value)[1] / SgSize
                                       : 1) *
                                  (input_value_size > sizeof(size_t) * 2
                                       ? ((size_t *)input_value)[2] / SgSize
                                       : 1))));
  }
  case CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT: {
    POCL_RETURN_ERROR_ON((input_value == NULL), CL_INVALID_VALUE,
                         "SG size wish not given.");
    size_t n_wish = *(size_t *)input_value;

    size_t nd[3];
    if (n_wish * SgSize > Dev->max_work_group_size) {
      nd[0] = nd[1] = nd[2] = 0;
      POCL_RETURN_GETINFO_ARRAY(size_t, param_value_size / sizeof(size_t), nd);
    } else {
      nd[0] = n_wish * SgSize;
      nd[1] = 1;
      nd[2] = 1;
      POCL_RETURN_GETINFO_ARRAY(size_t, param_value_size / sizeof(size_t), nd);
    }
  }
  default:
    POCL_RETURN_ERROR_ON(1, CL_INVALID_VALUE, "Unknown param_name: %u\n",
                         param_name);
  }
}

/*

Enumeration type and values for the param_name parameter to
clGetMemAllocInfoINTEL to query information about a Unified Shared Memory
allocation. Optional allocation properties may also be queried using
clGetMemAllocInfoINTEL:

typedef cl_uint cl_mem_info_intel;

#define CL_MEM_ALLOC_TYPE_INTEL         0x419A
#define CL_MEM_ALLOC_BASE_PTR_INTEL     0x419B
#define CL_MEM_ALLOC_SIZE_INTEL         0x419C
#define CL_MEM_ALLOC_DEVICE_INTEL       0x419D
// CL_MEM_ALLOC_FLAGS_INTEL - defined above

Enumeration type and values describing the type of Unified Shared Memory
allocation. Returned by clGetMemAllocInfoINTEL when param_name is
CL_MEM_ALLOC_TYPE_INTEL:

typedef cl_uint cl_unified_shared_memory_type_intel;

#define CL_MEM_TYPE_UNKNOWN_INTEL       0x4196
#define CL_MEM_TYPE_HOST_INTEL          0x4197
#define CL_MEM_TYPE_DEVICE_INTEL        0x4198
#define CL_MEM_TYPE_SHARED_INTEL        0x4199

*/

/*

Accepted value for the param_name parameter to clSetKernelExecInfo to specify
that the kernel may indirectly access Unified Shared Memory allocations of the
specified type:

#define CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL      0x4200
#define CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL    0x4201
#define CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL    0x4202

Accepted value for the param_name parameter to clSetKernelExecInfo to specify a
set of Unified Shared Memory allocations that the kernel may indirectly access:

#define CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL                  0x4203

*/

cl_int pocl_level0_set_kernel_exec_info_ext(
    cl_device_id Dev, unsigned ProgramDeviceI, cl_kernel Kernel,
    cl_uint param_name, size_t param_value_size, const void *param_value) {

  assert(Kernel->data[ProgramDeviceI] != nullptr);
  Level0SpecKernel *L0Kernel = (Level0SpecKernel *)Kernel->data[ProgramDeviceI];
  assert(L0Kernel != nullptr);
  switch (param_name) {
  case CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM: {
    if (Dev->svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
      return CL_SUCCESS;
    else {
      POCL_RETURN_ERROR_ON(
          1, CL_INVALID_OPERATION,
          "This device doesn't support fine-grain system allocations\n");
    }
  }
  case CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL:
  case CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL:
  case CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL: {
    cl_bool value;
    assert(param_value_size == sizeof(cl_bool));
    memcpy(&value, param_value, sizeof(cl_bool));
    ze_kernel_indirect_access_flag_t Flag;
    switch (param_name) {
    case CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL:
      Flag = ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST;
      break;
    case CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL:
      Flag = ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
      break;
    case CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL:
      Flag = ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE;
      break;
    }

    L0Kernel->setIndirectAccess(Flag, (value != CL_FALSE));
    return CL_SUCCESS;
  }

  case CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT:
  case CL_KERNEL_EXEC_INFO_SVM_PTRS:
  case CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL: {
    std::map<void *, size_t> UsedPtrs;
    cl_uint NumElem = param_value_size / sizeof(void *);
    if (NumElem == 0)
      return CL_INVALID_ARG_VALUE;
    void **Elems = (void **)param_value;
    size_t AllocationSize;
    void *RawPtr;
    // find the allocation sizes for the pointers. Needed for L0 API
    for (cl_uint i = 0; i < NumElem; ++i) {
      AllocationSize = 0;
      RawPtr = nullptr;
      // TODO: DEVICE ptrs do not have vm_ptr set, the check will fail.

      if (param_name == CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT) {
        pocl_raw_ptr *DevPtr =
            pocl_find_raw_ptr_with_dev_ptr(Kernel->context, Elems[i]);
        POCL_RETURN_ERROR_ON((DevPtr == nullptr), CL_INVALID_VALUE,
                             "Invalid pointer given to the call\n");
        AllocationSize = DevPtr->size;
        RawPtr = DevPtr->dev_ptr;
      } else {
        int err = pocl_svm_check_get_pointer(Kernel->context, Elems[i], 1,
                                             &AllocationSize, &RawPtr);
        POCL_RETURN_ERROR_ON((err != CL_SUCCESS), CL_INVALID_VALUE,
                             "Invalid pointer given to the call\n");
      }
      assert(AllocationSize > 0);
      UsedPtrs[RawPtr] = AllocationSize;
    }
    L0Kernel->setAccessedPointers(UsedPtrs);
    return CL_SUCCESS;
  }

  default:
    return CL_INVALID_VALUE;
  }
}

/*
cl_int
pocl_level0_create_finalized_command_buffer(cl_device_id Dev,
                                            cl_command_buffer_khr CmdBuf) {
  Level0Device *Device = (Level0Device *)Dev->data;
  CmdBuf->data[Dev->dev_id] = nullptr;
  void *CmdBufData = Device->createCmdBuf(CmdBuf);
  POCL_RETURN_ERROR_ON(
      (CmdBufData == nullptr), CL_OUT_OF_RESOURCES,
      "Failed to create LevelZero CmdList for command buffer\n");
  CmdBuf->data[Dev->dev_id] = CmdBufData;
  return CL_SUCCESS;
}

cl_int pocl_level0_free_command_buffer(cl_device_id Dev,
                                       cl_command_buffer_khr CmdBuf) {

  Level0Device *Device = (Level0Device *)Dev->data;
  void *CmdBufData = CmdBuf->data[Dev->dev_id];
  if (CmdBufData == nullptr)
    return CL_SUCCESS;
  Device->freeCmdBuf(CmdBufData);
  CmdBuf->data[Dev->dev_id] = nullptr;
  return CL_SUCCESS;
}
*/
