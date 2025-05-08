
#include "pocl.h"

#ifndef POCL_RAW_PTR_H
#define POCL_RAW_PTR_H

/**
 * Enumeration for raw buffer/pointer types managed by PoCL.
 */
typedef enum
{
  /* SVM from OpenCL 2.0. */
  POCL_RAW_PTR_SVM = 0,
  /* Intel USM extension. */
  POCL_RAW_PTR_INTEL_USM,
  /* cl_ext_buffer_device_address. */
  POCL_RAW_PTR_DEVICE_BUFFER
} pocl_raw_pointer_kind;

typedef struct _pocl_raw_ptr pocl_raw_ptr;
struct _pocl_raw_ptr
{
  /* The virtual address, if any.  NULL if there's none. */
  void *vm_ptr;
  /* The device address, if known. NULL if not. */
  void *dev_ptr;
  /* The owner device of the allocation, if any. Should be set to non-null for
     USM Device. */
  cl_device_id device;

  size_t size;
  /* A cl_mem for internal bookkeeping and implicit buffer migration. */
  cl_mem shadow_cl_mem;

  /* The raw pointer/buffer API used for the allocation. */
  pocl_raw_pointer_kind kind;

  struct
  {
    cl_mem_alloc_flags_intel flags;
    unsigned alloc_type;
  } usm_properties;

  struct _pocl_raw_ptr *prev, *next;
};

#endif /* POCL_RAW_PTR_H */
