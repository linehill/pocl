// TODO copyright

#ifndef POCL_RAW_PTR_SET_H
#define POCL_RAW_PTR_SET_H

#include "pocl_raw_ptr.h"

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct pocl_raw_ptr_set pocl_raw_ptr_set;

  POCL_EXPORT
  pocl_raw_ptr_set *pocl_raw_ptr_set_create (void);

  POCL_EXPORT
  void pocl_raw_ptr_set_destroy (pocl_raw_ptr_set *set);

  /** Insert new pocl_raw_ptr object to the set
   *
   * Returns non-zero value is the insertion took place. The set takes
   * ownership of the inserted object. */
  POCL_EXPORT
  int pocl_raw_ptr_set_insert (pocl_raw_ptr_set *set, pocl_raw_ptr *ptr);

  /** Return pocl_raw_ptr object corresponding to the 'vm_ptr'
   *
   * Returns NULL if object was not found.
   */
  POCL_EXPORT
  pocl_raw_ptr *pocl_raw_ptr_set_lookup_with_vm_ptr (pocl_raw_ptr_set *set,
                                                     const void *vm_ptr);

  /* TODO: desc */
  POCL_EXPORT
  pocl_raw_ptr *pocl_raw_ptr_set_lookup_with_dev_ptr (pocl_raw_ptr_set *set,
                                                      const void *dev_ptr);

  /* Return the head of the set iterable with utlist's DL_FOREACH macro.
   */
  POCL_EXPORT
  pocl_raw_ptr *pocl_raw_ptr_set_begin (pocl_raw_ptr_set *set);

  POCL_EXPORT void pocl_raw_ptr_set_remove(pocl_raw_ptr_set *Set,
                                           pocl_raw_ptr *Ptr);

  POCL_EXPORT
  void pocl_raw_ptr_set_erase (pocl_raw_ptr_set *set, pocl_raw_ptr *ptr);

  /**/
  POCL_EXPORT
  void pocl_raw_ptr_set_erase_all (pocl_raw_ptr_set *set);

  /**/
  POCL_EXPORT
  void pocl_raw_ptr_set_erase_all_by_shadow_mem (pocl_raw_ptr_set *Set,
                                                 cl_mem shadow_cl_mem);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POCL_RAW_PTR_SET_H */
