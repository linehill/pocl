// pocl_raw_ptr_set implementation using STL.

#include "pocl_raw_ptr_set.h"

#include "pocl_cl.h"
#include "utlist.h"

#include <map>

using AddrRange = std::pair<const void *, size_t>; // Base address and length

struct AddrRangeLess {
  bool operator()(const AddrRange &Lhs, const AddrRange &Rhs) const {
    const auto *LhsPtr = static_cast<const char *>(Lhs.first);
    const auto *RhsPtr = static_cast<const char *>(Rhs.first);
    return LhsPtr + Lhs.second <= RhsPtr;
  }
};

struct pocl_raw_ptr_set {
  pocl_raw_ptr *Head = nullptr;
  std::map<AddrRange, pocl_raw_ptr *, AddrRangeLess> VmPtrMap;
  std::map<AddrRange, pocl_raw_ptr *, AddrRangeLess> DevPtrMap;
};

void assertValidity(const pocl_raw_ptr *Ptr) {
  assert(Ptr && "Invalid pocl_raw_ptr argument!");
  assert((Ptr->vm_ptr != Ptr->dev_ptr || !Ptr->vm_ptr) &&
         "vm_ptr and dev_ptr can't be both set");
  assert(Ptr->size);
  assert(!Ptr->prev);
  assert(!Ptr->next);
}

extern "C" POCL_EXPORT pocl_raw_ptr_set *pocl_raw_ptr_set_create() {
  // TODO: catch OoM exception.
  return new pocl_raw_ptr_set;
}

extern "C" POCL_EXPORT void pocl_raw_ptr_set_destroy(pocl_raw_ptr_set *Set) {
  pocl_raw_ptr_set_erase_all(Set);
  delete Set;
}

extern "C" POCL_EXPORT int pocl_raw_ptr_set_insert(pocl_raw_ptr_set *Set,
                                                   pocl_raw_ptr *RawPtr) {
  assert(Set && "Invalid pocl_raw_ptr_set argument!");
  assertValidity(RawPtr);

  if (RawPtr->vm_ptr) {
    auto Insertion = Set->VmPtrMap.insert(
        std::make_pair(AddrRange{RawPtr->vm_ptr, RawPtr->size}, RawPtr));
    assert(Insertion.second && "Overlapping VM pointer!");
    (void)Insertion;
  }

  if (RawPtr->dev_ptr) {
    auto Insertion = Set->DevPtrMap.insert(
        std::make_pair(AddrRange{RawPtr->dev_ptr, RawPtr->size}, RawPtr));
    assert(Insertion.second && "Overlapping BDA pointer!");
    (void)Insertion;
  }

  DL_APPEND(Set->Head, RawPtr);

  return 1;
}

extern "C" POCL_EXPORT pocl_raw_ptr *
pocl_raw_ptr_set_lookup_with_vm_ptr(pocl_raw_ptr_set *Set, const void *VmPtr) {
  assert(Set && "Invalid pocl_raw_ptr_set argument!");
  auto Key = AddrRange{VmPtr, 1};
  auto It = Set->VmPtrMap.find(Key);
  return It != Set->VmPtrMap.end() ? It->second : nullptr;
}

extern "C" POCL_EXPORT pocl_raw_ptr *
pocl_raw_ptr_set_lookup_with_dev_ptr(pocl_raw_ptr_set *Set,
                                     const void *DevPtr) {
  assert(Set && "Invalid pocl_raw_ptr_set argument!");
  auto Key = AddrRange{DevPtr, 1};
  auto It = Set->DevPtrMap.find(Key);
  return It != Set->DevPtrMap.end() ? It->second : nullptr;
}

extern "C" POCL_EXPORT pocl_raw_ptr *
pocl_raw_ptr_set_begin(pocl_raw_ptr_set *Set) {
  assert(Set && "Invalid pocl_raw_ptr_set argument!");
  return Set->Head;
}

extern "C" POCL_EXPORT void pocl_raw_ptr_set_remove(pocl_raw_ptr_set *Set,
                                                    pocl_raw_ptr *Ptr) {
  assert(Set && "Invalid pocl_raw_ptr_set argument!");

  if (!Ptr)
    return;

  // TODO: membership check in debug mode.

  if (Ptr->vm_ptr)
    Set->VmPtrMap.erase(AddrRange{Ptr->vm_ptr, 1});
  if (Ptr->dev_ptr)
    Set->DevPtrMap.erase(AddrRange{Ptr->dev_ptr, 1});

  DL_DELETE(Set->Head, Ptr);
}

extern "C" POCL_EXPORT void pocl_raw_ptr_set_erase(pocl_raw_ptr_set *Set,
                                                   pocl_raw_ptr *Ptr) {
  pocl_raw_ptr_set_remove(Set, Ptr);
  free(Ptr);
}

extern "C" POCL_EXPORT void pocl_raw_ptr_set_erase_all(pocl_raw_ptr_set *Set) {
  assert(Set && "Invalid pocl_raw_ptr_set argument!");

  pocl_raw_ptr *Ptr;
  DL_FOREACH (Set->Head, Ptr) {
    free(Ptr);
  }

  Set->Head = nullptr;
  Set->VmPtrMap.clear();
  Set->DevPtrMap.clear();
}

extern "C" POCL_EXPORT void
pocl_raw_ptr_set_erase_all_by_shadow_mem(pocl_raw_ptr_set *Set,
                                         cl_mem ShadowClMem) {
  assert(Set && "Invalid pocl_raw_ptr_set argument!");
  assert(ShadowClMem && "Invalid cl_mem argument!");

  // Linear probe. This can be improved later if needed.
  pocl_raw_ptr *Ptr = nullptr;
  pocl_raw_ptr *Tmp = nullptr;
  DL_FOREACH_SAFE (Set->Head, Ptr, Tmp) {
    if (Ptr->shadow_cl_mem == ShadowClMem)
      pocl_raw_ptr_set_erase(Set, Ptr);
  }
}
