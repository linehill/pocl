; This is an "illegal" C function name on purpose. It's a magic
; handle based on which we know it's the special SG barrier function.
declare void @pocl.subgroup_barrier() convergent

define void @_Z21_cl_sub_group_barrierj(i32 %flags) convergent {
entry:
  call void @pocl.subgroup_barrier()
  ret void
}
