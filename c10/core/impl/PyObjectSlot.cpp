#include <c10/core/impl/PyObjectSlot.h>

namespace c10::impl {

PyObjectSlot::PyObjectSlot() : pyobj_interpreter_(nullptr), pyobj_(0) {}

PyObjectSlot::~PyObjectSlot() {
  maybe_destroy_pyobj();
}

void PyObjectSlot::maybe_destroy_pyobj() {
  if (owns_pyobj()) {
    TORCH_INTERNAL_ASSERT(pyobj_interpreter_ != nullptr);
    TORCH_INTERNAL_ASSERT(pyobj_ != 0);
    (*pyobj_interpreter_.load(std::memory_order_acquire))
        ->decref(_unchecked_untagged_pyobj(), /*has_pyobj_slot*/ true);
    // NB: this destructor can only be entered when there are no
    // references to this C++ object (obviously), NOR any references
    // to the PyObject (if there are references to the PyObject,
    // then the PyObject holds an owning reference to the tensor).
    // So it is OK to clear pyobj_ here as it is impossible for it to
    // be used again (modulo weak reference races)
    pyobj_.store(0, std::memory_order_relaxed);  // for safety
  }
}

PyInterpreter* PyObjectSlot::pyobj_interpreter() {
  return pyobj_interpreter_.load(std::memory_order_acquire);
}

PyObject* PyObjectSlot::_unchecked_untagged_pyobj() const {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<PyObject*>(pyobj_.load(std::memory_order_acquire) & ~0x1ULL);
}

PyInterpreter& PyObjectSlot::load_pyobj_interpreter() const {
  auto interpreter = pyobj_interpreter_.load(std::memory_order_acquire);
  if (interpreter) {
    return *interpreter;
  }
  TORCH_CHECK(false, "cannot access PyObject for Tensor - no interpreter set");
}

bool PyObjectSlot::owns_pyobj() {
  return (pyobj_.load(std::memory_order_acquire) & 1) != 0;
}

void PyObjectSlot::set_owns_pyobj(bool b) {
  uintptr_t expected = pyobj_.load(std::memory_order_relaxed);
  uintptr_t value;
  do {
    value = (expected & ~0x1ULL) | (b ? 1 : 0);
  } while (!pyobj_.compare_exchange_weak(
      expected, value, std::memory_order_release, std::memory_order_relaxed));
}

} // namespace c10::impl
