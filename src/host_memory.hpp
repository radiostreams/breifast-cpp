#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <mutex>
#include <new>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace breifast
{
/// Host blocks reused by exact size, so that one call's planes take no new allocation when the previous call had the
/// same shapes. Blocks come from `allocate_block`, the heap by default; the GPU module replaces it with page-locked
/// memory, which the GPU copies from at full speed. Blocks are never freed, only reused: the pool keeps the most one
/// call had alive at once.
class HostMemoryPool
{
public:
  /// Returns a block of at least `bytes` bytes, or nullptr on failure.
  using AllocateBlock = void *(*)(std::size_t bytes);

  static HostMemoryPool &instance()
  {
    static HostMemoryPool pool;
    return pool;
  }

  /// The default AllocateBlock: 64-byte aligned heap memory.
  static void *heap_block(const std::size_t bytes)
  {
    return ::operator new(bytes, std::align_val_t{64}, std::nothrow);
  }

  /// Blocks handed out before the call keep their origin: blocks are never freed, so mixing them is harmless.
  void set_allocate_block(AllocateBlock allocate_block)
  {
    const std::lock_guard lock{mutex_};
    allocate_block_ = allocate_block;
  }

  void *take(std::size_t bytes)
  {
    AllocateBlock allocate_block = nullptr;
    {
      const std::lock_guard lock{mutex_};
      if (auto &blocks = free_blocks_[bytes]; !blocks.empty())
      {
        void *const block = blocks.back();
        blocks.pop_back();
        return block;
      }
      allocate_block = allocate_block_;
    }
    void *const block = allocate_block(std::max<std::size_t>(bytes, 1)); // outside the lock: pinning is slow
    if (block == nullptr)
    {
      throw std::bad_alloc{};
    }
    return block;
  }

  void give_back(void *const block, const std::size_t bytes)
  {
    const std::lock_guard lock{mutex_};
    free_blocks_[bytes].push_back(block);
  }

private:
  std::mutex mutex_;
  AllocateBlock allocate_block_ = &heap_block;
  std::map<std::size_t, std::vector<void *>> free_blocks_;
};

/// Standard allocator over HostMemoryPool.
template <typename T> struct HostAllocator
{
  using value_type = T;

  HostAllocator() = default;
  template <typename U> HostAllocator(const HostAllocator<U> &) noexcept
  {
  }

  T *allocate(const std::size_t n)
  {
    return static_cast<T *>(HostMemoryPool::instance().take(n * sizeof(T)));
  }
  void deallocate(T *const pointer, const std::size_t n)
  {
    HostMemoryPool::instance().give_back(pointer, n * sizeof(T));
  }

  friend bool operator==(const HostAllocator &, const HostAllocator &) noexcept
  {
    return true;
  }
};

/// A plane cube in pooled host memory: page-locked once the GPU module has set the pool up.
using HostCube = xt::xtensor_container<xt::uvector<float, HostAllocator<float>>, 3, xt::layout_type::row_major>;
} // namespace breifast
