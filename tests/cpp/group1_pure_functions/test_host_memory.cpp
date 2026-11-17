#include <cstddef>
#include <new>

#include <catch2/catch_test_macros.hpp>

#include "host_memory.hpp"

using breifast::HostCube;
using breifast::HostMemoryPool;

// The pool is process-wide: each case uses sizes of its own, so that blocks left by another case do not interfere.

TEST_CASE("HostMemoryPool hands a given-back block to the next request of the same size", "[host_memory]")
{
  auto &pool = HostMemoryPool::instance();
  void *const block = pool.take(1001);
  pool.give_back(block, 1001);
  CHECK(pool.take(1001) == block);
  void *const other = pool.take(1002);
  CHECK(other != block);
  pool.give_back(block, 1001);
  pool.give_back(other, 1002);
}

TEST_CASE("HostMemoryPool throws bad_alloc when its allocate function fails", "[host_memory]")
{
  auto &pool = HostMemoryPool::instance();
  pool.set_allocate_block([](std::size_t) -> void * { return nullptr; });
  CHECK_THROWS_AS(pool.take(1003), std::bad_alloc);
  pool.set_allocate_block(&HostMemoryPool::heap_block);
  void *const block = pool.take(1003);
  CHECK(block != nullptr);
  pool.give_back(block, 1003);
}

TEST_CASE("A HostCube reuses the memory of a destroyed cube of the same shape", "[host_memory]")
{
  const float *first_data = nullptr;
  {
    HostCube cube = HostCube::from_shape({3, 5, 7});
    cube.fill(1.F);
    first_data = cube.data();
  }
  HostCube cube = HostCube::from_shape({3, 5, 7});
  CHECK(cube.data() == first_data);
}
