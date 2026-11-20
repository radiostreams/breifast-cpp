#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include <oneapi/tbb/parallel_for.h>

namespace breifast
{
/// Host copies collected while the GIL is held and run in parallel once it is released: the source arrays belong to
/// Python, so the caller keeps them alive until run() returns. Touches no Python state.
class HostCopies
{
public:
  void add(const void *source, void *destination, std::size_t bytes)
  {
    jobs_.push_back({static_cast<const std::byte *>(source), static_cast<std::byte *>(destination), bytes});
  }

  /// Copies everything, split into chunks of `chunk_bytes` spread over the TBB threads.
  void run(std::size_t chunk_bytes = std::size_t{4} << 20) const
  {
    struct Chunk
    {
      const std::byte *source;
      std::byte *destination;
      std::size_t bytes;
    };
    std::vector<Chunk> chunks;
    for (const Job &job : jobs_)
    {
      for (std::size_t offset = 0; offset < job.bytes; offset += chunk_bytes)
      {
        chunks.push_back({job.source + offset, job.destination + offset, std::min(chunk_bytes, job.bytes - offset)});
      }
    }
    tbb::parallel_for(std::size_t{0}, chunks.size(), [&](const std::size_t i)
                      { std::memcpy(chunks[i].destination, chunks[i].source, chunks[i].bytes); });
  }

private:
  struct Job
  {
    const std::byte *source;
    std::byte *destination;
    std::size_t bytes;
  };
  std::vector<Job> jobs_;
};
} // namespace breifast
