#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace core {

class HugePageBuffer {
public:
    HugePageBuffer() = default;
    explicit HugePageBuffer(std::size_t bytes, bool prefer_huge_pages = false) {
        allocate(bytes, prefer_huge_pages);
    }

    HugePageBuffer(const HugePageBuffer&) = delete;
    HugePageBuffer& operator=(const HugePageBuffer&) = delete;

    HugePageBuffer(HugePageBuffer&& other) noexcept {
        *this = std::move(other);
    }

    HugePageBuffer& operator=(HugePageBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            mmap_backed_ = other.mmap_backed_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.mmap_backed_ = false;
        }
        return *this;
    }

    ~HugePageBuffer() { release(); }

    void allocate(std::size_t bytes, bool prefer_huge_pages = false) {
        release();
        if (bytes == 0) return;
#if defined(__linux__)
        if (prefer_huge_pages) {
            void* ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (ptr != MAP_FAILED) {
                data_ = static_cast<std::uint8_t*>(ptr);
                size_ = bytes;
                mmap_backed_ = true;
                std::memset(data_, 0, size_);
                return;
            }
        }
#endif
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, bytes) != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }
        data_ = static_cast<std::uint8_t*>(ptr);
        size_ = bytes;
        mmap_backed_ = false;
        std::memset(data_, 0, size_);
    }

    void release() noexcept {
        if (!data_) return;
#if defined(__linux__)
        if (mmap_backed_) {
            munmap(data_, size_);
        } else {
            std::free(data_);
        }
#else
        std::free(data_);
#endif
        data_ = nullptr;
        size_ = 0;
        mmap_backed_ = false;
    }

    [[nodiscard]] std::uint8_t* data() noexcept { return data_; }
    [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool mmap_backed() const noexcept { return mmap_backed_; }

private:
    std::uint8_t* data_ = nullptr;
    std::size_t size_ = 0;
    bool mmap_backed_ = false;
};

} // namespace core
