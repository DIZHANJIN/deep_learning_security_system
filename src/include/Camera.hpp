#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstddef>
#include <cstdint>




#define CAMERA_WIDTH 1920
#define CAMERA_HEIGHT 1080
#define FRAME_FMT V4L2_PIX_FMT_NV12





class Camera {
public:

    struct Frame {
        int      index = -1;      // vb2 buffer index
        int      dmabuf_fd = -1;  // exported dmabuf fd (plane0)
        void*    ptr = nullptr;   // mmap pointer (plane0)
        size_t   length = 0;      // mmap length
        uint32_t bytesused = 0;   // actual payload
        int      width = 0;
        int      height = 0;
        int      stride = 0;      // bytesperline
        uint32_t pixfmt = 0;      // e.g., V4L2_PIX_FMT_NV12
    };


    Camera();
    ~Camera();

    bool start();
    void stop();

    bool get_frame(void*& ptr, size_t& len, int& index);
    void release_frame(int index);

	// New API: returns dmabuf fd + metadata (suitable for "Camera -> DMABUF -> RKNN")
    // timeout_ms < 0: wait forever
    bool get_frame_dma(Frame& out, int timeout_ms = -1);

    // Optional: directly query exported dmabuf fd by buffer index
    // Returns -1 if index is invalid or export not available.
    int  dmabuf_fd(int index) const;


    // Experiment 1: Camera dmabuf -> RGA copy -> dump NV12 file (verification only).
    // This helps you verify "driver <-> driver" buffer sharing via dma-buf.
    // Returns false if RGA/dma-heap is unavailable or any step fails.
    bool exp1_rga_copy_dump_nv12(const char* out_file,
                                const char* dma_heap_path = "/dev/dma_heap/system",
                                int timeout_ms = 1000);

private:
    void capture_loop();

    struct Buffer {
        void*  start = nullptr;
        size_t length = 0;
        int    dmabuf_fd = -1; // exported via VIDIOC_EXPBUF (plane0)
    };

	  int      width_  = CAMERA_WIDTH;
    int      height_ = CAMERA_HEIGHT;
    int      stride_ = 0;
    uint32_t pixfmt_ = 0;

    int fd_;
	// Per-buffer "last dequeued" bytesused (plane0)
    std::vector<uint32_t> bytesused_;
	
    std::vector<Buffer> buffers_;

    std::thread capture_thread_;
    std::atomic<bool> is_running_;

    std::queue<int> frame_queue_;
    std::mutex frame_mutex_;
    std::condition_variable frame_cond_;
};

