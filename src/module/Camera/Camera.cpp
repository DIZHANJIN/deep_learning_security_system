#include "Camera.hpp"

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <iostream>
#include <chrono>

// Experiment 1 helpers: dma-heap + librga
#include <linux/dma-heap.h>
#include <sys/stat.h>
#include <rga/im2d.h>
#include <rga/rga.h>

#define BUFFER_COUNT 4

static int xioctl(int fd, int request, void* arg)
{
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

static int alloc_dmabuf_from_heap(const char* heap_path, size_t size)
{
    int heap_fd = open(heap_path, O_RDONLY | O_CLOEXEC);
    if (heap_fd < 0) {
        return -1;
    }

    dma_heap_allocation_data data{};
    data.len = size;
    data.fd_flags = O_RDWR | O_CLOEXEC;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data) < 0) {
        close(heap_fd);
        return -1;
    }

    close(heap_fd);
    return data.fd;
}

static bool write_all(const char* path, const void* buf, size_t len)
{
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
    if (fd < 0) {
        return false;
    }
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, p + off, len - off);
        if (n < 0) {
            close(fd);
            return false;
        }
        off += static_cast<size_t>(n);
    }
    close(fd);
    return true;
}

Camera::Camera():  is_running_(false)
{
    const char* dev = "/dev/video11";
    fd_ = open(dev, O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        perror("open video device");
        throw std::runtime_error("open failed");
    }

    v4l2_format fmt{};
    
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = CAMERA_WIDTH;
    fmt.fmt.pix_mp.height = CAMERA_HEIGHT;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.field = V4L2_FIELD_ANY;
    fmt.fmt.pix_mp.num_planes = 1;   // 先按 1 处理

    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        throw std::runtime_error("set fmt failed");
    }


    	//Cache negotiated metadata (driver may adjust)
    	width_	= static_cast<int>(fmt.fmt.pix_mp.width);
    	height_ = static_cast<int>(fmt.fmt.pix_mp.height);
    	pixfmt_ = fmt.fmt.pix_mp.pixelformat;
    	stride_ = static_cast<int>(fmt.fmt.pix_mp.plane_fmt[0].bytesperline);


    v4l2_requestbuffers req{};
    req.count  = BUFFER_COUNT;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        perror("VIDIOC_REQBUFS");
        throw std::runtime_error("reqbuf failed");
    }

    buffers_.resize(req.count);
	  bytesused_.assign(req.count, 0);

    for (size_t i = 0; i < buffers_.size(); i++) {
        v4l2_buffer buf;
		    memset(&buf, 0, sizeof(buf));
        v4l2_plane planes[1];
		    memset(planes, 0, sizeof(planes));

		
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        
        // 【关键2】将 planes 数组挂载到 buf 上
        buf.m.planes = planes;
        buf.length   = 1; // 告诉驱动 planes 数组有多大

        if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) 
        {
        perror("VIDIOC_QUERYBUF");
        throw std::runtime_error("QUERYBUF failed");
        }
        
           std::cout << "buf[" << i << "] length=" << buf.m.planes[0].length
              << " offset=0x" << std::hex << buf.m.planes[0].m.mem_offset
              << std::dec << std::endl;

        // 防御：长度为 0 直接报错，不去 mmap
        if (buf.m.planes[0].length == 0) {
            std::cerr << "ERROR: plane length is 0, mmap would fail\n";
            throw std::runtime_error("invalid plane length");
        }



        
        buffers_[i].length = planes[0].length;


        buffers_[i].start = mmap(
            nullptr,
            planes[0].length,        // 使用 plane 的长度
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd_,
            planes[0].m.mem_offset   // 使用 plane 的偏移量
        );


        if (buffers_[i].start == MAP_FAILED) {
            perror("mmap");
            throw std::runtime_error("mmap " + std::to_string(i) + " failed");
        }


        // Export plane0 as dmabuf fd (zero-copy handle for downstream accelerators)
        // NOTE: This does NOT change the capture memory type (still MMAP).
        // It simply provides an FD alias to the same kernel buffer.
        v4l2_exportbuffer exp{};
        exp.type  = buf.type;
        exp.index = buf.index;
        exp.plane = 0;
        exp.flags = O_CLOEXEC;

		
        if (xioctl(fd_, VIDIOC_EXPBUF, &exp) < 0) {
            // Not all drivers support EXPBUF; keep running without dmabuf.
            // Your RK ISP vb2 generally supports this, but handle gracefully.
            buffers_[i].dmabuf_fd = -1;
            // Optional: perror("VIDIOC_EXPBUF");
        } else {
            buffers_[i].dmabuf_fd = exp.fd;
        }

        
        if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF(init)");
            throw std::runtime_error("QBUF init failed");
        }
    }
}

Camera::~Camera()
{
    stop();
    for (auto& b : buffers_) {
        if (b.start && b.length) {
            munmap(b.start, b.length);
        }
        if (b.dmabuf_fd >= 0) {
            close(b.dmabuf_fd);
            b.dmabuf_fd = -1;
        }
    }
    if (fd_ >= 0) {
        close(fd_);
		    fd_ = -1;
    }
}

bool Camera::start()
{
    if (is_running_) return true;

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        perror("STREAMON");
        return false;
    }

    is_running_ = true;
    capture_thread_ = std::thread(&Camera::capture_loop, this);
    return true;
}

void Camera::stop()
{
    if (!is_running_) return;

    is_running_ = false;
    if (capture_thread_.joinable())
        capture_thread_.join();

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
}

void Camera::capture_loop()
{
    while (is_running_) {
        v4l2_buffer buf{};
        v4l2_plane planes[1]{};

        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.m.planes = planes;
        buf.length   = 1;

        if (xioctl(fd_, VIDIOC_DQBUF, &buf) == 0) {
            {
                std::lock_guard<std::mutex> lk(frame_mutex_);
                frame_queue_.push(buf.index);
            }
            frame_cond_.notify_one();
        } else {
            // 非阻塞模式下 EAGAIN 很常见，避免空转吃满 CPU
            if (errno == EAGAIN) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            perror("VIDIOC_DQBUF");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;

		const int idx = static_cast<int>(buf.index);
				if (idx >= 0 && static_cast<size_t>(idx) < bytesused_.size()) {
					bytesused_[idx] = planes[0].bytesused;
				}
			
				{
					std::lock_guard<std::mutex> lk(frame_mutex_);
					frame_queue_.push(idx);
				}
				frame_cond_.notify_one();

			
        }
    }
}

bool Camera::get_frame(void*& ptr, size_t& len, int& index)
{
    std::unique_lock<std::mutex> lk(frame_mutex_);
    frame_cond_.wait(lk, [&]() { return !frame_queue_.empty(); });

    index = frame_queue_.front();
    frame_queue_.pop();

    ptr = buffers_[index].start;
    len = buffers_[index].length;
    return true;
}

bool Camera::get_frame_dma(Frame& out, int timeout_ms)
{
    std::unique_lock<std::mutex> lk(frame_mutex_);

    if (timeout_ms < 0) {
        frame_cond_.wait(lk, [&]() { return !frame_queue_.empty() || !is_running_; });
    } else {
        frame_cond_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                             [&]() { return !frame_queue_.empty() || !is_running_; });
    }

    if (frame_queue_.empty()) {
        return false;
    }

    const int index = frame_queue_.front();
    frame_queue_.pop();

    out.index     = index;
    out.ptr       = buffers_[index].start;
    out.length    = buffers_[index].length;
    out.dmabuf_fd = buffers_[index].dmabuf_fd;
    out.bytesused = (index >= 0 && static_cast<size_t>(index) < bytesused_.size()) ? bytesused_[index] : 0;
    out.width     = width_;
    out.height    = height_;
    out.stride    = stride_;
    out.pixfmt    = pixfmt_;
    return true;
}

int Camera::dmabuf_fd(int index) const
{
    if (index < 0 || static_cast<size_t>(index) >= buffers_.size()) return -1;
    return buffers_[index].dmabuf_fd;
}



bool Camera::exp1_rga_copy_dump_nv12(const char* out_file, const char* dma_heap_path, int timeout_ms)
{
    // 1) Get one frame (dmabuf fd) from camera
    Frame f{};
    if (!get_frame_dma(f, timeout_ms)) {
        std::cerr << "[exp1] get_frame_dma timeout/fail\n";
        return false;
    }
    if (f.dmabuf_fd < 0) {
        std::cerr << "[exp1] camera dmabuf_fd is invalid (EXPBUF unsupported?)\n";
        release_frame(f.index);
        return false;
    }
    // This experiment assumes NV12; if your camera negotiated a different format,
    // you can still try copy, but the RGA format mapping must be adjusted.
    if (f.pixfmt != V4L2_PIX_FMT_NV12) {
        std::cerr << "[exp1] warning: pixfmt is not NV12 (0x" << std::hex << f.pixfmt << std::dec
                  << "), this experiment expects NV12\n";
    }

    // 2) Allocate an output dmabuf to receive the copy
    // NV12 size is usually stride * height * 3/2
    const size_t out_size = static_cast<size_t>(f.stride) * static_cast<size_t>(f.height) * 3 / 2;
    int out_fd = alloc_dmabuf_from_heap(dma_heap_path, out_size);
    if (out_fd < 0) {
        std::cerr << "[exp1] alloc dmabuf failed, heap=" << dma_heap_path
                  << " (check /dev/dma_heap/*)\n";
        release_frame(f.index);
        return false;
    }

    // 3) RGA copy: input dmabuf -> output dmabuf
    // For NV12, librga commonly uses RK_FORMAT_YCbCr_420_SP
    const int rga_fmt = RK_FORMAT_YCbCr_420_SP;

    rga_buffer_t src = wrapbuffer_fd(f.dmabuf_fd, f.width, f.height, rga_fmt);
    rga_buffer_t dst = wrapbuffer_fd(out_fd,     f.width, f.height, rga_fmt);

    IM_STATUS st = imcopy(src, dst);
    if (st != IM_STATUS_SUCCESS) {
        std::cerr << "[exp1] imcopy failed: " << imStrError(st) << "\n";
        close(out_fd);
        release_frame(f.index);
        return false;
    }

    // 4) mmap output dmabuf ONCE for verification dump (learning aid)
    void* out_ptr = mmap(nullptr, out_size, PROT_READ, MAP_SHARED, out_fd, 0);
    if (out_ptr == MAP_FAILED) {
        std::cerr << "[exp1] mmap output dmabuf failed\n";
        close(out_fd);
        release_frame(f.index);
        return false;
    }

    const bool ok = write_all(out_file, out_ptr, out_size);
    munmap(out_ptr, out_size);
    close(out_fd);

    // 5) Return buffer to camera
    release_frame(f.index);

    if (!ok) {
        std::cerr << "[exp1] write dump file failed: " << out_file << "\n";
        return false;
    }

    std::cerr << "[exp1] dumped NV12 to " << out_file << " (" << out_size << " bytes)\n";
    return true;
}


void Camera::release_frame(int index)
{
    v4l2_buffer buf{};
    v4l2_plane planes[1]{};

    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = index;

    buf.m.planes = planes;
    buf.length   = 1;

    if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        perror("VIDIOC_QBUF(release)");
    }
}
