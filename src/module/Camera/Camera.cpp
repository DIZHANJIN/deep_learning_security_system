#include "Camera.hpp"

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>



#define BUFFER_COUNT 4

static int xioctl(int fd, int request, void* arg)
{
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

Camera::Camera(const char* dev):  is_running_(false)
{
    fd_ = open(dev, O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        perror("open video device");
        throw std::runtime_error("open failed");
    }

    v4l2_format fmt{};
    //memset(&fmt, 0, sizeof(struct v4l2_format));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = CAMERA_WIDTH;
    fmt.fmt.pix_mp.height = CAMERA_HEIGHT;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 1;   // 先按 1 处理

    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        throw std::runtime_error("set fmt failed");
    }


	// Cache negotiated metadata (driver may adjust)
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


        
        buffers_[i].length = buf.m.planes.length;


        buffers_[i].start = mmap(
            nullptr,
            buffers_[i].length,        // 使用 plane 的长度
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            fd_,
            buf.m.planes[0].m.mem_offset   // 使用 plane 的偏移量
        );

        if (buffers_[i].start == MAP_FAILED) {
            perror("mmap");
            throw std::runtime_error("mmap failed");
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
            perr or("VIDIOC_QBUF(init)");
            throw std::runtime_error("QBUF init failed");
        }


        xioctl(fd_, VIDIOC_QBUF, &buf);
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
		b.dmabuf_fd = -1;
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
    out.n    = pixfmt_;
    return true;
}

int Camera::dmabuf_fd(int index) const
{
    if (index < 0 || static_cast<size_t>(index) >= buffers_.size()) return -1;
    return buffers_[index].dmabuf_fd;
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
