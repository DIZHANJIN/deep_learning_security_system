

#include "Common.hpp"
#include "Model.hpp"
#include "Sensor.hpp"
#include "RknnPool.hpp"
#include "Camera.hpp" 

#include <future>
#include <rga/im2d.h>
#include <rga/rga.h>

#include <iostream>



// 获取欧几里得距离
static float get_duclidean_distance(float * output1, float * output2)
{
    float sum = 0;
    for(int i = 0; i < 128; i++) {
        sum += (output1[i] - output2[i]) * (output1[i] - output2[i]);
    }
    float norm = std::sqrt(sum);
    return norm;
}



// 使用 RGA 把 NV12 转为 BGR888（原始分辨率）
// nv12_ptr 来自 Camera 的缓冲区（mmap 地址）
// out_bgr 必须已经分配好，size = width * height * 3
static bool rga_nv12_to_bgr(void* nv12_ptr,
                            int width,
                            int height,
                            int stride,
                            uint8_t* out_bgr)
{
    // 输入：NV12
    rga_buffer_t src = wrapbuffer_virtualaddr(
        nv12_ptr,
        width,
        height,
        stride,
        RK_FORMAT_YCbCr_420_SP   // NV12
    );

    // 输出：BGR888
    rga_buffer_t dst = wrapbuffer_virtualaddr(
        out_bgr,
        width,
        height,
        width * 3,
        RK_FORMAT_BGR_888
    );

    IM_STATUS st = imcvtcolor(
                              src,
                              dst,
                              RK_FORMAT_YCbCr_420_SP,   // 源格式（NV12）
                              RK_FORMAT_RGB_888,        // 目标格式
                              0,                        // mode，一般填 0
                              0,                        // reserved
                              nullptr                   // 不需要 status 数组
                          );
    
    if (st != IM_STATUS_SUCCESS) {
        std::cerr << "RGA imcvtcolor NV12->BGR failed: " << imStrError(st) << std::endl;
        return false;
    }
    return true;
}


// // ============================ FaceRknnPool ============================

// // 构造函数，初始化线程池和模型
FaceRknnPool::FaceRknnPool()
{
    try {
        // 配置线程池，使用指定数量的线程
        thread_pool_ = std::make_unique<ThreadPool>(thread_num_);

        // 每个线程加载一个模型
        for(int i = 0; i < this->thread_num_; ++i) {
            retinaface_models_.push_back(std::make_shared<Retinaface>()); // 使用模型路径初始化模型

            facenet_models_.push_back(std::make_shared<Facenet>()); // 使用模型路径初始化模型
        }

    } catch(const std::bad_alloc & e) {
        std::cout << "Out of memory: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    // 初始化每个模型
    for(int i = 0; i < this->thread_num_; ++i) {

        auto ret = retinaface_models_[i]->init(retinaface_models_[0]->get_rknn_context(), i != 0);
        if(ret != 0) {
            std::cout << "Init rknn model failed!" << std::endl;
            exit(EXIT_FAILURE);
        }

        ret = facenet_models_[i]->init(facenet_models_[0]->get_rknn_context(), i != 0);
        if(ret != 0) {
            std::cout << "Init rknn model failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    this->retinaface_model_size_ = this->retinaface_models_[0]->get_model_width();
    this->facenet_model_size_    = this->facenet_models_[0]->get_model_width();
}

FaceRknnPool::~FaceRknnPool()
{}

// 向线程池添加推理任务
void FaceRknnPool::add_inference_task(std::shared_ptr<cv::Mat> src, ImageProcess & retinaface_image_process,
                                      bool is_generate_face_feature)
{

    // 将任务添加到线程池
    thread_pool_->enqueue(
        [&](std::shared_ptr<cv::Mat> original_img, bool is_generate_face_feature) { // 线程池执行的任务
            try {
                // 处理输入图像
                auto convert_img = retinaface_image_process.convert(*original_img);

                // 获取模型ID
                auto mode_id = get_model_id();

                // 创建一个新的图像来适配模型输入大小
                cv::Mat rgb_img =
                    cv::Mat::zeros(this->retinaface_models_[mode_id]->get_model_width(),
                                   this->retinaface_models_[mode_id]->get_model_height(), convert_img->type());

                // 将图像从BGR转换为RGB
                cv::cvtColor(*convert_img, rgb_img, cv::COLOR_BGR2RGB);

                retinaface_result results; // 存放推理结果
                // 使用模型进行推理
                this->retinaface_models_[mode_id]->inference(rgb_img.ptr(), &results,
                                                             retinaface_image_process.get_letter_box());

                // 是否是同一人脸
                bool is_check = false;

                if(results.count > 0 && is_face_recognition_) {
                    is_check = this->face_recognition(mode_id, *original_img, results, is_generate_face_feature);

                    uint64_t current_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                     std::chrono::system_clock::now().time_since_epoch())
                                                     .count();

                    if(this->pre_show_oled_timestamp_ == 0 ||
                       current_timestamp - this->pre_show_oled_timestamp_ > 500) {
                        this->pre_show_oled_timestamp_ = current_timestamp;

                        [[maybe_unused]] auto future = std::async(std::launch::async, [&]() { OLED::show(is_check); });
                    }
                } else if(results.count > 0 && is_generate_face_feature) {
                    is_check = this->face_recognition(mode_id, *original_img, results, is_generate_face_feature);
                    std::cout << "success record" << std::endl;
                    return;
                }

                cv::Scalar recognition_color{0, 255, 0};
                cv::Scalar color{255, 255, 255};

                // 进行图像后处理
                retinaface_image_process.image_post_process(*original_img, results,
                                                            is_check ? recognition_color : color);

                // 锁住结果队列，将推理结果加入队列
                std::lock_guard<std::mutex> lock_guard(this->image_results_mutex_);
                this->image_results_.push(std::move(original_img));
                this->image_results_cv_.notify_one();
            } catch(std::exception & e) {
                std::cout << "FaceRknnPool---add_inference_task: " << e.what() << std::endl;
            }

        },
        std::move(src), is_generate_face_feature); // 向线程池添加任务
}

// 获取当前应该使用的模型ID
int FaceRknnPool::get_model_id()
{
    std::lock_guard<std::mutex> lock(id_mutex_); // 锁住ID生成过程
    int mode_id = id_;                           // 获取当前模型ID
    id_++;                                       // 自增ID
    if(id_ == thread_num_) {                     // 如果ID达到线程数限制，重置为0
        id_ = 0;
    }
    return mode_id; // 返回模型ID
}

// 从结果队列中获取图像结果
std::shared_ptr<cv::Mat> FaceRknnPool::get_image_result_from_queue()
{
    std::unique_lock<std::mutex> lock(this->image_results_mutex_);
    this->image_results_cv_.wait(lock, [this]() { return !this->image_results_.empty(); });

    // 否则，获取队列中的第一个图像并移除它
    auto res = this->image_results_.front();
    this->image_results_.pop();
    return std::move(res); // 返回结果图像
}

int FaceRknnPool::get_retinaface_model_size()
{
    return this->retinaface_model_size_;
}

int FaceRknnPool::get_facenet_model_size()
{
    return this->facenet_model_size_;
}

int FaceRknnPool::get_facenet_feature_vector_size()
{
    return this->facenet_feature_vector_.size();
}

void FaceRknnPool::clean_image_results()
{
    std::lock_guard<std::mutex> lock(this->image_results_mutex_);
    while(!this->image_results_.empty()) {
        this->image_results_.pop();
    }
}

bool FaceRknnPool::face_recognition(int mode_id, cv::Mat & image, retinaface_result & results,
                                    bool is_generate_face_feature)
{

    retinaface_object * detect_result = &(results.object[0]);
    //边界检查
    int left   = detect_result->box.left;
    int top    = detect_result->box.top;
    int right  = detect_result->box.right;
    int bottom = detect_result->box.bottom;
    
    // 1. 确保左上角不小于 0
    left = std::max(0, left);
    top  = std::max(0, top);

    // 2. 确保右下角不超过图像边界
    right = std::min(image.cols, right);
    bottom = std::min(image.rows, bottom);

    // 3. 确保宽度和高度有效（防止裁剪后 width/height <= 0）
    int width  = right - left;
    int height = bottom - top;

    if (width <= 0 || height <= 0) {
        // 如果裁剪后的尺寸无效，直接返回（跳过本次特征提取）
        return false; 
    }
    cv::Mat crop_img = image(cv::Rect(left, top, width, height));
    
    /***
    cv::Mat crop_img                  = image(cv::Rect(detect_result->box.left, detect_result->box.top,
                                                       detect_result->box.right - detect_result->box.left,
                                                       detect_result->box.bottom - detect_result->box.top));
    ***/

    auto facenet_image_process =
        std::make_unique<ImageProcess>(crop_img.cols, crop_img.rows, this->get_facenet_model_size());

    auto convert_img = facenet_image_process->convert(crop_img);

    cv::Mat rgb_img = cv::Mat::zeros(this->facenet_models_[mode_id]->get_model_width(),
                                     this->facenet_models_[mode_id]->get_model_height(), convert_img->type());

    // 将图像从BGR转换为RGB
    cv::cvtColor(*convert_img, rgb_img, cv::COLOR_BGR2RGB);

    std::vector<float> out_fp32(128);

    this->facenet_models_[mode_id]->inference(rgb_img.ptr(), out_fp32, facenet_image_process->get_letter_box());

    if(is_generate_face_feature) {
        this->facenet_feature_vector_.push_back(std::move(out_fp32));
        return false;
    } else {
        if(this->is_face_recognition_) {
            for(auto out : facenet_feature_vector_) {
                float distance = get_duclidean_distance(out.data(), out_fp32.data());
                if(distance < 0.6) {
                    return true;
                }
            }
        }
    }
    return false;
}

void FaceRknnPool::change_face_recognition_status(bool status)
{
    this->is_face_recognition_ = status;
}

// ============================ SecurityRknnPool ============================

SecurityRknnPool::SecurityRknnPool()
{
    this->thread_num_ = RKNN_POOL_SIZE;

    init_yolo_post_process(YOLO11_LABEL_PATH);

    try {
        this->thread_pool_ = std::make_unique<ThreadPool>(this->thread_num_);

        for(int i = 0; i < this->thread_num_; ++i) {
            models_.push_back(std::make_shared<Yolo11>());
        }

    } catch(const std::bad_alloc & e) {
        std::cout << "Out of memory: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < this->thread_num_; ++i) {
        auto ret = models_[i]->init(models_[0]->get_rknn_context(), i != 0);
        if(ret != 0) {
            std::cout << "Init rknn model failed!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    this->yolo_model_size_ = this->models_[0]->get_model_width();
}

SecurityRknnPool::~SecurityRknnPool()
{
    deinit_yolo_post_process();
}

void SecurityRknnPool::add_inference_task(std::shared_ptr<cv::Mat> src, ImageProcess & image_process)
{
    thread_pool_->enqueue(
        [&](std::shared_ptr<cv::Mat> original_img) {

            // 显示当前年月日时分秒
            time_t now = time(nullptr);
            strftime(this->time_str_, sizeof(this->time_str_), "%Y-%m-%d %H:%M:%S", localtime(&now));
            std::string time_str_s(this->time_str_);
            cv::putText(*original_img, time_str_s, cv::Point(original_img->cols - 1200, original_img->rows - 80),
                        cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 255, 255), 5, cv::LINE_8);

            this->is_person = false;

            auto convert_img = image_process.convert(*original_img);

            auto mode_id = get_model_id();

            cv::Mat rgb_img = cv::Mat::zeros(this->models_[mode_id]->get_model_width(),
                                             this->models_[mode_id]->get_model_height(), convert_img->type());
            cv::cvtColor(*convert_img, rgb_img, cv::COLOR_BGR2RGB);

            yolo_result_list results;
            this->models_[mode_id]->inference(rgb_img.ptr(), &results, image_process.get_letter_box());

            if(results.count > 0) {
                for(int i = 0; i < results.count; ++i) {
                    if(results.results[i].cls_id == 0) {
                        this->is_person = true;
                        break;
                    }
                }
            }

            cv::Scalar color{255, 0, 255};
            image_process.image_post_process(*original_img, results, color);

            std::lock_guard<std::mutex> lock_guard(this->image_results_mutex_);

            this->image_results_.push(std::move(original_img));
        },
        std::move(src));
}





void SecurityRknnPool::add_inference_task_from_nv12(void* nv12_ptr,
                                                    int width,
                                                    int height,
                                                    int stride,
                                                    int camera_index,
                                                    Camera& camera,
                                                    ImageProcess& image_process)
{
    // 把 NV12 指针封装进一个 shared_ptr，仅仅是为了和现有线程池调用方式类似；
    // 自定义 deleter 在任务结束时归还这一帧到 camera（QBUF）。
    auto nv12_buf = std::shared_ptr<uint8_t>(
        static_cast<uint8_t*>(nv12_ptr),
        [&camera, camera_index](uint8_t* p) {
            // 不 delete p，因为这块内存是 camera 的 mmap 缓冲区，
            // 只需要把 index 重新 QBUF 回去。
            camera.release_frame(camera_index);
        });

    thread_pool_->enqueue(
        [this, &image_process, width, height, stride](std::shared_ptr<uint8_t> nv12_data) {

            // 1. 先做 NV12 → BGR（原始分辨率）的硬件转换
            cv::Mat original_bgr(height, width, CV_8UC3);
            if (!rga_nv12_to_bgr(nv12_data.get(), width, height, stride,
                                 original_bgr.data)) {
                return; // 失败就丢掉这帧
            }

            auto original_img = std::make_shared<cv::Mat>(std::move(original_bgr));

            // 2. 以下逻辑尽量复用你原来 add_inference_task 里的代码：
            //    - 画时间戳
            //    - image_process.convert 做 letterbox + resize
            //    - cvtColor BGR->RGB
            //    - yolo 推理
            //    - image_post_process 画框
            //    - image_results_ push(original_img)

            // 显示当前年月日时分秒
            time_t now = time(nullptr);
            strftime(this->time_str_, sizeof(this->time_str_), "%Y-%m-%d %H:%M:%S", localtime(&now));
            std::string time_str_s(this->time_str_);
            cv::putText(*original_img, time_str_s,
                        cv::Point(original_img->cols - 1200, original_img->rows - 80),
                        cv::FONT_HERSHEY_SIMPLEX, 3,
                        cv::Scalar(255, 255, 255), 5, cv::LINE_8);

            this->is_person = false;

            // 这里 convert() 内部就不再需要 NV12->BGR 了，
            // 它直接拿到的是一张 BGR 图。
            auto convert_img = image_process.convert(*original_img);

            auto mode_id = get_model_id();

            cv::Mat rgb_img = cv::Mat::zeros(
                this->models_[mode_id]->get_model_width(),
                this->models_[mode_id]->get_model_height(),
                convert_img->type());

            cv::cvtColor(*convert_img, rgb_img, cv::COLOR_BGR2RGB);

            yolo_result_list results;
            this->models_[mode_id]->inference(rgb_img.ptr(), &results, image_process.get_letter_box());

            if (results.count > 0) {
                for (int i = 0; i < results.count; ++i) {
                    if (results.results[i].cls_id == 0) {
                        this->is_person = true;
                        break;
                    }
                }
            }

            cv::Scalar color{255, 0, 255};
            image_process.image_post_process(*original_img, results, color);

            std::lock_guard<std::mutex> lock_guard(this->image_results_mutex_);
            this->image_results_.push(std::move(original_img));
        },
        std::move(nv12_buf));
}





int SecurityRknnPool::get_model_id()
{
    std::lock_guard<std::mutex> lock(id_mutex_);
    int mode_id = id_;
    id_++;
    if(id_ == thread_num_) {
        id_ = 0;
    }
    return mode_id;
}

std::shared_ptr<cv::Mat> SecurityRknnPool::get_image_result_from_queue(bool is_pop)
{
    std::lock_guard<std::mutex> lock_guard(this->image_results_mutex_);

    if(this->image_results_.empty()) {
        return nullptr;
    } else {
        auto res = std::make_shared<cv::Mat>(*this->image_results_.front());

        if(is_pop) {
            this->image_results_.pop();
        }

        return std::move(res);
    }
}

int SecurityRknnPool::get_yolo_model_size()
{
    return this->yolo_model_size_;
}
