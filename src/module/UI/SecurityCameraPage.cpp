// #include "Font.hpp"
// #include "Lvgl.hpp"
// #include "UI.hpp"
// #include "Util.hpp"
// #include <iostream>
// #include <mutex>
// #include <src/core/lv_obj.h>
// #include <src/layouts/flex/lv_flex.h>
// #include <src/misc/lv_area.h>
// #include <src/misc/lv_color.h>
// #include <src/misc/lv_palette.h>
// #include <thread>

// extern "C" {
// LV_IMAGE_DECLARE(bg);
// }

// static LvObject * main_screen;
// static LvImageDsc * image_dsc_ = new LvImageDsc;
// static LvLabel * alarm_label;
// static LvImage * image_;
// static LvTimer * timer;
// static LvObject * record_icon;
// static LvAnimation * record_icon_anim;

// static void anim_opacity_cb_(void * var, int32_t value)
// {
//     lv_obj_set_style_bg_opa((lv_obj_t *)var, value, 0);
// }


// void SecurityRknnPool::add_inference_task_dmabuf(const Camera::Frame& frame)
// {
//     // 把关键信息复制一份放到任务里（避免直接持有引用）
//     InferenceTask task;
//     task.dmabuf_fd = frame.dmabuf_fd;
//     task.width     = frame.width;
//     task.height    = frame.height;
//     task.stride    = frame.stride;
//     task.pixfmt    = frame.pixfmt;  // NV12

//     // 放到任务队列，唤醒工作线程
//     {
//         std::lock_guard<std::mutex> lk(queue_mutex_);
//         task_queue_.push(std::move(task));
//     }
//     queue_cv_.notify_one();
// }




// SecurityCameraPage::SecurityCameraPage(Camera & camera, ImageProcess & image_process,
//                                        SecurityRknnPool & security_rknn_pool, FFmpeg & ffmpeg)
//     : camera_(camera), image_process_(image_process), security_rknn_pool_(security_rknn_pool), ffmpeg_(ffmpeg)
// {
//     main_screen = new LvObject(nullptr);
//     main_screen->set_style_bg_image_src(&bg, 0).set_style_text_font(Font16::get_font(), 0);

//     image_ = new LvImage(main_screen->raw());
//     image_->align(LV_ALIGN_CENTER, 0, -50);

//     alarm_label = new LvLabel(main_screen->raw(), "human detected", lv_palette_main(LV_PALETTE_RED));
//     alarm_label->align(LV_ALIGN_CENTER, 0, 100).set_style_text_font(Font24::get_font(), 0).add_flag(LV_OBJ_FLAG_HIDDEN);

//     record_icon = new LvObject(image_->raw());

//     record_icon->add_flag(LV_OBJ_FLAG_HIDDEN)
//         .align(LV_ALIGN_TOP_RIGHT, -10, 10)
//         .set_size(100, 40)
//         .set_flex_flow(LV_FLEX_FLOW_ROW)
//         .set_flex_align(LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER)
//         .set_style_border_color(lv_color_hex(0xff0000), 0)
//         .set_style_bg_opa(LV_OPA_30, 0)
//         .remove_flag(LV_OBJ_FLAG_SCROLLABLE);

//     LvObject record_icon_dot(record_icon->raw());

//     record_icon_dot.set_size(20, 20)
//         .set_style_radius(20, 0)
//         .set_style_border_width(0, 0)
//         .set_style_bg_color(lv_color_hex(0xff0000), 0)
//         .remove_flag(LV_OBJ_FLAG_SCROLLABLE);

//     LvLabel record_icon_label(record_icon->raw(), "REC", lv_color_black());

//     record_icon_anim = new LvAnimation();
//     record_icon_anim->set_var(record_icon_dot.raw())
//         .set_exec_cb(anim_opacity_cb_)
//         .set_duration(1000)
//         .set_delay(0)
//         .set_values(0, 255)
//         .set_path_cb(lv_anim_path_ease_in_out)
//         .set_repeat_count(LV_ANIM_REPEAT_INFINITE);

//     LvButton back_button(main_screen->raw(), "back");
//     back_button.set_pos(10, 10).add_event_cb(
//         [&](lv_event_t * e, void * data) { PageManager::getInstance().switchToPage(PageManager::PageType::MAIN_PAGE); },
//         LV_EVENT_CLICKED, nullptr);

//     LvObject record_container(main_screen->raw());
//     record_container.set_size(LV_SIZE_CONTENT, LV_SIZE_CONTENT)
//         .align(LV_ALIGN_BOTTOM_MID, -250, 0)
//         .set_style_bg_opa(LV_OPA_TRANSP, 0)
//         .set_style_border_width(0, 0)
//         .set_flex_flow(LV_FLEX_FLOW_COLUMN)
//         .set_flex_align(LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
//     LvSwitch record_switch(record_container.raw());
//     record_switch.set_size(80, 40).add_event_cb(
//         [&](lv_event_t * e, void * data) {
//             auto target = (lv_obj_t *)lv_event_get_target(e);
//             if(lv_obj_has_state(target, LV_STATE_CHECKED)) {
//                 this->inner_start_record();
//             } else {
//                 this->inner_stop_record();
//             }
//         },
//         LV_EVENT_VALUE_CHANGED, nullptr);
//     LvLabel label(record_container.raw(), "vedio", lv_color_white());

//     LvObject auto_record_container(main_screen->raw());
//     auto_record_container.set_size(LV_SIZE_CONTENT, LV_SIZE_CONTENT)
//         .align(LV_ALIGN_BOTTOM_MID, 0, 0)
//         .set_style_bg_opa(LV_OPA_TRANSP, 0)
//         .set_style_border_width(0, 0)
//         .set_flex_flow(LV_FLEX_FLOW_COLUMN)
//         .set_flex_align(LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
//     LvSwitch auto_record_switch(auto_record_container.raw());
//     auto_record_switch.set_size(80, 40).add_event_cb(
//         [&](lv_event_t * e, void * data) {
//             auto target = (lv_obj_t *)lv_event_get_target(e);
//             if(lv_obj_has_state(target, LV_STATE_CHECKED)) {
//                 is_auto_record_ = true;
//             } else {
//                 is_auto_record_ = false;
//             }
//         },
//         LV_EVENT_VALUE_CHANGED, nullptr);
//     LvLabel auto_record_label(auto_record_container.raw(), "auto record", lv_color_white());

//     LvObject alarm_container(main_screen->raw());
//     alarm_container.set_size(LV_SIZE_CONTENT, LV_SIZE_CONTENT)
//         .align(LV_ALIGN_BOTTOM_MID, 250, 0)
//         .set_style_bg_opa(LV_OPA_TRANSP, 0)
//         .set_style_border_width(0, 0)
//         .set_flex_flow(LV_FLEX_FLOW_COLUMN)
//         .set_flex_align(LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
//     LvSwitch alarm_switch(alarm_container.raw());
//     alarm_switch.set_size(80, 40).add_event_cb(
//         [&](lv_event_t * e, void * data) {
//             auto target = (lv_obj_t *)lv_event_get_target(e);
//             if(lv_obj_has_state(target, LV_STATE_CHECKED)) {
//                 is_alarm_ = true;
//             } else {
//                 is_alarm_ = false;
//             }
//         },
//         LV_EVENT_VALUE_CHANGED, nullptr);
//     LvLabel alarm_label(alarm_container.raw(), "auto call police", lv_color_white());

//     timer = new LvTimer(
//         [&](lv_timer_t * t, void * data) {
//             std::unique_lock<std::mutex> lock(frame_mutex_);
//             frame_cv_.wait(lock, [this]() { return !is_rtsp_turn_ || !is_running_; });

//             if(!is_running_) {
//                 return;
//             }

//             auto image_result = security_rknn_pool_.get_image_result_from_queue();

//             is_rtsp_turn_ = true;
//             frame_cv_.notify_one();

//             if(image_result) {

//                 cv::resize(*image_result, *image_result, cv::Size(800, 450));

//                 memset(image_dsc_, 0, sizeof(LvImageDsc));

//                 image_dsc_->raw()->data      = image_result->data;
//                 image_dsc_->raw()->data_size = image_result->total() * image_result->elemSize();
//                 image_dsc_->raw()->header.w  = image_result->cols;
//                 image_dsc_->raw()->header.h  = image_result->rows;
//                 image_dsc_->raw()->header.cf = LV_COLOR_FORMAT_RGB888;

//                 image_->set_src(image_dsc_->raw());
//             }
//         },
//         10, nullptr);

//     timer->pause();
// }

// void SecurityCameraPage::show()
// {
//     lv_screen_load(main_screen->raw());

//     is_running_ = true;

//     timer->resume();

//     //ffmpeg_.start_process_frame();

//     std::thread([this]() {
//         try {
//             camera_.start();
//             while(is_running_) {
				
// 				void* dma_ptr;
// 				size_t dma_len;
// 				int dma_index;
				
// 				camera_.get_frame(dma_ptr, dma_len, dma_index);

				
// 				cv::Mat nv12(
// 					CAMERA_HEIGHT * 3 / 2,
// 					CAMERA_WIDTH,
// 					CV_8UC1,
// 					dma_ptr
// 				);
				
// 				cv::Mat rgb;
// 				image_process_.nv12_to_rgb(nv12, rgb);
// 				camera_.release_frame(dma_index);

//                 security_rknn_pool_.add_inference_task(std::make_shared<cv::Mat>(rgb), image_process_);
//                 {
//                     std::unique_lock<std::mutex> lock(frame_mutex_);
//                     // 等待直到轮到RTSP线程取帧
//                     frame_cv_.wait(lock, [this]() { return is_rtsp_turn_; });
//                     auto image_result = security_rknn_pool_.get_image_result_from_queue(true);

//                     is_rtsp_turn_ = false;
//                     frame_cv_.notify_one();

//                     if(image_result) {
//                         ffmpeg_.push_frame(std::move(image_result));

//                         // 如果自动录像开启，并且检测到人，并且is_auto_record_start为false，记录is_auto_record_start_time，开始录像
//                         // 如果自动录像开启，并且检测到人，并且is_auto_record_start为true，记录is_auto_record_start_time，继续录像
//                         // 如果自动录像开启，没有检测到人，并且is_auto_record_start为true，且is_auto_record_start_time超过当前时间3秒，则停止录像
//                         if(is_auto_record_) {
//                             if(security_rknn_pool_.is_person) {
//                                 if(!is_auto_record_start_) {
//                                     is_auto_record_start_   = true;
//                                     auto_record_start_time_ = std::time(nullptr);
//                                     this->inner_start_record();
//                                     std::cout << "start vedio" << std::endl;
//                                 } else {
//                                     auto_record_start_time_ = std::time(nullptr);
//                                 }
//                             } else {
//                                 if(is_auto_record_start_ && std::time(nullptr) - auto_record_start_time_ >
//                                                                 SECURITY_CAMERA_PAGE_AUTO_RECORD_DELAY_TIME) {
//                                     is_auto_record_start_ = false;
//                                     this->inner_stop_record();

//                                     std::cout << "stop vedio1" << std::endl;
//                                 }
//                             }
//                         } else {
//                             if(is_auto_record_start_) {
//                                 is_auto_record_start_ = false;
//                                 this->inner_stop_record();

//                                 std::cout << "stop vedio2" << std::endl;
//                             }
//                         }

//                         if(is_alarm_ && security_rknn_pool_.is_person && !is_alarm_start_) {
//                             is_alarm_start_ = true;
//                             std::thread([this]() {
//                                 lv_async_call([](void *) { alarm_label->remove_flag(LV_OBJ_FLAG_HIDDEN); }, nullptr);
//                                 // execute_command("aplay -D bluealsa:DEV=D0:6A:81:16:56:27,PROFILE=a2dp /root/alert.wav");
//                                 std::this_thread::sleep_for(std::chrono::seconds(2));
//                                 is_alarm_start_ = false;
//                                 lv_async_call([](void *) { alarm_label->add_flag(LV_OBJ_FLAG_HIDDEN); }, nullptr);
//                             }).detach();
//                         }
//                     }
//                 }
//             }
//             camera_.stop();
//         } catch(std::exception & e) {
//             std::cerr << "Error: " << e.what() << std::endl;
//         }
//     }).detach();
// }

// void SecurityCameraPage::hide()
// {
//     //ffmpeg_.stop_process_frame();

//     is_running_ = false;

//     timer->pause();
// }

// void SecurityCameraPage::inner_start_record()
// {
//     ffmpeg_.start_record();

//     lv_async_call(
//         [](void *) {
//             record_icon->remove_flag(LV_OBJ_FLAG_HIDDEN);
//             record_icon_anim->start();
//         },
//         nullptr);
// }

// void SecurityCameraPage::inner_stop_record()
// {
//     ffmpeg_.stop_record();

//     lv_async_call(
//         [](void *) {
//             record_icon->add_flag(LV_OBJ_FLAG_HIDDEN);
//             record_icon_anim->stop();
//         },
//         nullptr);
// }


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