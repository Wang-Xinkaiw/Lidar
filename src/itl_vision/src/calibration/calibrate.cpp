#include "itl_vision/calibration/calibrate.h"
#include <string>
#include <sys/stat.h>
#include <iostream>

namespace itl_vision {

CameraCalibrator::CameraCalibrator(const CalibrationConfig& config) 
    : config_(config), world_points_(config.world_points.toVector()) {}

bool CameraCalibrator::initialize(const std::string& camera_params_path) {
    cv::FileStorage fs(camera_params_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open camera parameters file: " << camera_params_path << std::endl;
        return false;
    }
    
    fs["camera_matrix"] >> camera_matrix_;
    fs["dist_coeffs"] >> dist_coeffs_;
    fs.release();
    
    is_initialized_ = true;
    std::cout << "Camera calibrator initialized successfully!" << std::endl;
    return true;
}

void CameraCalibrator::processImage(const cv::Mat& input_image) {
    cv::resize(input_image, current_image_, 
               cv::Size(config_.image.width, config_.image.height));
    
    display_image_ = current_image_.clone();
    
    if (is_calibrating_) {
        drawCalibrationUI();

    if (adjusting_point_) {
        updateRoiDisplay(current_adjust_x_, current_adjust_y_);
    }} 
    else {
        cv::putText(display_image_, "Press Enter to Calibrate", 
                   cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 3, 
                   cv::Scalar(0, 0, 255), 2);
    }
}

void CameraCalibrator::processCompressedImage(const std::vector<unsigned char>& compressed_data) {
    if (!is_initialized_) return;
    
    cv::Mat img = cv::imdecode(compressed_data, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to decode compressed image" << std::endl;
        return;
    }
    
    processImage(img);
}

void CameraCalibrator::solvePnP() {
    if (picked_points_.size() != world_points_.size()) {
        std::cerr << "Point count mismatch: " << picked_points_.size() 
                  << " vs " << world_points_.size() << std::endl;
        return;
    }
    
    int method = cv::SOLVEPNP_EPNP;
    if (config_.pnp_method == "ITERATIVE") {
        method = cv::SOLVEPNP_ITERATIVE;
    }
    
    bool pnp_success = cv::solvePnP(world_points_, picked_points_, 
                               camera_matrix_, dist_coeffs_, 
                               rvec_, tvec_, false, method);
    
    if (pnp_success) {
        std::cout << "Calibration successful!" << std::endl;
        std::cout << "Rotation vector: " << rvec_ << std::endl;
        std::cout << "Translation vector: " << tvec_ << std::endl;
        saveCalibrationResult();
        calibration_complete_ = true;
        should_exit_ = true; // 设置退出标志
    } else {
        std::cerr << "Calibration failed!" << std::endl;
    }
    
    picked_points_.clear();
    is_calibrating_ = false;
    adjusting_point_ = false;
}

void CameraCalibrator::startCalibration() {
    if (!is_calibrating_) {
        is_calibrating_ = true;
        picked_points_.clear();
        calibration_complete_ = false;
        should_exit_ = false;
        std::cout << "Calibration started. Please select " 
                  << world_points_.size() << " points." << std::endl;
    }
}

void CameraCalibrator::cancelCalibration() {
    is_calibrating_ = false;
    adjusting_point_ = false;
    picked_points_.clear();
    std::cout << "Calibration cancelled" << std::endl;
}

void CameraCalibrator::handleMouseEvent(int event, int x, int y, int flags) {
    (void)flags;
    if (!is_calibrating_ || !is_initialized_ || adjusting_point_) return;
    
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            if (is_calibrating_) {
                current_adjust_x_ = x;
                current_adjust_y_ = y;
                adjustPointWithKeyboard(current_adjust_x_, current_adjust_y_);
            }
            break;
            
        case cv::EVENT_MOUSEMOVE:
            if (!adjusting_point_ && x >= 0 && y >= 0 && 
                x < current_image_.cols && y < current_image_.rows) {
                updateRoiDisplay(x, y);
            }
            break;
            
        default:
            break;
    }
}

void CameraCalibrator::handleKeyEvent(int key) {
    switch (key) {
        case 13: // Enter key
            if (!is_calibrating_) {
                startCalibration();
            }
            break;
            
        case 27: // ESC key
            if (is_calibrating_) {
                cancelCalibration();
            }
            break;
            
        default:
            break;
    }
}

void CameraCalibrator::adjustPointWithKeyboard(int& x, int& y) {
    adjusting_point_ = true;
    std::cout << "Adjusting point " << (picked_points_.size() + 1) 
              << ". Use WASD to move, N to confirm." << std::endl;
    
    bool point_confirmed = false;
    
    while (!point_confirmed && adjusting_point_) {
        // 更新ROI显示
        updateRoiDisplay(x, y);
        
        // 在主图像上显示调整状态
        display_image_ = current_image_.clone();
        drawCalibrationUI();
        
        std::string adjust_text = "Adjusting point " + std::to_string(picked_points_.size() + 1);
        cv::putText(display_image_, adjust_text, 
                   cv::Point(50, 250), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                   cv::Scalar(0, 255, 255), 2);
        cv::putText(display_image_, "WASD: Move, N: Confirm", 
                   cv::Point(50, 280), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                   cv::Scalar(0, 255, 255), 2);
        cv::putText(display_image_, "Position: x=" + std::to_string(x) + " y=" + std::to_string(y), 
                   cv::Point(50, 310), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                   cv::Scalar(0, 255, 255), 2);
        
        // 在主图像上标记当前调整点
        cv::circle(display_image_, cv::Point(x, y), 8, cv::Scalar(0, 255, 255), 2);
        cv::circle(display_image_, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), -1);
        
        // 显示图像
        cv::imshow("calibrate", display_image_);
        if (!roi_image_.empty()) {
            cv::imshow("ROI", roi_image_);
        }
        
        int key = cv::waitKey(30); // 等待按键
        
        switch (key) {
            case 'w': case 'W':
                y = std::max(config_.image.roi_zoom_size, y - 1);
                break;
            case 'a': case 'A':
                x = std::max(config_.image.roi_zoom_size, x - 1);
                break;
            case 's': case 'S':
                y = std::min(current_image_.rows - config_.image.roi_zoom_size, y + 1);
                break;
            case 'd': case 'D':
                x = std::min(current_image_.cols - config_.image.roi_zoom_size, x + 1);
                break;
            case 'n': case 'N':
                point_confirmed = true;
                break;
            case 27: // ESC
                adjusting_point_ = false;
                cancelCalibration();
                return;
        }
    }
    
    if (point_confirmed) {
        // 应用缩放因子并保存点
        int scaled_x = static_cast<int>(x * config_.image.scale_factor * 2);
        int scaled_y = static_cast<int>(y * config_.image.scale_factor * 2);
        
        std::cout << "Selected point " << (picked_points_.size() + 1) 
                  << ": x=" << scaled_x << " y=" << scaled_y << std::endl;
        picked_points_.push_back(cv::Point2f(scaled_x, scaled_y));
        
        if (picked_points_.size() == world_points_.size()) {
            solvePnP();
        }
    }
    
    adjusting_point_ = false;
}

bool CameraCalibrator::getCalibrationResult(cv::Mat& rvec, cv::Mat& tvec) const {
    if (!calibration_complete_) return false;
    
    rvec = rvec_.clone();
    tvec = tvec_.clone();
    return true;
}

void CameraCalibrator::updateRoiDisplay(int x, int y) {
    if (current_image_.empty()) return;
    
    int half_size = config_.image.roi_zoom_size / 2;
    int x_start = std::max(0, x - half_size);
    int y_start = std::max(0, y - half_size);
    int x_end = std::min(current_image_.cols, x + half_size);
    int y_end = std::min(current_image_.rows, y + half_size);
    
    if (x_end - x_start > 0 && y_end - y_start > 0) {
        cv::Mat roi = current_image_(cv::Rect(x_start, y_start, 
                                            x_end - x_start, y_end - y_start));
        cv::resize(roi, roi_image_, cv::Size(config_.window.roi_width, 
                                           config_.window.roi_height));
        
        // 绘制十字准星
        int center_x = roi_image_.cols / 2;
        int center_y = roi_image_.rows / 2;
        cv::line(roi_image_, cv::Point(center_x, center_y - 20), 
                cv::Point(center_x, center_y + 20), cv::Scalar(0, 0, 255), 2);
        cv::line(roi_image_, cv::Point(center_x - 20, center_y), 
                cv::Point(center_x + 20, center_y), cv::Scalar(0, 0, 255), 2);
            
        std::string coord_text = "x:" + std::to_string(x) + " y:" + std::to_string(y);
        cv::putText(roi_image_, coord_text, 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                   cv::Scalar(255, 255, 255), 2);
    
    }
}

void CameraCalibrator::saveCalibrationResult() {
    // 检查目录是否存在，不存在则创建
    std::string output_dir = config_.output_path.substr(0, config_.output_path.find_last_of('/'));
    
    // 创建目录（如果不存在）
    struct stat st;
    if (stat(output_dir.c_str(), &st) == -1) {
        // 目录不存在，创建它
        std::string cmd = "mkdir -p " + output_dir;
        int result = system(cmd.c_str());
        if (result != 0) {
            std::cerr << "Failed to create directory: " << output_dir << std::endl;
            return;
        }
        std::cout << "Created directory: " << output_dir << std::endl;
    }
    
    cv::FileStorage fs(config_.output_path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open output file: " << config_.output_path << std::endl;
        return;
    }
    
    fs << "world_rvec" << rvec_;
    fs << "world_tvec" << tvec_;
    fs.release();
    
    std::cout << "Calibration results saved to: " << config_.output_path << std::endl;
}

void CameraCalibrator::drawCalibrationUI() {
    if (display_image_.empty()) return;
    
    // 显示已选择的点数
    std::string point_count_text = "Points selected: " + std::to_string(picked_points_.size()) + 
                                   "/" + std::to_string(world_points_.size());
    cv::putText(display_image_, point_count_text, 
               cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1.5, 
               cv::Scalar(0, 0, 255), 2);
    
    // 显示操作提示
    cv::putText(display_image_, "Press 'n' to confirm point", 
               cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
               cv::Scalar(0, 0, 255), 2);
    cv::putText(display_image_, "Use WASD to adjust position", 
               cv::Point(50, 180), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
               cv::Scalar(0, 0, 255), 2);
    
    // 绘制已选择的点
    for (size_t i = 0; i < picked_points_.size(); ++i) {
        // 将缩放后的坐标转换回显示坐标
        int display_x = static_cast<int>(picked_points_[i].x / (config_.image.scale_factor * 2));
        int display_y = static_cast<int>(picked_points_[i].y / (config_.image.scale_factor * 2));
        
        cv::circle(display_image_, cv::Point(display_x, display_y), 10, 
                  cv::Scalar(0, 255, 0), 2);
        cv::putText(display_image_, std::to_string(i + 1), 
                   cv::Point(display_x + 15, display_y - 15), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }

    if (calibration_complete_) {
        cv::putText(display_image_, "Calibration complete! Exiting...", 
                   cv::Point(50, 250), cv::FONT_HERSHEY_SIMPLEX, 1.5, 
                   cv::Scalar(0, 255, 0), 3);
    }
}

} // namespace itl_vision