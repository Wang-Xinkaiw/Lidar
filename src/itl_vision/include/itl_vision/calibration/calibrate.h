#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <functional>
#include "calibration_params.h"

namespace itl_vision {

class CameraCalibrator {
public:
    using MouseCallback = std::function<void(int, int, int, int, void*)>;

    explicit CameraCalibrator(const CalibrationConfig& config);
    ~CameraCalibrator() = default;

    bool initialize(const std::string& camera_params_path);
    
    void processImage(const cv::Mat& input_image);
    void processCompressedImage(const std::vector<unsigned char>& compressed_data);
    
    void startCalibration();
    void cancelCalibration();
    
    void handleMouseEvent(int event, int x, int y, int flags);
    void handleKeyEvent(int key);
    
    // Getters
    cv::Mat getDisplayImage() const { return display_image_; }
    cv::Mat getRoiImage() const { return roi_image_; }
    bool getCalibrationResult(cv::Mat& rvec, cv::Mat& tvec) const;
    bool isCalibrating() const { return is_calibrating_; }
    bool isCalibrationComplete() const { return calibration_complete_; }
    bool shouldExit() const { return should_exit_; }
    
    void setMouseCallback(const MouseCallback& callback) { mouse_callback_ = callback; }

private:
    void solvePnP();
    void updateRoiDisplay(int x, int y);
    void saveCalibrationResult();
    void drawCalibrationUI();
    void adjustPointWithKeyboard(int& x, int& y);
    
    CalibrationConfig config_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    cv::Mat rvec_;
    cv::Mat tvec_;
    
    cv::Mat current_image_;
    cv::Mat display_image_;
    cv::Mat roi_image_;
    
    std::vector<cv::Point2f> picked_points_;
    std::vector<cv::Point3f> world_points_;

    MouseCallback mouse_callback_;
    
    bool is_calibrating_ = false;
    bool is_initialized_ = false;
    bool calibration_complete_ = false;
    bool should_exit_ = false;
    bool adjusting_point_ = false;
    int current_adjust_x_ = 0;
    int current_adjust_y_ = 0;
};

} // namespace itl_vision