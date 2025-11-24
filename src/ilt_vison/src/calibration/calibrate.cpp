#include "src/ilt_vison/include/ilt_vison/calibration/calibrate.h"

namespace itl_vision {

CameraCalibrator::CameraCalibrator(const CalibrationConfig& config) 
    : config_(config), world_points_(config.world_points.toVector()) {}

bool CameraCalibrator::initialize(const std::string& camera_params_path) {
    cv::FileStorage fs(camera_params_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return false;
    }
    
    fs["camera_matrix"] >> camera_matrix_;
    fs["dist_coeffs"] >> dist_coeffs_;
    fs.release();
    
    is_initialized_ = true;
    return true;
}

void CameraCalibrator::processImage(const cv::Mat& input_image) {
    cv::resize(input_image, current_image_, 
               cv::Size(config_.image.width, config_.image.height));
    
    display_image_ = current_image_.clone();
    
    if (is_calibrating_) {
        drawCalibrationUI();
    } else {
        cv::putText(display_image_, "Press Enter to Calibrate !!!", 
                   cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 3, 
                   cv::Scalar(0, 0, 255), 2);
    }
}

void CameraCalibrator::solvePnP() {
    if (picked_points_.size() != world_points_.size()) {
        return;
    }
    
    int method = cv::SOLVEPNP_EPNP;
    if (config_.pnp_method == "ITERATIVE") {
        method = cv::SOLVEPNP_ITERATIVE;
    }
    
    bool success = cv::solvePnP(world_points_, picked_points_, 
                               camera_matrix_, dist_coeffs_, 
                               rvec_, tvec_, false, method);
    
    if (success) {
        saveCalibrationResult();
        picked_points_.clear();
        is_calibrating_ = false;
    }
}

} // namespace itl_vision