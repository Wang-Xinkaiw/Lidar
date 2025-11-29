#pragma once
#include <vector>
#include <opencv2/core/types.hpp>

namespace itl_vision {

struct WindowConfig {
    int width;
    int height;
    int pos_x;
    int pos_y;
    int roi_width;
    int roi_height;
};

struct ImageConfig {
    int width;
    int height;
    double scale_factor;
    int roi_zoom_size = 100;
};

struct WorldPoints {
    cv::Point3f self_fortress;//己方堡垒底部
    cv::Point3f self_tower;//己方前哨站底部
    cv::Point3f enemy_base;//敌方基地引导灯
    cv::Point3f enemy_tower;//敌方前哨战引导灯
    cv::Point3f enemy_high;//敌方高地左挡板底部
    
    std::vector<cv::Point3f> toVector() const {
        return {self_fortress, self_tower, enemy_base, enemy_tower, enemy_high};
    }
};

struct TopicConfig {
    std::string image_topic = "camera_image";
    std::string compressed_topic = "compressed_image";
};

struct CalibrationConfig {
    WindowConfig window;
    ImageConfig image;
    WorldPoints world_points;
    TopicConfig topics;

    std::string pnp_method;
    std::string output_path;
    std::string camera_params_path;
};

} // namespace itl_vision