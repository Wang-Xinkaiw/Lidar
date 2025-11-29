#include "itl_vision/calibration/calibrate.h"
#include "itl_vision/calibration/calibration_params.h"
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

using namespace std::chrono_literals;

class CalibrationNode : public rclcpp::Node {
public:
    explicit CalibrationNode(const rclcpp::NodeOptions& options) 
        : Node("camera_calibration_node", options) {
        
        declare_parameters();
        auto config = load_configuration();
        
        // 初始化标定器
        calibrator_ = std::make_unique<itl_vision::CameraCalibrator>(config);
        
        std::string camera_params_path = get_parameter("calibration.camera_params_path").as_string();
        if (!calibrator_->initialize(camera_params_path)) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize camera parameters");
            return;
        }

        setup_subscribers();
        setup_windows();
        
        // 创建定时器用于显示窗口
        display_timer_ = this->create_wall_timer(33ms, [this]() { this->display_images(); });

        RCLCPP_INFO(this->get_logger(), 
            "\n"
            "    __   ______   __   \n"
            "   / /  /_  __/  / /   \n"
            "  / /    / /    / /    \n"
            " / /    / /    / /__   \n"
            "/_/    /_/    /____/   \n");
        RCLCPP_INFO(get_logger(), "Camera calibration node started");
    }

    ~CalibrationNode() {
        cv::destroyAllWindows();
    }

private:
    void declare_parameters() {
        // 世界坐标
        declare_world_points();
        
        // 窗口配置
        declare_window_params();
        
        // 图像配置
        declare_image_params();
        
        // 相机参数
        declare_calibration_params();
        
        // 话题配置
        declare_topic_params();
    }
    
    void declare_world_points() {
        this->declare_parameter("world_points.self_fortress", std::vector<double>{5.471, -7.5, 0.0});
        this->declare_parameter("world_points.self_tower", std::vector<double>{10.936, -11.161, 0.868});
        this->declare_parameter("world_points.enemy_base", std::vector<double>{25.49, -7.5, 1.24524});
        this->declare_parameter("world_points.enemy_tower", std::vector<double>{16.925, -3.625, 1.745});
        this->declare_parameter("world_points.enemy_high", std::vector<double>{20.20, -10.8, 0.8});
    }
    
    void declare_window_params() {
        this->declare_parameter("window.width", 1440);
        this->declare_parameter("window.height", 900);
        this->declare_parameter("window.pos_x", 480);
        this->declare_parameter("window.pos_y", 180);
        this->declare_parameter("window.roi_width", 400);
        this->declare_parameter("window.roi_height", 400);
    }
    
    void declare_image_params() {
        this->declare_parameter("image.width", 1536);
        this->declare_parameter("image.height", 1125);
        this->declare_parameter("image.scale_factor", 1.3333333333);
        this->declare_parameter("image.roi_zoom_size", 100);
    }
    
    void declare_calibration_params() {
        this->declare_parameter("calibration.pnp_method", "EPNP");
        this->declare_parameter("calibration.output_path", "src/ilt_vison/config/out_matrix.yaml");
        this->declare_parameter("calibration.camera_params_path", "/home/chichu/Lidar/ITL_ws/src/itl_vision/config/camera_params.yaml");
    }
    
    void declare_topic_params() {
        this->declare_parameter("topics.image_topic","image_raw");
        this->declare_parameter("topics.compressed_topic","compressed_image");
    }
    
    itl_vision::CalibrationConfig load_configuration() {
        itl_vision::CalibrationConfig config;

        load_calibration_params(config);
        load_world_points(config);
        load_window_config(config);
        load_image_config(config);
        load_topic_config(config);
        
        return config;
    }
    
    void load_calibration_params(itl_vision::CalibrationConfig& config) {
        config.camera_params_path = this->get_parameter("calibration.camera_params_path").as_string();
        config.output_path = this->get_parameter("calibration.output_path").as_string();
        config.pnp_method = this->get_parameter("calibration.pnp_method").as_string();
    }
    
    void load_world_points(itl_vision::CalibrationConfig& config) {
        auto fortress = this->get_parameter("world_points.self_fortress").as_double_array();
        auto tower = this->get_parameter("world_points.self_tower").as_double_array();
        auto base = this->get_parameter("world_points.enemy_base").as_double_array();
        auto enemy_tower = this->get_parameter("world_points.enemy_tower").as_double_array();
        auto enemy_high = this->get_parameter("world_points.enemy_high").as_double_array();
        
        config.world_points.self_fortress = cv::Point3f(fortress[0], fortress[1], fortress[2]);
        config.world_points.self_tower = cv::Point3f(tower[0], tower[1], tower[2]);
        config.world_points.enemy_base = cv::Point3f(base[0], base[1], base[2]);
        config.world_points.enemy_tower = cv::Point3f(enemy_tower[0], enemy_tower[1], enemy_tower[2]);
        config.world_points.enemy_high = cv::Point3f(enemy_high[0], enemy_high[1], enemy_high[2]);
    }
    
    void load_window_config(itl_vision::CalibrationConfig& config) {
        config.window.width = this->get_parameter("window.width").as_int();
        config.window.height = this->get_parameter("window.height").as_int();
        config.window.pos_x = this->get_parameter("window.pos_x").as_int();
        config.window.pos_y = this->get_parameter("window.pos_y").as_int();
        config.window.roi_width = this->get_parameter("window.roi_width").as_int();
        config.window.roi_height = this->get_parameter("window.roi_height").as_int();
    }
    
    void load_image_config(itl_vision::CalibrationConfig& config) {
        config.image.width = this->get_parameter("image.width").as_int();
        config.image.height = this->get_parameter("image.height").as_int();
        config.image.scale_factor = this->get_parameter("image.scale_factor").as_double();
        config.image.roi_zoom_size = this->get_parameter("image.roi_zoom_size").as_int();
    }
    
    void load_topic_config(itl_vision::CalibrationConfig& config) {
        config.topics.image_topic = this->get_parameter("topics.image_topic").as_string();
        config.topics.compressed_topic = this->get_parameter("topics.compressed_topic").as_string();
    }
    
    void setup_subscribers() {
        auto config = load_configuration();

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            config.topics.image_topic, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                handle_image_message(msg);
            });

        compressed_image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            config.topics.compressed_topic, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                handle_compressed_image_message(msg);
            });
        
        RCLCPP_INFO(this->get_logger(), "Subscribers created for topics: %s, %s", 
                   config.topics.image_topic.c_str(), config.topics.compressed_topic.c_str());
    }
    
    void handle_image_message(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            auto cv_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
            calibrator_->processImage(cv_image);
            has_new_image_ = true;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    void handle_compressed_image_message(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        try {
            std::vector<unsigned char> data(msg->data.begin(), msg->data.end());
            calibrator_->processCompressedImage(data);
            has_new_image_ = true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Compressed image processing exception: %s", e.what());
        }
    }
    
    void setup_windows() {
        auto config = load_configuration();
        
        // 创建主窗口
        cv::namedWindow("calibrate", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("calibrate", config.window.width, config.window.height);
        cv::moveWindow("calibrate", config.window.pos_x, config.window.pos_y);
        
        // 创建ROI窗口
        cv::namedWindow("ROI", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("ROI", config.window.roi_width, config.window.roi_height);
        cv::moveWindow("ROI", 0, 0);
        
        // 设置鼠标回调
        cv::setMouseCallback("calibrate", 
            [](int event, int x, int y, int flags, void* userdata) {
                auto node = static_cast<CalibrationNode*>(userdata);
                node->mouse_callback(event, x, y, flags, userdata);
            }, this);
    }
    
    void mouse_callback(int event, int x, int y, int flags, void* userdata) {
        (void)userdata; 
        calibrator_->handleMouseEvent(event, x, y, flags);
        
        // 更新ROI显示
        auto roi_image = calibrator_->getRoiImage();
        if (!roi_image.empty()) {
            cv::imshow("ROI", roi_image);
        }
    }
    
    void display_images() {
        // 检查是否需要退出
        if (calibrator_->shouldExit()) {
            RCLCPP_INFO(this->get_logger(), "Calibration complete, shutting down...");
            rclcpp::shutdown();
            return;
        }
        
        if (has_new_image_) {
            auto display_image = calibrator_->getDisplayImage();
            if (!display_image.empty()) {
                cv::imshow("calibrate", display_image);
            }
            has_new_image_ = false;
        }
        
        // 处理键盘输入
        int key = cv::waitKey(10);
        if (key != -1) {
            calibrator_->handleKeyEvent(key);
        }
    }
    
    std::unique_ptr<itl_vision::CameraCalibrator> calibrator_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_image_sub_;
    rclcpp::TimerBase::SharedPtr display_timer_;
    bool has_new_image_ = false;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    rclcpp::NodeOptions options;
    auto node = std::make_shared<CalibrationNode>(options);
    RCLCPP_INFO(node->get_logger(), "Calibration node started, spinning...");
    
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}

RCLCPP_COMPONENTS_REGISTER_NODE(CalibrationNode)