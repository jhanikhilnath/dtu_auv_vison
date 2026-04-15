import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from auv_vision.image_enhancement import VisionEnhancerCUDA

class EnhancementNode(Node):
    def __init__(self):
        super().__init__('enhancement_node')

        # --- CUDA Hardware Checker ---
        cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()

        if cuda_device_count > 0:
            gpu_name = cv2.cuda.DeviceInfo(0).name()
            self.get_logger().info(f"CUDA Enabled: True (Detected GPU: {gpu_name})")
        else:
            self.get_logger().error(f"CUDA Enabled: False! OpenCV was not compiled with CUDA, or the container cannot see the GPU.")
        # -----------------------------

        self.subscription = self.create_subscription(
            Image,
            '/auv/camera/raw',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, '/auv/camera/enhanced', 10)
        self.bridge = CvBridge()

        # Initialize the Hybrid CPU/GPU enhancer
        self.enhancer = VisionEnhancerCUDA(gamma=1.5, clahe_clip=2.0)
        self.get_logger().info("Enhancement Node started. Processing pipeline initialized.")

    def listener_callback(self, msg):
        try:
            # 1. Convert ROS message to standard CPU NumPy array
            cpu_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # 2. Pass to our class (The class handles the GPU upload/download automatically)
            processed_frame = self.enhancer.process_frame(cpu_frame)

            # 3. Convert the downloaded result back to a ROS message
            enhanced_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            self.publisher_.publish(enhanced_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = EnhancementNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
