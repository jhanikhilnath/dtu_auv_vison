import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

from auv_vision.image_enhancement import VisionEnhancer


class EnhancementNode(Node):
    def __init__(self):
        super().__init__('enhancement_node')

        cv2.ocl.setUseOpenCL(True)
        self.get_logger().info(f"OpenCL Enabled: {cv2.ocl.useOpenCL()}")

        self.subscription = self.create_subscription(
            Image,
            '/auv/camera/raw',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(
            Image, '/auv/camera/enhanced', 10)
        self.bridge = CvBridge()

        self.enhancer = VisionEnhancer(gamma=1.5, clahe_clip=2.0)

    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if cv2.ocl.useOpenCL():
                frame = cv2.UMat(frame)

            processed_frame = self.enhancer.process_frame(frame)

            if isinstance(processed_frame, cv2.UMat):
                processed_frame = processed_frame.get()

            enhanced_msg = self.bridge.cv2_to_imgmsg(processed_frame, "bgr8")
            self.publisher_.publish(enhanced_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process frame: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = EnhancementNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
