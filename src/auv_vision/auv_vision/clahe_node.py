import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class OpenCLCLAHEEnhancer:
    """Apply CLAHE using OpenCL"""

    def __init__(self, clip_limit: float = 3.0, tile_grid_size=(8, 8)):
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply(self, image):
        # Convert to UMat
        umat_image = cv2.UMat(image)

        # separate l,a,b
        lab = cv2.cvtColor(umat_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # apply CLAHE
        l = self._clahe.apply(l)

        # merge
        lab = cv2.merge([l, a, b])
        result_umat = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # convert to normal np array
        return result_umat.get()


class CLAHERosNode(Node):
    def __init__(self):
        super().__init__('clahe_enhancement_node')

        # sub and pub
        self.subscription = self.create_subscription(
            Image,
            '/auv/camera/raw',
            self.image_callback,
            10)
        self.publisher_ = self.create_publisher(
            Image, '/auv/camera/enhanced', 10)

        self.bridge = CvBridge()
        self.enhancer = OpenCLCLAHEEnhancer()

        # OpenCL in OpenCV
        cv2.ocl.setUseOpenCL(True)
        self.get_logger().info(f"OpenCV OpenCL: {cv2.ocl.useOpenCL()}")

    def image_callback(self, msg):
        # convert msg to img
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # apply above set algorithm
        enhanced_image = self.enhancer.apply(cv_image)

        # tonvert to ros msg and publish
        enhanced_msg = self.bridge.cv2_to_imgmsg(
            enhanced_image, encoding='bgr8')
        self.publisher_.publish(enhanced_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CLAHERosNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
