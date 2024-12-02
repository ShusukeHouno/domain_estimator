#!/usr/bin/env python3.9

# predictor depends
from .cnn_model.model.predict_model import PredictModel

# ros depends
import rclpy
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs
from rclpy.node import Node
from geometry_msgs.msg import (
    Quaternion,
    TransformStamped,
    Transform,
    Vector3Stamped,
    Vector3,
    PoseStamped,
    Pose,
)
from nav_msgs.msg import Path
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String


# other depends
import numpy as np
import cv2


class path_estimator(Node):
    def __init__(self):
        """_summary_"""
        super().__init__("path_estimator_node")

        self.dbg_cnt = 0

        self.domain_weights = {
            'parking': "/home/aisl/whill_e2e_houno_test_ws/src/domain_estimator/weight/DEBUG_nom_PATH_MODEL_1018_S_parking2_CCP_epc500",
            'sakaki': "/home/aisl/whill_e2e_houno_test_ws/src/test_bringup/test_weight/epoch100/model_name_CCP_epochs100",
            'DomG': "/home/aisl/whill_e2e_houno_test_ws/src/domain_estimator/weight/DEBUG_nom_PATH_MODEL_1015_DomG_CCP_epc500",
            'grass': "/home/aisl/whill_e2e_houno_test_ws/src/domain_estimator/weight/DEBUG_nom_PATH_MODEL_DomG_grass_CCP_epc500",
            'parkingNocar': "/home/aisl/whill_e2e_houno_test_ws/src/test_bringup/test_weight/epoch100/model_name_CCP_epochs100",
            'hallway': "/home/aisl/whill_e2e_houno_test_ws/src/domain_estimator/weight/DEBUG_nom_PATH_MODEL_hallway_11_27_CCP_epc500",
            'road': "/home/aisl/whill_e2e_houno_test_ws/src/domain_estimator/weight/DEBUG_nom_PATH_MODEL_rode_11_30_CCP_epc500"
        }

        # self.declare_parameter("model.type", "CCP")
        # self.declare_parameter(
        #     "model.weight",
        #     "/home/aisl/whill_e2e_test_ws/src/test_bringup/test_weight/epoch100/model_name_CCP_epochs100",
        # )

        # 初期値を設定
        self.current_domain = None
        initial_weight = self.domain_weights['parking']  # 初期重みを設定
        self.model = PredictModel(model_name="CCP", weight_path=initial_weight, estimation=True).get_model()

        # model_type = self.get_parameter("model.type").get_parameter_value().string_value
        # model_weight = (
        #     self.get_parameter("model.weight").get_parameter_value().string_value
        # )

        # # init prediction model
        # self.model = PredictModel(
        #     model_name=model_type, weight_path=model_weight, estimation=True
        # )
        # self.model = self.model.get_model()

        # tf settings
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # for image process
        self.internal_param = None
        # self.internal_param = []
        self.internal_param = [
            915.072,
            0,
            650.107,
            0,
            914.836,
            357.393,
        ]
        self.bridge = CvBridge()

        # init pub/sub
        self.path_pub = self.create_publisher(Path, "/estimator/result/raw_path", 10)
        self.estimation_result_img_pub = self.create_publisher(
            Image, "/estimator/result/path_image", 10
        )
        self.camera_param_sub = self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self.camera_param_cb, 10
        )
        self.image_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_cb, 10
        )
        self.domain_sub = self.create_subscription(String, "/estimated_domain", self.domain_cb, 10
        )

    def domain_cb(self, msg: String):
        """domain sub callback

        Args:
            msg (String): to get domain
        """
        print(f"-----------------------------------------------------------------------------------Receive domain: {msg.data}")
        # ドメインに対応する重みが存在するか確認
        if msg.data in self.domain_weights:
            new_weight = self.domain_weights[msg.data]
            if self.current_domain != msg.data:  # ドメインが変更された場合のみ再ロード
                self.current_domain = msg.data
                self.load_model(new_weight)
                self.get_logger().info(f"Switched model to domain: {msg.data}")
        else:
            self.get_logger().warn(f"Unknown domain: {msg.data}")

    def load_model(self, weight_path):
        """モデルの重みを再ロードする"""
        try:
            self.model = PredictModel(model_name="CCP", weight_path=weight_path, estimation=True).get_model()
            self.get_logger().info(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Model loaded with weight: {weight_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model weight: {e}")



    def camera_param_cb(self, msg: CameraInfo):
        """CameraInfo sub callback

        Args:
            msg (CameraInfo): to get camera internal params
        """
        self.internal_param = msg.k
    def image_cb(self, msg: Image):
        """Image sub callback

        Args:
            msg (sensor_msgs.msg.Image): Image data
        """
        if self.internal_param is not None:
            try:
                input_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                normalized_img = self.normalize_img(input_img)
            except Exception as err:
                self.get_logger().error(f"Caught error {err}")
                return

            # reshape for estimation
            resized_height = 128
            resized_width = 240
            normalized_img = normalized_img.reshape(1, 128, 240, 3)

            # estimate
            result = self.model.predict(normalized_img)

            # パス推定結果を使用して処理
            self.process_path_estimation(input_img, result)

    def process_path_estimation(self, input_img, result):
        """推定結果を処理"""
        height, width, _ = input_img.shape
        path = np.array(
            (
                (result[0][0] * width),
                (result[0][1] * height),
                (result[0][2] * width),
                (result[0][3] * height),
            ),
            dtype=np.uint16,
        )

        dbg_result_img = cv2.line(
            input_img,
            pt1=(int(path[0]), int(path[1])),
            pt2=(int(path[2]), int(path[3])),
            color=(0, 0, 255),
            thickness=3,
        )

        dbg_result_img_msg = self.bridge.cv2_to_imgmsg(
            dbg_result_img, encoding="rgb8"
        )

        start_point = [path[0], path[1]]
        end_point = [path[2], path[3]]

        path_data = Path()
        path_data.header.frame_id = "base_link"
        path_data.header.stamp = self.get_clock().now().to_msg()
        path_data.poses.append(self.transform_image_point(start_point))
        path_data.poses.append(self.transform_image_point(end_point))
        self.path_pub.publish(path_data)
        self.estimation_result_img_pub.publish(dbg_result_img_msg)


    # def image_cb(self, msg: Image):
    #     """Image sub callback

    #     Args:
    #         msg (sensor_msgs.msg.Image): Image data
    #     """
    #     if self.internal_param is not None:
    #         try:
    #             input_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
    #             # input_img = cv2.cvtColor(src=input_img, code=cv2.COLOR_BGR2RGB)
    #             normalized_img = self.normalize_img(input_img)
    #         except Exception as err:
    #             self.get_logger().error("Caught error {}".format(err))
    #             return

    #         # reshape for estimation
    #         resized_height = 128
    #         resized_width = 240

    #         normalized_img = normalized_img.reshape(1, 128, 240, 3)

    #         # estimate
    #         result = self.model.predict(normalized_img)

    #         print(result)
    #         height, width, _ = input_img.shape
    #         path = np.array(
    #             (
    #                 (result[0][0] * width),
    #                 (result[0][1] * height),
    #                 (result[0][2] * width),
    #                 (result[0][3] * height),
    #             ),
    #             dtype=np.uint16,
    #         )

    #         dbg_result_img = cv2.line(
    #             input_img,
    #             pt1=(int(path[0]), int(path[1])),
    #             pt2=(int(path[2]), int(path[3])),
    #             color=(0, 0, 255),
    #             thickness=3,
    #         )

    #         dbg_result_img_msg = self.bridge.cv2_to_imgmsg(
    #             dbg_result_img, encoding="rgb8"
    #         )

    #         start_point = [path[0], path[1]]
    #         end_point = [path[2], path[3]]

    #         path_data = Path()
    #         path_data.header.frame_id = "base_link"
    #         path_data.header.stamp = self.get_clock().now().to_msg()
    #         path_data.poses.append(self.transform_image_point(start_point))
    #         path_data.poses.append(self.transform_image_point(end_point))
    #         self.path_pub.publish(path_data)
    #         self.estimation_result_img_pub.publish(dbg_result_img_msg)

    def normalize_img(self, img: cv2.Mat):
        """normalize image

        Args:
            img (cv2.Mat): raw size image data

        Returns:
            np.ndarray: normalized image, 240x128
        """
        ret_img = cv2.resize(img, (240, 128))
        ret_img = np.array(ret_img, np.float32)
        ret_img /= 255.0
        return ret_img

    def transform_image_point(self, img_point: list):
        """Transform the point projected on the screen to the base_link based 3D-point.

        Args:
            img_point (list): path point projected on the screen

        Returns:
            geometry_msgs.msg.PoseStamped: PoseStamped(one of the path)
        """
        try:
            base_to_cam_transform = self.tf_buffer.lookup_transform(
                "base_link", "camera_color_optical_frame", rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn("{}".format(e.args))
            ret = PoseStamped()
            return ret
        # [fx,0,cx,fy,cy,0]
        c_offset = np.array([self.internal_param[2], self.internal_param[5]])
        img_point = np.array(img_point, dtype=np.float32)
        img_point -= c_offset

        img_point = np.array(
            [img_point[0], img_point[1], self.internal_param[0]], dtype=np.float32
        )
        print(img_point)

        # img to camera frame
        img_point /= img_point[1]  # gather img points to y = 1 plane
        img_point *= (
            base_to_cam_transform.transform.translation.z
        )  # re-scale the point by height of camera frame

        # camera frame to base_link
        base_vector = Vector3Stamped()
        base_vector.vector.x = float(img_point[0])
        base_vector.vector.y = float(img_point[1])
        base_vector.vector.z = float(img_point[2])
        base_to_cam_rot = TransformStamped()
        base_to_cam_rot.transform.rotation = base_to_cam_transform.transform.rotation

        base_vector = tf2_geometry_msgs.do_transform_vector3(
            base_vector, base_to_cam_rot
        )

        base_to_point = PoseStamped()
        base_to_point.header.frame_id = "base_link"
        base_to_point.header.stamp = self.get_clock().now().to_msg()
        base_to_point.pose.position.x = (
            base_vector.vector.x + base_to_cam_transform.transform.translation.x
        )
        base_to_point.pose.position.y = (
            base_vector.vector.y + base_to_cam_transform.transform.translation.y
        )
        base_to_point.pose.position.z = (
            base_vector.vector.z + base_to_cam_transform.transform.translation.z
        )

        return base_to_point


def main(args=None):
    rclpy.init(args=args)

    estimator = path_estimator()

    rclpy.spin(estimator)
    estimator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()