import time
import scipy.io as scio
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image as PILImage
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F

import cv2
import threading
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge

class domain_estimator(Node):
    def __init__(self):
        
        super().__init__("domain_estimator_node")
        self.publisher_ = self.create_publisher(String, "/estimated_domain", 10)
        self.timer = self.create_timer(5, self.timer_callback)
        self.image_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_cb, 10
        )
        self.camera_param_sub = self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self.camera_param_cb, 10
        )
        self.bridge = CvBridge()
        self.shared_data = None 
        self.image  = None
        
        # 設定
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        snapshot_path = '/home/aisl/whill_e2e_houno_test_ws/src/domain_estimator/snapshot/2024-11-30/extra_triplet_margin_1.0-005.pt'
        dataset_names = ['parking', 'sakaki', 'DomG', 'grass', 'parkingNocar', 'hallway','road']

        # モデルと特徴量のロード
        self.net = load_model(snapshot_path, device=self.device)
        self.features = load_features(dataset_names)

        
        
    def image_cb(self, msg: Image):
        """Image sub callback

        Args:
            msg (sensor_msgs.msg.Image): Image data
        """
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            # self.get_logger().info(f"Image received successfully. Shape: {self.image.shape}")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
        #self.image =  self.bridge.imgmsg_to_cv2(msg, "rgb8")
        
    def camera_param_cb(self, msg: CameraInfo):
        """CameraInfo sub callback

        Args:
            msg (CameraInfo): to get camera internal params
        """
        self.internal_param = msg.k
                
    
    def timer_callback(self):
        if self.image is not None:
           print(f"Processing image with shape: {self.image.shape}")
        
        msg = String()
        
        # 画像の前処理設定
        self.transform_test = transforms.Compose([
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # # RealSense パイプラインの初期化
        # self.pipeline = init_realsense()

        # 共有データ
        shared_data = SharedData()
        
        # ROSノードの作成
        # domain_estimator_node = domain_estimator(shared_data)

        # # 可視化スレッドの開始
        # vis_thread = threading.Thread(target=visualize_features, args=(shared_data, features, dataset_names, stop_event))
        # vis_thread.start()

        # 表示と評価スレッドを開始
        #display_thread = threading.Thread(target=capture_and_display, args=(stop_event))
        self.closest_dataset = evaluation_loop(self.image, self.net, self.features, self.transform_test, self.device, shared_data)
        #display_thread.start()
        #eval_thread.start()

        # スレッドの終了を待機
        #display_thread.join()
        #eval_thread.join()
        
        msg.data = self.closest_dataset
            
        # msg.data = self.closest_dataset or "No valid datasets"
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

# MarketNet の定義
class MarketNet(nn.Module):
    def __init__(self, num_classes=196, num_domains=7):
        super(MarketNet, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.resnet_layer = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(self.resnet_layer.children())[:-2])
        output_features = 2048

        # クラス分類用
        self.fc_class = nn.Linear(output_features, num_classes)
        self.fc_domain = nn.Linear(output_features, num_domains)

        # Batch Normalization
        self.pool_bn = nn.BatchNorm1d(output_features)
        nn.init.constant_(self.pool_bn.weight, 1)
        nn.init.constant_(self.pool_bn.bias, 0)
        nn.init.normal_(self.fc_class.weight, std=0.001)
        nn.init.constant_(self.fc_class.bias, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.pool_bn(x)
        class_output = self.fc_class(x)
        domain_output = self.fc_domain(x)
        return x, class_output, domain_output

# 評価ループと可視化スレッドで共有するデータを管理するクラス
class SharedData:
    def __init__(self):
        self.input_feature = None  # 最新の入力画像の特徴量
        self.closest_dataset = None  # 現在の最も近いデータセット

# モデルのロード
def load_model(snapshot_path, num_classes=196, num_domains=7, device="cpu"):
    model = MarketNet(num_classes, num_domains).to(device)
    if os.path.exists(snapshot_path):
        state_dict = torch.load(snapshot_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("Model loaded successfully.")
        return model
    else:
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")


# 特徴量のロード
def load_features(dataset_names):
    features = {}
    for dataset_name in dataset_names:
        mat_file_path = f'/home/aisl/whill_e2e_houno_test_ws/src/domain_estimator/feature/{dataset_name}_features.mat'
        if os.path.exists(mat_file_path):
            data = scio.loadmat(mat_file_path)
            feature_key = f'{dataset_name}_features'
            if feature_key in data:
                features[dataset_name] = data[feature_key]
                print(f"Loaded {dataset_name} features, shape: {features[dataset_name].shape}")
            else:
                print(f"Key '{feature_key}' not found in {mat_file_path}")
        else:
            print(f"File not found: {mat_file_path}")
    return features


# 入力画像の評価
def evaluate_image(image, model, features, transform, device):
    # 前処理
    input_tensor = transform(image).unsqueeze(0).to(device)
    #print(f"前処理後の画像次元数: {input_tensor.shape}")

    # 特徴量の抽出
    with torch.no_grad():
        input_feature, _, _ = model(input_tensor)
    input_feature = input_feature.cpu().numpy().squeeze()

    # 各データセットとの距離を計算
    distances = {}
    for dataset_name, dataset_features in features.items():
        if dataset_features.shape[1] != 2048:
            print(f"Skipping {dataset_name} due to mismatched feature dimensions.")
            continue
        similarity = cosine_similarity(input_feature.reshape(1, -1), dataset_features).mean()
        distances[dataset_name] = 1 - similarity

    return input_feature, distances


# 評価ループ
def evaluation_loop(input_image, model, features, transform, device, shared_data):
    #while not stop_event.is_set():
        # # RealSenseから最新フレームを取得
        # frames = pipeline.wait_for_frames()
        # color_frame = frames.get_color_frame()
        # if not color_frame:
        #     continue
        # realsense_image = np.asanyarray(color_frame.get_data())

        print("evalution_loop")

        # PIL イメージに変換
        image = PILImage.fromarray(input_image)

        print("evalution_loop2")

        # 評価
        input_feature, distances = evaluate_image(image, model, features, transform, device)

        # 評価結果を出力
        print("\nDistances to datasets:")
        for dataset_name, distance in distances.items():
            print(f"{dataset_name}: {distance:.4f}")
            
        msg = String()

        # 最も近いデータセットを特定
        if distances:
            closest_dataset = min(distances, key=distances.get)
            print(f"\nClosest dataset: {closest_dataset} with distance: {distances[closest_dataset]:.4f}")
            msg.data = closest_dataset
            
        else:
            print("No valid datasets found for comparison.")
            msg.data = "No valid datasets"
            


        shared_data.input_feature = input_feature
        return closest_dataset
            

        #time.sleep(5)  # 10秒待機



        
def main(args=None):
    rclpy.init(args=args)

    estimator = domain_estimator()

    rclpy.spin(estimator)
    estimator.destroy_node()
    rclpy.shutdown()

# メイン処理
if __name__ == "__main__":
    main()
    
    


