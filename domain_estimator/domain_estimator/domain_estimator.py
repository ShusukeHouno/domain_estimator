import time
import scipy.io as scio
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
import pyrealsense2 as rs
import cv2
import threading
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class domain_estimator(Node):
    def __init__(self):
        super().__init__("domain_estimator_node")
        self.publisher_ = self.create_publisher(String, 'estimated_domain', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
    
    def timer_callback(self):
        msg = String()
        msg.data = 'Hello, ROS 2!'
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
        self.lock = threading.Lock()  # スレッド間の同期用ロック


# RealSense カメラを初期化
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


# RealSense カメラから画像を取得
def capture_and_display(pipeline, stop_event):
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        realsense_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("RealSense Image", realsense_image)
        if cv2.waitKey(1) & 0xFF == 27:  # Escキーで停止
            stop_event.set()


# モデルのロード
def load_model(snapshot_path, num_classes=196, num_domains=5, device="cpu"):
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
        mat_file_path = f'feature/{dataset_name}_features.mat'
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
def evaluation_loop(pipeline, model, features, transform, device, shared_data, stop_event):
    while not stop_event.is_set():
        # RealSenseから最新フレームを取得
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        realsense_image = np.asanyarray(color_frame.get_data())

        # PIL イメージに変換
        image = Image.fromarray(realsense_image)

        # 評価
        input_feature, distances = evaluate_image(image, model, features, transform, device)

        # 評価結果を出力
        print("\nDistances to datasets:")
        for dataset_name, distance in distances.items():
            print(f"{dataset_name}: {distance:.4f}")

        # 最も近いデータセットを特定
        if distances:
            closest_dataset = min(distances, key=distances.get)
            print(f"\nClosest dataset: {closest_dataset} with distance: {distances[closest_dataset]:.4f}")
        else:
            print("No valid datasets found for comparison.")

        # 特徴量を共有データに保存
        with shared_data.lock:
            shared_data.input_feature = input_feature

        time.sleep(1)  # 10秒待機


# 特徴量可視化用の関数
def visualize_features(shared_data, features, dataset_names, stop_event):
    while not stop_event.is_set():
        try:
            with shared_data.lock:
                input_feature = shared_data.input_feature
            if input_feature is None:
                time.sleep(1)  # 特徴量が設定されるまで待機
                continue

            all_features = []
            all_labels = []

            # データセットの特徴量を収集
            for label, (dataset_name, dataset_features) in enumerate(features.items()):
                if dataset_features.shape[1] == 2048:  # 2048次元でない場合はスキップ
                    all_features.append(dataset_features)
                    all_labels.extend([label] * dataset_features.shape[0])

            # 入力画像の特徴量を追加
            all_features.append(input_feature.reshape(1, -1))
            all_labels.append(len(features))  # 新しいラベルを付与

            all_features = np.vstack(all_features)
            all_labels = np.array(all_labels)

            # t-SNEを用いて次元削減
            print("Performing t-SNE...")
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features = tsne.fit_transform(all_features)

            # プロット
            plt.figure(figsize=(10, 8))
            for label in np.unique(all_labels):
                label_indices = np.where(all_labels == label)
                if label == len(features):
                    plt.scatter(reduced_features[label_indices, 0], reduced_features[label_indices, 1], label="Input Image", marker="*", s=200)
                else:
                    plt.scatter(reduced_features[label_indices, 0], reduced_features[label_indices, 1], label=dataset_names[label])

            plt.legend()
            plt.title("t-SNE Visualization of Features")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            plt.show(block=False)
            plt.pause(10)  # 10秒間隔で更新
            plt.clf()  # 次のプロットのためにクリア

        except Exception as e:
            print(f"Visualization error: {e}")
            break
        
# メイン処理
if __name__ == "__main__":
    try:
        # 設定
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        snapshot_path = 'snapshot/2024-11-27/extra_triplet_margin_1.0-005.pt'
        dataset_names = ['parking', 'sakaki', 'DomG', 'grass', 'parkingNocar', 'hallway','road']

        # モデルと特徴量のロード
        net = load_model(snapshot_path, device=device)
        features = load_features(dataset_names)

        # 画像の前処理設定
        transform_test = transforms.Compose([
            transforms.Resize((720, 1280)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # RealSense パイプラインの初期化
        pipeline = init_realsense()

        # スレッド停止イベント
        stop_event = threading.Event()

        # 共有データ
        shared_data = SharedData()

        # # 可視化スレッドの開始
        # vis_thread = threading.Thread(target=visualize_features, args=(shared_data, features, dataset_names, stop_event))
        # vis_thread.start()

        # 表示と評価スレッドを開始
        display_thread = threading.Thread(target=capture_and_display, args=(pipeline, stop_event))
        eval_thread = threading.Thread(target=evaluation_loop, args=(pipeline, net, features, transform_test, device, shared_data, stop_event))
        display_thread.start()
        eval_thread.start()

        # スレッドの終了を待機
        display_thread.join()
        eval_thread.join()

    except KeyboardInterrupt:
        print("停止: 評価ループを終了しました。")
        stop_event.set()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
