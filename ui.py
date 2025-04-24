import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, \
    QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon
import cv2
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Worker:
    def __init__(self):
        self.model = None

    def load_model(self):
        model_path, _ = QFileDialog.getOpenFileName(None, "选择模型文件", "", "模型文件 (*.pt)")
        if model_path:
            self.model = YOLO(model_path)
            return self.model is not None
        return False

    def detect_image(self, image):
        results = self.model.predict(image)
        return results


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("第二小组：无人机高速公路车祸检测")
        # self.setWindowIcon(QIcon("icon.png"))
        self.setGeometry(300, 150, 800, 400)

        # 创建两个 QLabel 分别显示左右图像
        self.label1 = QLabel()
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setMinimumSize(580, 450)  # 设置大小
        self.label1.setStyleSheet('border:3px solid #6950a1; background-color: black;')  # 添加边框并设置背景颜色为黑色

        self.label2 = QLabel()
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setMinimumSize(580, 450)  # 设置大小
        self.label2.setStyleSheet('border:3px solid #6950a1; background-color: black;')  # 添加边框并设置背景颜色为黑色

        # 水平布局，用于放置左右两个 QLabel
        layout = QVBoxLayout()
        # layout.addWidget(self.label1)
        hbox_video = QHBoxLayout()
        hbox_video.addWidget(self.label1)  # 左侧显示原始图像
        hbox_video.addWidget(self.label2)  # 右侧显示检测后的图像
        layout.addLayout(hbox_video)
        self.worker = Worker()
        # 创建按钮布局
        hbox_buttons = QHBoxLayout()
        # 添加模型选择按钮
        self.load_model_button = QPushButton("📁模型选择")
        self.load_model_button.clicked.connect(self.load_model)
        self.load_model_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.load_model_button)

        # 添加图片检测按钮
        self.image_detect_button = QPushButton("💾图片检测")
        self.image_detect_button.clicked.connect(self.detect_image)
        self.image_detect_button.setEnabled(False)
        self.image_detect_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.image_detect_button)

        # 添加显示检测物体按钮
        self.display_objects_button = QPushButton("🔍显示检测物体")
        self.display_objects_button.clicked.connect(self.show_detected_objects)
        self.display_objects_button.setEnabled(False)
        self.display_objects_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.display_objects_button)

        # 添加退出按钮
        self.exit_button = QPushButton("❌退出")
        self.exit_button.clicked.connect(self.exit_application)
        self.exit_button.setFixedSize(120, 30)
        hbox_buttons.addWidget(self.exit_button)

        layout.addLayout(hbox_buttons)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.current_results = None

    def detect_image(self):
        image_path, _ = QFileDialog.getOpenFileName(None, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png)")
        if image_path:
            image = cv2.imread(image_path)
            if image is not None:
                self.current_results = self.worker.detect_image(image)
                if self.current_results:
                    annotated_image = self.current_results[0].plot()
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
                    height1, width1, channel1 = image_rgb.shape
                    bytesPerLine1 = 3 * width1
                    qimage1 = QImage(image_rgb.data, width1, height1, bytesPerLine1, QImage.Format_RGB888)
                    pixmap1 = QPixmap.fromImage(qimage1)
                    self.label1.setPixmap(pixmap1.scaled(self.label1.size(), Qt.KeepAspectRatio))

                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
                    height2, width2, channel2 = annotated_image.shape
                    bytesPerLine2 = 3 * width2
                    qimage2 = QImage(annotated_image.data, width2, height2, bytesPerLine2, QImage.Format_RGB888)
                    pixmap2 = QPixmap.fromImage(qimage2)
                    self.label2.setPixmap(pixmap2.scaled(self.label2.size(), Qt.KeepAspectRatio))

    def show_detected_objects(self):
        if self.current_results:
            det_info = self.current_results[0].boxes.cls
            object_count = len(det_info)
            object_info = f"识别到的事故数量：{object_count}\n"
            object_dict = {}
            class_names_dict = self.current_results[0].names
            for class_id in det_info:
                class_name = class_names_dict[int(class_id)]
                if class_name in object_dict:
                    object_dict[class_name] += 1
                else:
                    object_dict[class_name] = 1
            sorted_objects = sorted(object_dict.items(), key=lambda x: x[1], reverse=True)
            for obj_name, obj_count in sorted_objects:
                object_info += f"{obj_name}: {obj_count}\n"
            self.show_message_box("识别结果", object_info)
        else:
            self.show_message_box("识别结果", "未检测到物体")

    def show_message_box(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    def load_model(self):
        if self.worker.load_model():
            self.image_detect_button.setEnabled(True)
            self.display_objects_button.setEnabled(True)

    def exit_application(self):
        # 终止程序运行
        sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())