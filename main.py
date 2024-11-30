import sys
import cv2
from deep_translator import GoogleTranslator
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import easyocr

model = {
    'YOLOv8 Лёгкая': 'yolov8n.pt',
    'YOLOv8 Быстрая': 'yolov8s.pt',
    'YOLOv8 Средняя': 'yolov8m.pt',
    'YOLOv8 Точная': 'yolov8l.pt',
    'YOLOv8 Сверх точная': 'yolov8x.pt',
}

languages = {
    "Русский": "ru",
    "Английский": "en",
    "Французский": "fr",
    "Немецкий": "de",
    "Испанский": "es"
}


def translate_text(text, target_language):
    try:
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        return f"Ошибка перевода: {e}"


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание объектов и текста")
        self.setGeometry(100, 100, 1200, 800)

        self.model = None
        self.image_path = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.model_selector = QComboBox()
        self.model_selector.addItems(model.keys())
        self.model_selector.currentIndexChanged.connect(self.load_selected_model)
        self.layout.addWidget(self.model_selector)

        self.image_label = QLabel("Выберите изображение")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setFixedSize(1000, 600)
        self.layout.addWidget(self.image_label)

        button_layout = QHBoxLayout()

        self.select_button = QPushButton("Выбрать изображение")
        self.select_button.clicked.connect(self.select_image)
        button_layout.addWidget(self.select_button)

        self.detect_button = QPushButton("Распознать объекты")
        self.detect_button.clicked.connect(self.detect_objects)
        button_layout.addWidget(self.detect_button)

        self.ocr_button = QPushButton("Распознать текст")
        self.ocr_button.clicked.connect(self.recognize_text_with_easyocr)
        button_layout.addWidget(self.ocr_button)

        self.translate_button = QPushButton("Перевести текст")
        self.translate_button.clicked.connect(self.translate_text)
        button_layout.addWidget(self.translate_button)

        self.layout.addLayout(button_layout)

        self.language_selector = QComboBox()
        self.language_selector.addItems(languages.keys())
        self.layout.addWidget(self.language_selector)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

        self.load_selected_model()

    def load_selected_model(self):
        model_name = self.model_selector.currentText()
        model_path = model[model_name]
        self.result_text.setText(f"Загрузка модели: {model_name}...")

        try:
            if 'yolov8' in model_path:
                self.model = YOLO(f'models/{model_path}')
                self.result_text.append(f"Модель {model_name} успешно загружена.")
            else:
                self.result_text.append(f"Модель {model_name} не поддерживается.")
        except Exception as e:
            self.result_text.setText(f"Ошибка загрузки модели: {e}")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
            self.result_text.clear()

    def detect_objects(self):
        if not self.image_path:
            self.result_text.setText("Сначала выберите изображение.")
            return

        if not self.model:
            self.result_text.setText("Модель не загружена. Выберите модель из списка.")
            return

        image = cv2.imread(self.image_path)

        results = self.model(image)

        annotated_image = results[0].plot()

        height, width, channel = annotated_image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

        detected_objects = results[0].boxes.data.cpu().numpy()
        self.result_text.clear()
        if detected_objects.size > 0:
            self.result_text.append("Распознанные объекты:")
            for box in detected_objects:
                x1, y1, x2, y2, confidence, class_id = box
                class_name = self.model.names[int(class_id)]
                self.result_text.append(f"- {translate_text(class_name, 'ru')}: {confidence:.2f}, координаты: ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})")
        else:
            self.result_text.append("Объекты не обнаружены.")

    def recognize_text_with_easyocr(self):
        if not self.image_path:
            self.result_text.setText("Сначала выберите изображение.")
            return

        try:
            reader = easyocr.Reader(['ru', 'en'], gpu=False)

            results = reader.readtext(self.image_path, detail=1)

            self.result_text.clear()
            if results:
                self.result_text.append("Распознанный текст:")
                for bbox, text, confidence in results:
                    self.result_text.append(f"{text} (доверие: {confidence:.2f})")
            else:
                self.result_text.append("Текст не обнаружен.")
        except Exception as e:
            self.result_text.setText(f"Ошибка распознавания текста: {e}")

    def translate_text(self):
        current_text = self.result_text.toPlainText()
        if not current_text.strip():
            self.result_text.setText("Сначала распознайте текст.")
            return

        selected_language = self.language_selector.currentText()
        target_language = languages[selected_language]

        try:
            translated_text = translate_text(current_text, target_language)
            self.result_text.setText(f"Переведённый текст ({selected_language}):\n{translated_text}")
        except Exception as e:
            self.result_text.setText(f"Ошибка перевода текста: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
