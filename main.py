import sys
import cv2
import pytesseract
from deep_translator import GoogleTranslator
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

# Указываем путь к исполняемому файлу Tesseract (необходимо для Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\75942\PycharmProjects\final_project\Tesseract-OCR\tesseract.exe'

model = {'Лёгкая модель': 'yolov8n.pt',
         'Быстрая модель': 'yolov8s.pt',
         'Средняя модель': 'yolov8m.pt',
         'Точная модель': 'yolov8l.pt'}


def translate_to_russian(english_text):
    """Перевод текста с английского на русский."""
    translated = GoogleTranslator(source='en', target='ru').translate(english_text)
    return translated


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание объектов и текста")
        self.setGeometry(100, 100, 1200, 800)

        # Инициализируем переменные для модели и текущего изображения
        self.model = None
        self.image_path = None

        # Основной виджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Макет
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Выпадающий список для выбора модели
        self.model_selector = QComboBox()
        self.model_selector.addItems(model.keys())
        self.model_selector.currentIndexChanged.connect(self.load_selected_model)
        self.layout.addWidget(self.model_selector)

        # Метка для изображения
        self.image_label = QLabel("Выберите изображение")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setFixedSize(1000, 600)
        self.layout.addWidget(self.image_label)

        # Горизонтальный макет для кнопок
        button_layout = QHBoxLayout()

        # Кнопка для выбора изображения
        self.select_button = QPushButton("Выбрать изображение")
        self.select_button.clicked.connect(self.select_image)
        button_layout.addWidget(self.select_button)

        # Кнопка для распознавания объектов
        self.detect_button = QPushButton("Распознать объекты")
        self.detect_button.clicked.connect(self.detect_objects)
        button_layout.addWidget(self.detect_button)

        # Кнопка для распознавания текста
        self.ocr_button = QPushButton("Распознать текст")
        self.ocr_button.clicked.connect(self.recognize_text)
        button_layout.addWidget(self.ocr_button)

        self.layout.addLayout(button_layout)

        # Поле для вывода результатов
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout.addWidget(self.result_text)

        # Загрузить модель по умолчанию
        self.load_selected_model()

    def load_selected_model(self):
        """Загружает выбранную модель YOLO."""
        model_name = model[self.model_selector.currentText()]
        self.result_text.setText(f"Загрузка модели: {model_name}...")
        try:
            self.model = YOLO(f'models/{model_name}')
            self.result_text.append(f"Модель {model_name} успешно загружена.")
        except Exception as e:
            self.result_text.setText(f"Ошибка загрузки модели: {e}")

    def select_image(self):
        """Открывает диалог выбора изображения."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
            self.result_text.clear()

    def detect_objects(self):
        """Распознаёт объекты на выбранном изображении."""
        if not self.image_path:
            self.result_text.setText("Сначала выберите изображение.")
            return

        if not self.model:
            self.result_text.setText("Модель не загружена. Выберите модель из списка.")
            return

        # Загрузка изображения
        image = cv2.imread(self.image_path)

        # Распознавание объектов с использованием YOLO
        results = self.model(image)

        # Рисуем результаты на изображении
        annotated_image = results[0].plot()

        # Конвертация изображения для отображения в PyQt
        height, width, channel = annotated_image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Отображение обработанного изображения
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

        # Обработка результатов и вывод в текстовое поле
        detected_objects = results[0].boxes.data.cpu().numpy()
        self.result_text.clear()
        if detected_objects.size > 0:
            self.result_text.append("Распознанные объекты:")
            for box in detected_objects:
                x1, y1, x2, y2, confidence, class_id = box
                class_name = self.model.names[int(class_id)]
                self.result_text.append(f"- {translate_to_russian(class_name)}: {confidence:.2f}, координаты: ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})")
        else:
            self.result_text.append("Объекты не обнаружены.")

    def recognize_text(self):
        """Распознаёт текст на выбранном изображении."""
        if not self.image_path:
            self.result_text.setText("Сначала выберите изображение.")
            return

        # Загрузка изображения
        image = cv2.imread(self.image_path)

        try:
            # Предварительная обработка изображения
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            processed_image = cv2.GaussianBlur(binary, (1, 1), 0)

            # Увеличение размера изображения
            scale_percent = 150
            width = int(processed_image.shape[1] * scale_percent / 100)
            height = int(processed_image.shape[0] * scale_percent / 100)
            resized_image = cv2.resize(processed_image, (width, height), interpolation=cv2.INTER_CUBIC)

            # Распознавание текста
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(resized_image, lang='rus+eng', config=custom_config)
            self.result_text.setText("Распознанный текст:\n" + text)

        except Exception as e:
            self.result_text.setText(f"Ошибка распознавания текста: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
