import sys
import cv2
from deep_translator import GoogleTranslator
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QComboBox, QGroupBox, QStatusBar, QTabWidget, QSlider, QSpinBox, QRadioButton, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO
from moviepy import VideoFileClip
import ffmpeg
import easyocr
import speech_recognition as sr
from pydub import AudioSegment
import os

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

light_style = """
    QMainWindow { background-color: #ffffff; }
    QLabel, QGroupBox, QPushButton, QStatusBar(), QTabWidget, QTextEdit, QComboBox, QSlider { color: #000000; }
"""

dark_style = """
    QMainWindow { background-color: #2b2b2b; }
    QLabel, QGroupBox, QPushButton, QStatusBar, QTabWidget, QTextEdit, QComboBox, QSlider, QRadioButton { color: #ffffff; }
    QGroupBox { border: 1px solid #4CAF50; }
    QPushButton { background-color: #3b3b3b; border: 1px solid #4CAF50; padding: 5px; }
    QPushButton:hover { background-color: #4CAF50; color: #000000; }
    QTextEdit, QComboBox, QSlider, QRadioButton { background-color: #3b3b3b; color: #ffffff; }
    QTabWidget::pane { background-color: #1e1e1e; border: none; }
    QTabBar::tab { background-color: #2b2b2b; color: #ffffff; padding: 8px; }
    QTabBar::tab:selected { background-color: #3b3b3b; color: #ffffff; border-bottom: 2px solid #4CAF50; }
"""


def translate_text(text, target_language):
    try:
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        return f"Ошибка перевода: {e}"


class VideoProcessingThread(QThread):
    frame_processed = pyqtSignal(QImage)
    position_changed = pyqtSignal(int)
    current_frame_data = None

    def __init__(self, model, video_path, conf_threshold, parent=None):
        super().__init__(parent)
        self.model = model
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.running = True
        self.paused = False
        self.current_frame = 0
        self.total_frames = 0
        self.writer = None

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print("Ошибка: Невозможно открыть видео.")
                return

            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            temp_output_path = "temp_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

            while cap.isOpened() and self.running:
                if self.paused:
                    self.msleep(100)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(frame, conf=self.conf_threshold)
                annotated_frame = results[0].plot()
                self.writer.write(annotated_frame)

                height, width, channel = annotated_frame.shape
                bytes_per_line = 3 * width
                qt_image = QImage(annotated_frame.data, width, height, bytes_per_line,
                                  QImage.Format_RGB888).rgbSwapped()
                self.frame_processed.emit(qt_image)

                self.current_frame += 1
                self.position_changed.emit(self.current_frame)

            cap.release()
            self.writer.release()

            # Добавляем звук к обработанному видео
            #self.add_audio_to_video(temp_output_path)

        except Exception as e:
            print(f"Ошибка во время выполнения обработки видео: {e}")

    def add_audio_to_video(self, temp_video_path):
        try:
            # The output file path where the final video with audio will be saved
            output_path = "output_video_with_audio.mp4"
            try:
                ffmpeg.input(temp_video_path).output(
                    self.video_path, vcodec='copy', acodec='aac', strict='experimental'
                ).run(capture_stdout=True, capture_stderr=True)
            except ffmpeg.Error as e:
                print("FFmpeg Error:", e.stderr.decode())
            # Input video (without audio) and audio
            video_stream = ffmpeg.input(temp_video_path)
            audio_stream = ffmpeg.input(self.video_path)


            # Output with video codec and audio codec copied
            ffmpeg.output(video_stream, audio_stream, output_path, vcodec='copy', acodec='aac',
                          strict='experimental').run()

            print(f"Video with audio saved to {output_path}")
        except Exception as e:
            print(f"Error processing video or audio: {e}")

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        self.wait()


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распознавание объектов и текста")
        self.setGeometry(0, 0, 1200, 800)

        self.model = None
        self.image_path = None
        self.video_path = None
        self.conf_threshold = 0.5
        self.input_size = 640
        self.dark_mode = False
        self.video_thread = None
        # Центральный виджет и основной макет
        self.central_widget = QTabWidget()  # Используем QTabWidget
        self.setCentralWidget(self.central_widget)

        # Вкладки
        self.image_tab = QWidget()
        self.video_tab = QWidget()
        self.settings_tab = QWidget()

        self.central_widget.addTab(self.image_tab, "Распознавание изображения")

        self.central_widget.addTab(self.video_tab, "Распознавание видео")


        self.central_widget.addTab(self.settings_tab, "Настройки")

        self.init_image_tab()
        self.init_video_tab()
        self.init_audio_tab()
        self.init_settings_tab()

        self.load_selected_model()
        self.apply_style()

    def init_image_tab(self):
        main_layout = QVBoxLayout(self.image_tab)

        # Панель выбора модели
        self.model_group = QGroupBox("Выбор модели")
        self.model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_layout = QVBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(model.keys())
        self.model_selector.currentIndexChanged.connect(self.load_selected_model)
        model_layout.addWidget(self.model_selector)
        self.model_group.setLayout(model_layout)
        main_layout.addWidget(self.model_group)

        # Панель изображения
        self.image_group = QGroupBox("Изображение")
        self.image_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        image_layout = QVBoxLayout()
        self.image_label = QLabel("Выберите изображение")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #4CAF50; background-color: #f0f0f0;")
        self.image_label.setFixedSize(1000, 550)
        image_layout.addWidget(self.image_label)
        self.image_group.setLayout(image_layout)
        main_layout.addWidget(self.image_group)

        # Кнопки действий
        self.button_group = QGroupBox("Действия")
        self.button_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("Выбрать изображение")
        self.select_button.setIcon(QIcon("icons/select.png"))
        self.select_button.clicked.connect(self.select_image)
        button_layout.addWidget(self.select_button)

        self.detect_button = QPushButton("Распознать объекты")
        self.detect_button.setIcon(QIcon("icons/detect.png"))
        self.detect_button.clicked.connect(self.detect_objects)
        button_layout.addWidget(self.detect_button)

        self.ocr_button = QPushButton("Распознать текст")
        self.ocr_button.setIcon(QIcon("icons/ocr.png"))
        self.ocr_button.clicked.connect(self.recognize_text_with_easyocr)
        button_layout.addWidget(self.ocr_button)

        self.translate_button = QPushButton("Перевести текст")
        self.translate_button.setIcon(QIcon("icons/translate.png"))
        self.translate_button.clicked.connect(self.translate_text)
        button_layout.addWidget(self.translate_button)

        self.save_text_button = QPushButton("Сохранить текст")
        self.save_text_button.clicked.connect(self.save_recognized_text)
        button_layout.addWidget(self.save_text_button)

        self.save_image_button = QPushButton("Сохранить изображение")
        self.save_image_button.clicked.connect(self.save_annotated_image)
        button_layout.addWidget(self.save_image_button)

        self.button_group.setLayout(button_layout)
        main_layout.addWidget(self.button_group)

        # Результаты
        self.result_group = QGroupBox("Результаты")
        self.result_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        result_layout = QVBoxLayout()
        self.language_selector = QComboBox()
        self.language_selector.addItems(languages.keys())
        result_layout.addWidget(self.language_selector)
        self.result_text = QTextEdit()
        self.result_text.setStyleSheet("font-size: 14px; background-color: #f9f9f9;")
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        self.result_group.setLayout(result_layout)
        main_layout.addWidget(self.result_group)

        # Панель состояния
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def init_video_tab(self):
        main_layout = QVBoxLayout(self.video_tab)
        self.model_group = QGroupBox("Выбор модели")
        self.model_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        model_layout = QVBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(model.keys())
        self.model_selector.currentIndexChanged.connect(self.load_selected_model)
        model_layout.addWidget(self.model_selector)
        self.model_group.setLayout(model_layout)
        main_layout.addWidget(self.model_group)

        # Панель видео
        self.video_group = QGroupBox("Видео")
        self.video_label = QLabel("Выберите видео")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFixedSize(1000, 550)
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        self.video_group.setLayout(video_layout)
        main_layout.addWidget(self.video_group)

        # Кнопки управления видео
        self.button_group = QGroupBox("Управление видео")
        button_layout = QHBoxLayout()

        self.select_video_button = QPushButton("Выбрать видео")
        self.select_video_button.clicked.connect(self.select_video)
        button_layout.addWidget(self.select_video_button)

        self.start_video_button = QPushButton("Запустить")
        self.start_video_button.clicked.connect(self.process_video)
        button_layout.addWidget(self.start_video_button)

        self.pause_video_button = QPushButton("Пауза")
        self.pause_video_button.clicked.connect(self.pause_video)
        button_layout.addWidget(self.pause_video_button)

        self.resume_video_button = QPushButton("Продолжить")
        self.resume_video_button.clicked.connect(self.resume_video)
        button_layout.addWidget(self.resume_video_button)

        self.seek_video_button = QPushButton("Перемотка")
        self.seek_video_button.clicked.connect(self.seek_video)
        button_layout.addWidget(self.seek_video_button)

        self.save_video_button = QPushButton("Сохранить видео")
        self.save_video_button.clicked.connect(self.save_video)
        button_layout.addWidget(self.save_video_button)

        self.stop_video_button = QPushButton("Остановить")
        self.stop_video_button.clicked.connect(self.stop_video)
        button_layout.addWidget(self.stop_video_button)

        self.save_audio_button = QPushButton("Сохранить звук")
        self.save_audio_button.clicked.connect(self.save_audio)
        button_layout.addWidget(self.save_audio_button)

        self.ocr_frame_button = QPushButton("Распознать текст на кадре")
        self.ocr_frame_button.clicked.connect(self.recognize_text_on_paused_frame)
        button_layout.addWidget(self.ocr_frame_button)

        self.kgjrgg = QPushButton

        self.button_group.setLayout(button_layout)
        main_layout.addWidget(self.button_group)

        # Слайдер для перемотки
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setRange(0, 100)
        self.video_slider.sliderReleased.connect(self.seek_to_position)
        main_layout.addWidget(self.video_slider)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def init_audio_tab(self):
        self.audio_tab = QWidget()
        main_layout = QVBoxLayout(self.audio_tab)

        # Панель действий
        self.audio_action_group = QGroupBox("Действия")
        button_layout = QHBoxLayout()

        self.select_audio_button = QPushButton("Выбрать аудио")
        self.select_audio_button.clicked.connect(self.select_audio)
        button_layout.addWidget(self.select_audio_button)

        self.recognize_audio_button = QPushButton("Распознать речь")
        self.recognize_audio_button.clicked.connect(self.recognize_audio)
        button_layout.addWidget(self.recognize_audio_button)

        self.save_audio_text_button = QPushButton("Сохранить текст")
        self.save_audio_text_button.clicked.connect(self.save_recognized_audio_text)
        button_layout.addWidget(self.save_audio_text_button)

        self.audio_action_group.setLayout(button_layout)
        main_layout.addWidget(self.audio_action_group)

        # Текст результатов
        self.audio_result_group = QGroupBox("Результаты")
        result_layout = QVBoxLayout()
        self.audio_result_text = QTextEdit()
        self.audio_result_text.setReadOnly(True)
        result_layout.addWidget(self.audio_result_text)
        self.audio_result_group.setLayout(result_layout)
        main_layout.addWidget(self.audio_result_group)

        self.central_widget.addTab(self.audio_tab, "Распознавание звука")

    def recognize_text_on_paused_frame(self):
        if not self.video_thread or not self.video_thread.paused:
            self.status_bar.showMessage("Видео не на паузе. Поставьте видео на паузу для распознавания текста.", 5000)
            return

        frame = self.video_thread.current_frame_data
        if frame is None:
            self.result_text.setText("Нет данных текущего кадра для распознавания текста.")
            return

        try:
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en', 'ru'], gpu=False)  # Adjust languages as needed
            results = reader.readtext(frame, detail=1)

            self.result_text.clear()
            if results:
                self.result_text.append("Распознанный текст:")
                for bbox, text, confidence in results:
                    self.result_text.append(f"{text} (доверие: {confidence:.2f})")
            else:
                self.result_text.append("Текст не обнаружен.")
        except Exception as e:
            self.result_text.setText(f"Ошибка распознавания текста: {e}")

    def init_settings_tab(self):
        settings_layout = QVBoxLayout(self.settings_tab)

        # Настройка порога
        conf_group = QGroupBox("Порог уверенности")
        conf_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        conf_layout = QVBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 100)
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        conf_layout.addWidget(QLabel("Установите порог уверенности:"))
        conf_layout.addWidget(self.conf_slider)
        conf_group.setLayout(conf_layout)
        settings_layout.addWidget(conf_group)

        # Переключатель режима интерфейса
        theme_group = QGroupBox("Режим интерфейса")
        theme_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        theme_layout = QHBoxLayout()
        self.light_mode_button = QRadioButton("Светлый режим")
        self.light_mode_button.setChecked(not self.dark_mode)
        self.light_mode_button.toggled.connect(self.toggle_theme)

        self.dark_mode_button = QRadioButton("Тёмный режим")
        self.dark_mode_button.setChecked(self.dark_mode)
        self.dark_mode_button.toggled.connect(self.toggle_theme)

        theme_layout.addWidget(self.light_mode_button)
        theme_layout.addWidget(self.dark_mode_button)
        theme_group.setLayout(theme_layout)
        settings_layout.addWidget(theme_group)

        settings_layout.addStretch()

    def update_conf_threshold(self, value):
        self.conf_threshold = value / 100.0
        self.status_bar.showMessage(f"Порог уверенности обновлён: {self.conf_threshold:.2f}")

    def toggle_theme(self):
        self.dark_mode = self.dark_mode_button.isChecked()
        self.apply_style()

    def apply_style(self):
        if self.dark_mode:
            self.setStyleSheet(dark_style)
            # Устанавливаем темный фон для image_label
            self.image_label.setStyleSheet("border: 2px solid #4CAF50; background-color: #2b2b2b; color: #ffffff;")
            # Устанавливаем темный фон для текстового поля результатов
            self.result_text.setStyleSheet("font-size: 14px; background-color: #3b3b3b; color: #ffffff;")
            # Устанавливаем стили для вкладок
            self.central_widget.setStyleSheet(
                "QTabWidget::pane { background-color: #1e1e1e; border: none; }"
                "QTabBar::tab { background-color: #2b2b2b; color: #ffffff; padding: 8px; }"
                "QTabBar::tab:selected { background-color: #3b3b3b; color: #ffffff; border-bottom: 2px solid #4CAF50; }"
            )
            self.status_bar.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        else:
            self.setStyleSheet(light_style)
            self.image_label.setStyleSheet("border: 2px solid #4CAF50; background-color: #f0f0f0; color: #000000;")
            self.result_text.setStyleSheet("font-size: 14px; background-color: #f9f9f9; color: #000000;")
            self.central_widget.setStyleSheet(
                "QTabWidget::pane { background-color: #ffffff; border: none; }"
                "QTabBar::tab { background-color: #f0f0f0; color: #000000; padding: 8px; }"
                "QTabBar::tab:selected { background-color: #e0e0e0; border-bottom: 2px solid #4CAF50; }"
            )
            self.status_bar.setStyleSheet("background-color: #ffffff; color: #000000;")
        self.video_label.setStyleSheet(
            "border: 2px solid #4CAF50; background-color: #2b2b2b;" if self.dark_mode else
            "border: 2px solid #4CAF50; background-color: #f0f0f0;"
        )

    def load_selected_model(self):
        model_name = self.model_selector.currentText()
        model_path = model[model_name]
        self.status_bar.showMessage(f"Загрузка модели: {model_name}...")

        try:
            self.model = YOLO(f'models/{model_path}')
            self.status_bar.showMessage(f"Модель {model_name} успешно загружена.", 5000)
        except Exception as e:
            self.status_bar.showMessage(f"Ошибка загрузки модели: {e}")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(self.image_path)
            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))
            self.result_text.clear()
            self.status_bar.showMessage("Изображение выбрано.")

    def detect_objects(self):
        if not self.image_path:
            self.result_text.setText("Сначала выберите изображение.")
            return

        if not self.model:
            self.result_text.setText("Модель не загружена. Выберите модель из списка.")
            return

        image = cv2.imread(self.image_path)
        results = self.model.predict(image, conf=self.conf_threshold, imgsz=self.input_size)
        annotated_image = results[0].plot()

        # Сохраняем аннотированное изображение для последующего сохранения
        self.annotated_image = cv2.cvtColor(annotated_image,
                                            cv2.COLOR_RGB2BGR)  # Преобразование в формат BGR для OpenCV

        # Отображаем изображение
        height, width, channel = annotated_image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

        detected_objects = results[0].boxes.data.cpu().numpy()
        self.result_text.clear()
        if detected_objects.size > 0:
            self.result_text.append("Распознанные объекты:")
            for box in detected_objects:
                x1, y1, x2, y2, confidence, class_id = box
                class_name = self.model.names[int(class_id)]
                self.result_text.append(
                    f"- {translate_text(class_name, 'ru')}: {confidence:.2f}, координаты: ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})")
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

    def select_audio(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать аудио", "", "Аудиофайлы (*.wav *.mp3)",
                                                   options=options)
        if file_path:
            self.audio_path = file_path
            self.status_bar.showMessage(f"Выбрано аудио: {os.path.basename(file_path)}", 5000)

    def recognize_audio(self):
        if not hasattr(self, 'audio_path') or not self.audio_path:
            self.status_bar.showMessage("Сначала выберите аудиофайл.", 5000)
            return

        recognizer = sr.Recognizer()
        sound = AudioSegment.from_mp3(self.audio_path)
        sound.export("result.wav", format="wav")
        try:
            with sr.AudioFile("result.wav") as source:
                audio_data = recognizer.record(source)
                recognized_text = recognizer.recognize_google(audio_data, language='ru-RU')
                self.audio_result_text.setPlainText(recognized_text)
                self.status_bar.showMessage("Распознавание речи завершено.", 5000)
        except Exception as e:
            self.audio_result_text.setPlainText(f"Ошибка распознавания: {e}")
            self.status_bar.showMessage(f"Ошибка распознавания: {e}", 5000)

    def save_recognized_audio_text(self):
        if not self.audio_result_text.toPlainText().strip():
            self.status_bar.showMessage("Нет текста для сохранения.", 5000)
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить текст", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(self.audio_result_text.toPlainText())
                self.status_bar.showMessage(f"Текст успешно сохранен в {file_path}.", 5000)
            except Exception as e:
                self.status_bar.showMessage(f"Ошибка сохранения текста: {e}", 5000)

    def save_recognized_text(self):
        if not self.result_text.toPlainText().strip():
            self.status_bar.showMessage("Нет текста для сохранения.", 5000)
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить текст", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(self.result_text.toPlainText())
                self.status_bar.showMessage(f"Текст успешно сохранен в {file_path}.", 5000)
            except Exception as e:
                self.status_bar.showMessage(f"Ошибка сохранения текста: {e}", 5000)

    def save_annotated_image(self):
        if not hasattr(self, 'annotated_image') or self.annotated_image is None:
            self.status_bar.showMessage("Нет изображения для сохранения.", 5000)
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            try:
                cv2.imwrite(file_path, self.annotated_image)
                self.status_bar.showMessage(f"Изображение успешно сохранено в {file_path}.", 5000)
            except Exception as e:
                self.status_bar.showMessage(f"Ошибка сохранения изображения: {e}", 5000)

    def save_audio(self):
        if not self.video_path or not os.path.exists(self.video_path):
            self.status_bar.showMessage("Пожалуйста, выберите видео для сохранения звука.", 5000)
            return

        # Ask for the audio output file path
        output_audio_path, _ = QFileDialog.getSaveFileName(self, "Сохранить звук", "", "Аудиофайлы (*.mp3 *.wav)")
        if not output_audio_path:
            return

        try:
            video = VideoFileClip(self.video_path)
            video.audio.write_audiofile(output_audio_path)
            self.status_bar.showMessage(f"Звук сохранен: {output_audio_path}", 5000)
        except Exception as e:
            self.status_bar.showMessage(f"Ошибка сохранения звука: {e}", 5000)

    def select_video(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать видео", "", "Видео файлы (*.mp4 *.avi *.mov)",
                                                   options=options)
        if file_path:
            self.video_path = file_path
            self.video_label.setText(f"Выбрано видео: {os.path.basename(file_path)}")
            self.status_bar.showMessage(f"Видео {os.path.basename(file_path)} успешно загружено.", 5000)

    def save_video(self):
        output_file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "", "Видео файлы (*.mp4)")
        temp_output_path = os.path.abspath("temp_video.mp4")  # Убедитесь, что путь абсолютный
        if output_file_path:
            if os.path.exists(temp_output_path):
                try:
                    os.rename(temp_output_path, output_file_path)
                    self.status_bar.showMessage(f"Видео сохранено: {output_file_path}", 5000)
                except OSError as e:
                    self.status_bar.showMessage(f"Ошибка при сохранении: {e}", 5000)
            else:
                self.status_bar.showMessage("Временный файл не найден. Видео не было создано.", 5000)

    def process_video(self):
        if not self.video_path or not os.path.exists(self.video_path):
            self.status_bar.showMessage("Пожалуйста, выберите корректное видео.", 5000)
            return

        if not self.model:
            self.status_bar.showMessage("Пожалуйста, загрузите модель YOLO.", 5000)
            return

        try:
            self.video_thread = VideoProcessingThread(self.model, self.video_path, self.conf_threshold)
            self.video_thread.frame_processed.connect(self.update_video_frame)
            self.video_thread.start()
            self.status_bar.showMessage("Обработка видео начата.", 5000)
        except Exception as e:
            self.status_bar.showMessage(f"Ошибка запуска обработки видео: {e}", 5000)

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.status_bar.showMessage("Обработка видео остановлена.", 5000)

    def update_video_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))

    def pause_video(self):
        if self.video_thread:
            self.video_thread.pause()
            self.status_bar.showMessage("Видео поставлено на паузу.", 5000)

    def resume_video(self):
        if self.video_thread:
            self.video_thread.resume()
            self.status_bar.showMessage("Продолжение обработки видео.", 5000)

    def seek_video(self):
        frame_number, ok = QInputDialog.getInt(self, "Перемотка", "Введите номер кадра:", min=0, max=10000)
        if ok and self.video_thread:
            self.video_thread.seek(frame_number)

    def update_video_slider(self, position):
        if self.video_thread and self.video_thread.total_frames:
            slider_value = int((position / self.video_thread.total_frames) * 100)
            self.video_slider.setValue(slider_value)

    def seek_to_position(self):
        slider_value = self.video_slider.value()
        if self.video_thread and self.video_thread.total_frames:
            frame_position = int((slider_value / 100) * self.video_thread.total_frames)
            self.video_thread.seek(frame_position)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec_())
