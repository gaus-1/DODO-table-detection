import argparse
import logging
import os
from typing import Any, Dict, List, Tuple

import cv2
import pandas as pd
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class ObjectDetector:
    """Обертка над детектором объектов (классификация человека)."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_people(self, frame: Any) -> List[Tuple[int, int, int, int]]:
        """Возвращает координаты (x1, y1, x2, y2) всех людей на кадре."""
        results = self.model(frame, classes=[0], verbose=False)
        boxes = []
        if not results:
            return boxes
            
        for result in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, _conf, _cls = result
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes


class TableTracker:
    """Автомат состояний для стола (свободен / занят) с фильтром дребезга."""

    STATE_EMPTY = "EMPTY"
    STATE_OCCUPIED = "OCCUPIED"

    def __init__(self, roi: Tuple[int, int, int, int], fps: float, debounce_seconds: float = 2.0):
        self.roi = roi
        self.debounce_frames = int(fps * debounce_seconds)

        self.current_state = self.STATE_EMPTY
        self.consecutive_matches = 0
        self.consecutive_misses = 0

    def _check_overlap(self, person_box: Tuple[int, int, int, int]) -> bool:
        """Проверка пересечения BBox человека и столика."""
        px1, py1, px2, py2 = person_box
        rx1, ry1, rx2, ry2 = self.roi

        if px2 < rx1 or px1 > rx2:
            return False
        if py2 < ry1 or py1 > ry2:
            return False
        return True

    def process_frame(self, people_boxes: List[Tuple[int, int, int, int]]) -> Tuple[str, bool]:
        """Обновляет состояние стола на основе списка детекций."""
        is_occupied_now = any(self._check_overlap(box) for box in people_boxes)
        state_changed = False

        if is_occupied_now:
            self.consecutive_misses = 0
            self.consecutive_matches += 1

            if self.current_state == self.STATE_EMPTY and self.consecutive_matches >= self.debounce_frames:
                self.current_state = self.STATE_OCCUPIED
                state_changed = True
        else:
            self.consecutive_matches = 0
            self.consecutive_misses += 1

            if self.current_state == self.STATE_OCCUPIED and self.consecutive_misses >= self.debounce_frames:
                self.current_state = self.STATE_EMPTY
                state_changed = True

        return self.current_state, state_changed


class AnalyticsEngine:
    """Сбор статистики и расчет среднего времени простоя столика."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def log_event(self, state: str, timestamp_sec: float) -> None:
        """Записывает событие смены состояния."""
        event_name = "APPROACH" if state == TableTracker.STATE_OCCUPIED else "EMPTY"
        self.events.append({
            "event": event_name,
            "timestamp": timestamp_sec
        })
        logging.info("Событие изменено: %s (%.2f сек)", event_name, timestamp_sec)

    def generate_report(self) -> None:
        """Воспроизводит финальный отчет с расчетом средних величин."""
        if not self.events:
            logging.info("Событий не зафиксировано.")
            return

        df = pd.DataFrame(self.events)
        print("\n--- Хронология событий ---")
        print(df.to_string(index=False))

        delays = []
        last_empty_time = None

        for _, row in df.iterrows():
            if row["event"] == "EMPTY":
                last_empty_time = row["timestamp"]
            elif row["event"] == "APPROACH" and last_empty_time is not None:
                delay = row["timestamp"] - last_empty_time
                delays.append(delay)
                last_empty_time = None

        if delays:
            avg_delay = sum(delays) / len(delays)
            print(f"\nСреднее время между уходом гостя и подходом следующего: {avg_delay:.2f} сек.")
        else:
            print("\nНедостаточно данных для расчета среднего времени (требуется полный цикл Уход -> Подход).")


class VideoProcessor:
    """Оркестратор обработки видеопотока."""

    def __init__(self, video_path: str, output_path: str, detector: ObjectDetector, analytics: AnalyticsEngine):
        self.video_path = video_path
        self.output_path = output_path
        self.detector = detector
        self.analytics = analytics

    def run(self) -> None:
        """Запуск цикла чтения, детекции и записи."""
        if not os.path.isfile(self.video_path):
            logging.error("[Security] Файл видео не найден или недопустмый путь: %s", self.video_path)
            return
            
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error("Не удалось открыть видео: %s", self.video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, first_frame = cap.read()
        if not ret:
            logging.error("Ошибка при чтении первого кадра.")
            cap.release()
            return

        logging.info("Выделите столик (ROI) на видео. Нажмите ENTER для подтверждения или 'c' для отмены.")
        roi = cv2.selectROI("Select Table", first_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select Table")

        x, y, w, h = roi
        if w == 0 or h == 0:
            logging.warning("Область не выделена, работа завершена.")
            cap.release()
            return

        roi_coords = (x, y, x + w, y + h)
        
        # ОПТИМИЗАЦИЯ ПЕРФОРМАНСА: YOLO на CPU работает очень медленно на 30+ fps.
        # Для детекции стола достаточно делать ~5 проверок в секунду.
        # Это ускоряет скрипт в 6-7 раз, не ломая debounce-логику.
        process_interval = max(1, int(fps / 5))
        effective_fps = fps / process_interval
        tracker = TableTracker(roi=roi_coords, fps=effective_fps, debounce_seconds=2.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        current_state = TableTracker.STATE_EMPTY

        logging.info("Гонка видео начата (оптимизировано, детекция каждые %d кадра, 5/sec)...", process_interval)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp_sec = frame_idx / fps

            # 1. Тяжелая детекция вызывается только на каждый N-й кадр
            if frame_idx % process_interval == 0 or frame_idx == 1:
                people_boxes = self.detector.detect_people(frame)
                current_state, state_changed = tracker.process_frame(people_boxes)

                if state_changed:
                    self.analytics.log_event(current_state, timestamp_sec)

            # 2. Отрисовка происходит ПЛАВНО на СТАРЫХ КОРДИНАТАХ каждый кадр (сохраняем вид!)
            roi_color = (0, 255, 0) if current_state == TableTracker.STATE_EMPTY else (0, 0, 255)
            rx1, ry1, rx2, ry2 = roi_coords
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_color, 3)

            text = f"Status: {current_state}"
            cv2.putText(frame, text, (rx1, ry1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, roi_color, 2)

            out.write(frame)

            if frame_idx % int(fps * 5) == 0:
                percent = (frame_idx / total_frames) * 100
                logging.info("Обработано кадров: %d/%d (%.1f%%)", frame_idx, total_frames, percent)

        cap.release()
        out.release()
        logging.info("Обработка успешно завершена.")

        self.analytics.generate_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализатор занятости столика на основе видео.")
    parser.add_argument("--video", type=str, required=True, help="Путь к исходному видео")
    parser.add_argument("--output", type=str, default="output.mp4", help="Путь к сохраняемому видео")
    args = parser.parse_args()

    detector_instance = ObjectDetector()
    analytics_instance = AnalyticsEngine()
    
    processor = VideoProcessor(
        video_path=args.video, 
        output_path=args.output, 
        detector=detector_instance, 
        analytics=analytics_instance
    )
    processor.run()
