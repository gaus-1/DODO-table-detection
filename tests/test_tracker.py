import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import TableTracker

def test_tracker_debounce_empty_to_occupied():
    """Тест: переход из EMPTY в OCCUPIED только после debounce-кадров."""
    roi = (100, 100, 200, 200)
    fps = 10
    debounce_seconds = 0.2  # 2 кадра задержки
    
    tracker = TableTracker(roi=roi, fps=fps, debounce_seconds=debounce_seconds)
    assert tracker.current_state == TableTracker.STATE_EMPTY
    
    # Кадр 1: человек вошел в зону (1 кадр)
    state, changed = tracker.process_frame([(150, 150, 250, 250)])
    assert state == TableTracker.STATE_EMPTY  # Статус еще не изменился
    assert not changed
    
    # Кадр 2: человек все еще в зоне -> смена статуса
    state, changed = tracker.process_frame([(150, 150, 250, 250)])
    assert state == TableTracker.STATE_OCCUPIED
    assert changed

def test_tracker_debounce_occupied_to_empty():
    """Тест: переход из OCCUPIED в EMPTY после ухода."""
    roi = (10, 10, 50, 50)
    tracker = TableTracker(roi=roi, fps=10, debounce_seconds=0.1) # 1 кадр задержки
    
    # Искусственно ставим статус
    tracker.current_state = TableTracker.STATE_OCCUPIED
    
    # Кадр 1: человек ушел из зоны (0 кадров пересечения)
    state, changed = tracker.process_frame([(100, 100, 150, 150)])
    # Так как debounce = 1 кадр, смена статуса произойдет при первом же пропуске 
    # (по логике >= debounce_frames).
    assert state == TableTracker.STATE_EMPTY
    assert changed

def test_overlap_logic():
    """Проверка правильности вычисления пересечения BBox-ов."""
    roi = (0, 0, 100, 100)
    tracker = TableTracker(roi=roi, fps=30, debounce_seconds=1.0)
    
    # Полностью внутри
    assert tracker._check_overlap((10, 10, 90, 90)) is True
    
    # Пересекается краем
    assert tracker._check_overlap((50, 50, 150, 150)) is True
    
    # Снаружи (справа снизу)
    assert tracker._check_overlap((200, 200, 300, 300)) is False
    
    # Снаружи (слева сверху)
    assert tracker._check_overlap((-50, -50, -10, -10)) is False
