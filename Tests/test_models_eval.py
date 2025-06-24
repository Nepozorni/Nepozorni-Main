import pytest
from evaluate import evaluate, is_attentive
from run_model import *
from PIL import Image
import numpy as np
import os

def test_evaluate_logic_changes():
    # Different predictions should reset counter
    result1 = evaluate("Box1", "one_hand", 0, 0)
    result2 = evaluate("Box1", "no_hands", 3, 3)
    assert isinstance(result1, int)
    assert isinstance(result2, int)
    assert 0 <= result1 <= 100
    assert 0 <= result2 <= 100

def test_is_attentive_threshold():
    assert is_attentive(80.0) is True
    assert is_attentive(49.0) is False

def test_run_model_on_dummy_image():
    dummy_image = np.zeros((360, 640, 3), dtype=np.uint8)  # black image
    label, output = run_model("./Models/model-21-05-2025.pt", image=dummy_image)
    assert isinstance(label, str)
    assert isinstance(output, str)
    assert len(label) > 0  # model should return something
    assert "INFERENCE TIME" in output

def test_run_model_on_file():
    if not os.path.exists("./Tests/sample.png"):
        pytest.skip("sample.png not available for test")

    label, output = run_model("./Models/model-21-05-2025.pt", image_path="./Tests/sample.png")
    assert isinstance(label, str)
    assert "INFERENCE TIME" in output