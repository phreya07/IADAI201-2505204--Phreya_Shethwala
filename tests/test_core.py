import numpy as np
from core import load_layout, predict_slots, prepare_batch, summarize

def layout():
    return load_layout(b'{"slots":[{"id":"A1","points":[[0,0],[1,0],[1,1],[0,1]]}]}')

def test_summary_thresholds():
    image=np.full((224,224,3),128,dtype=np.uint8)
    results=predict_slots(image,layout(),None,.99)
    assert summarize(results)["available"] == 1

def test_invalid_layout():
    try: load_layout(b'{"slots":[]}')
    except ValueError: return
    raise AssertionError("Invalid layout was accepted")

def test_inference_does_not_double_normalize():
    crop = np.full((100, 100, 3), 200, dtype=np.uint8)
    batch = prepare_batch([crop])
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert np.isclose(batch.mean(), 200.0)

def test_real_model_probability_direction():
    class FakeModel:
        def predict(self, batch, verbose=0):
            assert batch.min() >= 0 and batch.max() <= 255
            return np.array([[0.90]], dtype=np.float32)
    image = np.full((224,224,3),128,dtype=np.uint8)
    results = predict_slots(image, layout(), FakeModel(), 0.50)
    assert results[0].status == "Occupied"
    assert results[0].confidence > 0.89
