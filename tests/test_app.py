import app as app_module


def test_home_returns_status_message():
    assert app_module.home() == {"message": "Churn Prediction API running"}


def test_predict_returns_model_prediction(monkeypatch):
    class StubModel:
        def predict(self, data):
            assert data.tolist() == [[45, 80.0, 24, 3]]
            return [1]

    monkeypatch.setattr(app_module, "model", StubModel())

    result = app_module.predict(45, 80.0, 24, 3)

    assert result == {"churn_prediction": 1}
