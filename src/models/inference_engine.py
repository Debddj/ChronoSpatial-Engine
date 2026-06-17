from src.data_pipeline.transforms import preprocess_frame
from src.models.cnn_extractor import CNNExtractor
from src.models.ann_regressor import ANNRegressor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class InferenceEngine:
    def __init__(self, config):
        cnn_cfg = config.get("model", {}).get("cnn_extractor", {})
        ann_cfg = config.get("model", {}).get("ann_regressor", {})
        
        self.extractor = CNNExtractor(feature_dim=cnn_cfg.get("feature_dim", 128))
        self.regressor = ANNRegressor(
            input_dim=ann_cfg.get("input_dim", 128),
            risk_threshold=ann_cfg.get("risk_threshold", 0.80)
        )
        logger.info("InferenceEngine successfully compiled and initialized.")

    def run_inference(self, frame):
        # 1. Transform frame
        preprocessed = preprocess_frame(frame)
        # 2. Extract spatial features
        features = self.extractor.extract_features(preprocessed)
        # 3. Assess temporal risk
        results = self.regressor.predict_risk(features)
        return results
