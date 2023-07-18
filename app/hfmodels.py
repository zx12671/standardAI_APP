from sentence_transformers import CrossEncoder

# config
class Config(object):
    """配置参数"""
    def __init__(self):
        self.model_name = "csclarke/MARS-Encoder"


class PredictModel(object):
    def __init__(self):
        self.config = Config()
        self.model = CrossEncoder(self.config.model_name)

    def prediction_model(self, text1, text2):
        return self.model.predict([(text1, text2)])


predictModel = PredictModel()

if __name__ == "__main__":
    print(predictModel.prediction_model("能不能吃","可以吃"))

