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
    import time
    start = time.time()
    for i in range(100):
        predictModel.prediction_model("以上对于我的显存依旧不够，大家有比我更好的条件的可以试试以上代码", \
                                      "适当地增加batch size（1,2,4,8,16,32...）以及图片尺寸大小(512*512...)  \
                                            经过各种资料调研，只为降低显存占用，")
    print(time.time() - start)

