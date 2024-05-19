from yaml import safe_load
class Config:
    def __init__(self, config_path: str) -> None:
        # PipelineSetting
        self.config = safe_load(open(config_path, 'r', encoding='utf-8'))
        self.PipelineName = self.config["InitPipeline"]["PipelineName"]
        self.ControlNetName = self.config["InitPipeline"]["ControlNetName"]
        self.DiffusionModelSavePath = self.config["InitPipeline"]["DiffusionModelSavePath"]
        # TrainSetting
        self.PretrainedModelPath = self.config["Train"]["PretrainedModelPath"]
        self.ControlNetModelPath = self.config["Train"]["ControlNetModelPath"]
        self.DatasetPath = self.config["Train"]["DatasetPath"]
        self.OriginalDatasetPath = self.config["Train"]["OriginalDatasetPath"]
        self.OutputPath = self.config["Train"]["OutputPath"]
        self.ImageSize = int(self.config["Train"]["ImageSize"])
        self.BatchSize = int(self.config["Train"]["BatchSize"])
        self.Epochs = int(self.config["Train"]["Epochs"])
        self.CheckpointStep = int(self.config["Train"]["CheckpointStep"])
        self.CheckpointLimit = int(self.config["Train"]["CheckpointLimit"])
        self.LearningRate = float(self.config["Train"]["LearningRate"])
        self.Use8BitAdam = bool(self.config["Train"]["Use8BitAdam"])
        self.EnableXformersMemoryEfficientAttention = bool(self.config["Train"]["EnableXformersMemoryEfficientAttention"])
        self.MaxSample = int(self.config["Train"]["MaxSample"])
        self.TestSample = int(self.config["Train"]["TestSample"])
        self.UsePretrainedTextEncoder = bool(self.config["Train"]["UsePretrainedTextEncoder"])
        # TextEncoderTrainSetting
        self.TextTrainEpochs = int(self.config["TextEncoderTrain"]["Epochs"])
        self.TextTrainSteps = int(self.config["TextEncoderTrain"]["TrainSteps"])
        self.TextTrainBatchSize = int(self.config["TextEncoderTrain"]["BatchSize"])
        self.TextTrainLearningRate = float(self.config["TextEncoderTrain"]["LearningRate"])
        self.TextEncoderOutputPath = self.config["TextEncoderTrain"]["OutputPath"]
        self.TextTrainDevice = self.config["TextEncoderTrain"]["Device"]
        # ValidationSetting
        self.ValidationStep = int(self.config["Validation"]["ValidationStep"])
        self.ValidationPrompt = self.config["Validation"]["ValidationPrompt"]
        self.ValidationNumber = int(self.config["Validation"]["ValidationNumber"])
        self.ValidationImage = self.config["Validation"]["ValidationImage"]
        # InjectSetting
        self.InjectType = self.config["Inject"]["InjectType"]
        self.InjectImage = self.config["Inject"]["InjectImage"]
        self.OptimizeWord = self.config["Inject"]["OptimizeWord"]
        self.OriginalWord = self.config["Inject"]["OriginalWord"]
        self.TextTrigger = self.config["Inject"]["TextTrigger"]



