from experiments.task1_pick_banana.config import TrainConfig as PickBananaTrainConfig
from experiments.task2_close_cap.config import TrainConfig as CloseCapTrainConfig

CONFIG_MAPPING = {
    "task1_pick_banana": PickBananaTrainConfig,
    "task2_insert_vial": InsertVialTrainConfig,
}
