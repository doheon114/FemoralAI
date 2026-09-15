from src.trainers.counterfactual_inpainting_femoral import CounterfactualInpaintingFemoralTrainer
from src.trainers.trainer import BaseTrainer


def build_trainer(task_name: str, *args, **kwargs) -> BaseTrainer:
    if task_name == 'counterfactual_inpainting_femoral':
        return CounterfactualInpaintingFemoralTrainer(*args, **kwargs)
    else:
        raise ValueError(f'Unsupported task provided: {task_name}')
