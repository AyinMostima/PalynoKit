# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

import torch

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_results
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first




from torchvision.ops import focal_loss
TORCHVISION_FOCAL_LOSS_AVAILABLE = True
LOGGER.info("Using torchvision.ops.focal_loss.")



class ClassificationTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a classification model.

    This trainer handles the training process for image classification tasks, supporting both YOLO classification models
    and torchvision models.

    Attributes:
        model (ClassificationModel): The classification model to be trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (List[str]): Names of the loss functions used during training.
        validator (ClassificationValidator): Validator instance for model evaluation.

    Methods:
        set_model_attributes: Set the model's class names from the loaded dataset.
        get_model: Return a modified PyTorch model configured for training.
        setup_model: Load, create or download model for classification.
        build_dataset: Create a ClassificationDataset instance.
        get_dataloader: Return PyTorch DataLoader with transforms for image preprocessing.
        preprocess_batch: Preprocess a batch of images and classes.
        progress_string: Return a formatted string showing training progress.
        get_validator: Return an instance of ClassificationValidator.
        label_loss_items: Return a loss dict with labelled training loss items.
        plot_metrics: Plot metrics from a CSV file.
        final_eval: Evaluate trained model and save validation results.
        plot_training_samples: Plot training samples with their annotations.

    Examples:
        >>> from ultralytics.models.yolo.classify import ClassificationTrainer
        >>> args = dict(model="yolo11n-cls.pt", data="imagenet10", epochs=3)
        >>> trainer = ClassificationTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize a ClassificationTrainer object.

        This constructor sets up a trainer for image classification tasks, configuring the task type and default
        image size if not specified.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Examples:
            >>> from ultralytics.models.yolo.classify import ClassificationTrainer
            >>> args = dict(model="yolo11n-cls.pt", data="imagenet10", epochs=3)
            >>> trainer = ClassificationTrainer(overrides=args)
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

        self.fl_gamma = self.args.fl_gamma if hasattr(self.args, 'fl_gamma') else 2.0
        self.fl_alpha = self.args.fl_alpha if hasattr(self.args, 'fl_alpha') else -1  # 默认为 -1 (不使用 alpha)
        LOGGER.info(f"Focal Loss parameters: gamma={self.fl_gamma}, alpha={self.fl_alpha}")

    def set_model_attributes(self):
        """Set the YOLO model's class names from the loaded dataset."""
        self.model.names = self.data["names"]

    # def get_model(self, cfg=None, weights=None, verbose=True):
    #     """
    #     Return a modified PyTorch model configured for training YOLO.
    #
    #     Args:
    #         cfg (Any): Model configuration.
    #         weights (Any): Pre-trained model weights.
    #         verbose (bool): Whether to display model information.
    #
    #     Returns:
    #         (ClassificationModel): Configured PyTorch model for classification.
    #     """
    #     model = ClassificationModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
    #     if weights:
    #         model.load(weights)
    #
    #     for m in model.modules():
    #         if not self.args.pretrained and hasattr(m, "reset_parameters"):
    #             m.reset_parameters()
    #         if isinstance(m, torch.nn.Dropout) and self.args.dropout:
    #             m.p = self.args.dropout  # set dropout
    #     for p in model.parameters():
    #         p.requires_grad = True  # for training
    #     return model

    def setup_model(self):
        """
        Load, create or download model for classification tasks.
        """
        import torchvision  # scope for faster 'import ultralytics'

        # --- 注意：这里的逻辑如果加载 torchvision 模型，替换 loss 可能需要在之后 ---
        # 如果模型是直接从 torchvision 加载的，它可能没有 'criterion' 属性
        # Focal Loss 的替换主要针对基于 ultralytics.nn.tasks.ClassificationModel 的模型
        is_torchvision_model = False
        if str(self.model) in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[self.model](
                weights="IMAGENET1K_V1" if self.args.pretrained else None
            )
            # 对于 torchvision 模型，损失函数通常在训练循环中外部应用，而不是模型内部属性
            # 如果你想对 torchvision 模型使用 Focal Loss，需要在 BaseTrainer 的训练步骤中处理
            # 但这里的代码是修改 ClassificationModel 的，所以我们主要关注 else 分支
            is_torchvision_model = True
            ckpt = None
        else:
            # 这是加载 YOLOv8-cls 或自定义模型的情况，会调用 get_model
            ckpt = super().setup_model()  # 这会调用上面的 get_model 方法

        # 确保模型输出维度正确
        # 对于 torchvision 模型，需要手动修改最后一层
        if is_torchvision_model:
            ClassificationModel.reshape_outputs(self.model, self.data["nc"])
            LOGGER.warning(
                "Loaded a torchvision model. Focal Loss replacement in get_model() might not apply directly. "
                "Loss for torchvision models is typically handled externally in the training loop.")
            # 如果确实想对 torchvision 模型强制使用内部 criterion (不推荐)，可以在这里尝试添加
            # try:
            #     focal_loss_instance = FocalLoss(alpha=self.fl_alpha, gamma=self.fl_gamma, reduction='mean')
            #     self.model.criterion = focal_loss_instance # 尝试添加属性
            # except Exception as e:
            #     LOGGER.error(f"Could not attach criterion to torchvision model: {e}")

        return ckpt

    # def setup_model(self):
    #     """
    #     Load, create or download model for classification tasks.
    #
    #     Returns:
    #         (Any): Model checkpoint if applicable, otherwise None.
    #     """
    #     import torchvision  # scope for faster 'import ultralytics'
    #
    #     if str(self.model) in torchvision.models.__dict__:
    #         self.model = torchvision.models.__dict__[self.model](
    #             weights="IMAGENET1K_V1" if self.args.pretrained else None
    #         )
    #         ckpt = None
    #     else:
    #         ckpt = super().setup_model()
    #     ClassificationModel.reshape_outputs(self.model, self.data["nc"])
    #     return ckpt

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Create a ClassificationDataset instance given an image path and mode.

        Args:
            img_path (str): Path to the dataset images.
            mode (str): Dataset mode ('train', 'val', or 'test').
            batch (Any): Batch information (unused in this implementation).

        Returns:
            (ClassificationDataset): Dataset for the specified mode.
        """
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Return a modified PyTorch model configured for training YOLO.
        Modifies the model's criterion to use Focal Loss.
        """
        # --- 修改：获取模型后，替换损失函数 ---
        # 1. 先调用原始方法获取模型
        model = ClassificationModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        # 2. 模型参数设置 (保持不变)
        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training

        # 3. **替换模型内部的损失函数**
        try:
            # 假设 ClassificationModel 内部的损失函数属性名为 criterion
            # 创建 FocalLoss 实例，使用从 args 获取的参数
            focal_loss_instance = FocalLoss(alpha=self.fl_alpha, gamma=self.fl_gamma, reduction='mean')
            # **关键步骤：替换模型内部的 criterion**
            model.criterion = focal_loss_instance
            LOGGER.info(
                f"Successfully replaced model.criterion with FocalLoss (gamma={self.fl_gamma}, alpha={self.fl_alpha}).")
        except AttributeError:
            LOGGER.error("Failed to replace model.criterion. The attribute name might be different.")
            # 如果失败，你可能需要检查 ClassificationModel 的源代码，确定损失函数属性的确切名称
            # 可能是 self.loss_fn, self.loss 等
        except Exception as e:
            LOGGER.error(f"An error occurred while replacing the loss function: {e}")

        return model


    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Return PyTorch DataLoader with transforms to preprocess images.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train', 'val', or 'test' mode.

        Returns:
            (torch.utils.data.DataLoader): DataLoader for the specified dataset and mode.
        """
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode)

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)
        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """Returns an instance of ClassificationValidator for validation."""
        self.loss_names = ["loss"]
        return yolo.classify.ClassificationValidator(
            self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labelled training loss items tensor.

        Args:
            loss_items (torch.Tensor, optional): Loss tensor items.
            prefix (str): Prefix to prepend to loss names.

        Returns:
            (Dict[str, float] | List[str]): Dictionary of loss items or list of loss keys if loss_items is None.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)  # save results.png

    def final_eval(self):
        """Evaluate trained model and save validation results."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def plot_training_samples(self, batch, ni):
        """
        Plot training samples with their annotations.

        Args:
            batch (Dict[str, torch.Tensor]): Batch containing images and class labels.
            ni (int): Number of iterations.
        """
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
