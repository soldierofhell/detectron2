import time
import logging
from detectron2.engine import DefaultTrainer, SimpleTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from apex import amp


class ApexTrainer(DefaultTrainer):
    def __init__(self, cfg, optimization_level):
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        model, optimizer = amp.initialize(model, optimizer, opt_level=optimization_level)
        data_loader = self.build_train_loader(cfg)

        SimpleTrainer.__init__(self, model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def run_step(self):
        assert self.model.training
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        loss_dict = self.model(data)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)
        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)
        self.optimizer.zero_grad()
        with amp.scale_loss(losses, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        self.optimizer.step()
