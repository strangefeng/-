"""Ultralytics 8.4.22 adapter: retain dual-head loss, select checkpoints on OTM.

The upstream training validator ignores args.end2end during training. Decode the
existing one2many outputs solely for metrics, preserving raw dual-head loss input.
"""
from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

def decode_one_to_many(head, raw):
    previous=head.end2end
    try:
        head.end2end=False  # _inference must produce xywh for NMS
        return head._inference(raw)
    finally:
        head.end2end=previous

class OneToManyValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.otm_head=model.model[-1] if self.training else None
        self.end2end=False

    def postprocess(self, preds):
        if self.training and self.otm_head.end2end:
            if not isinstance(preds,tuple) or 'one2many' not in preds[1]:
                raise RuntimeError('Pinned YOLO26 output contract changed')
            preds=decode_one_to_many(self.otm_head,preds[1]['one2many'])
        return super().postprocess(preds)

class OneToManyTrainer(DetectionTrainer):
    def get_validator(self):
        self.loss_names='box_loss','cls_loss','dfl_loss'
        return OneToManyValidator(self.test_loader,save_dir=self.save_dir,args=copy(self.args),_callbacks=self.callbacks)
