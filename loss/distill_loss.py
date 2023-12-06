import torch
from torch.nn import functional as F
from torch.cuda import amp


class DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, outputs, feat, inputs, labels, target_cam, target_view):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        base_loss = self.base_criterion(outputs, feat, labels, target_cam)
        if self.distillation_type == 'none':
            return base_loss

        # don't backprop through the teacher
        with amp.autocast(enabled=True):
            with torch.no_grad():
                teacher_outputs, _ = self.teacher_model(inputs, labels, cam_label=target_cam, view_label=target_view)

        if self.distillation_type == 'soft':
            T = self.tau
            # if isinstance(outputs, list):
            #     outputs = outputs[0]
            #     teacher_outputs = teacher_outputs[0]
            distillation_losses = []
            for i, (output, teacher_output) in enumerate(zip(outputs, teacher_outputs)):
                loss = F.kl_div(
                    F.log_softmax(output / T, dim=1),
                    F.log_softmax(teacher_output / T, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (T * T) / output.numel()
                distillation_losses.append(loss)
            distillation_loss = sum(distillation_losses) / len(distillation_losses)

        elif self.distillation_type == 'hard':
            # distillation_loss = F.cross_entropy(outputs, teacher_outputs.argmax(dim=1))
            # print('teacher:', teacher_outputs[0].max(1)[1].shape)
            # print('label:', labels.shape)
            distillation_loss = F.cross_entropy(outputs[0], teacher_outputs[0].max(1)[1])
        loss = base_loss + distillation_loss * self.alpha
        return loss