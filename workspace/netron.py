import torch
from models.PoseNet import PoseNet

model = PoseNet()
# model.eval()
model.train()  # Ensure training mode is set

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "posenet.onnx",
    input_names=["studentID_301416917"],
    # output_names=["xyz", "wpqr"],
    output_names=["loss1_xyz", "loss1_wpqr", "loss2_xyz", "loss2_wpqr", "loss3_xyz", "loss3_wpqr"],
    opset_version=11
)

print("Model exported to posenet.onnx")
