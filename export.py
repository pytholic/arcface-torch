import onnx
import torch
import torch.nn as nn
from config import *
from models import *

args = Config()

if args.backbone == "resnet18":
    model = resnet_face18(use_se=args.use_se)
elif args.backbone == "resnet34":
    model = resnet34()
elif args.backbone == "resnet50":
    model = resnet50()
elif args.backbone == "mobilenetv3":
    model = MobileNetV3()
elif args.backbone == "slimnet":
    model = SlimNet()
elif args.backbone == "model-timm":
    model = create_model("mobilenetv3_small_050")

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)
model.to(device)
state = torch.load("/home/jovyan/haseeb-rnd/haseeb-data/rnd.ml.pivo-facereid/checkpoints/resnet18_83.pth")
model.load_state_dict(state)
model.eval()

input_name = ["input"]
output_name = ["output"]
input = torch.randn(1, 3, 128, 128).to(device)
torch.onnx.export(
    model.module,
    input,
    "./model.onnx",
    input_names=input_name,
    output_names=output_name,
    verbose=True,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
)


onnx_model = onnx.load("./model.onnx")
onnx.checker.check_model(onnx_model)
