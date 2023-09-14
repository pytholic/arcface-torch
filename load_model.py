import torch
from torchsummary import summary

from models import *
from utils import compute_model_complexity

if __name__ == "__main__":
    model = create_model("mobilenetv3_small_050")
    x = (1, 3, 128, 128)
    x = torch.randn(x)
    out = model(x)

    print(out.shape)

    params, flops = compute_model_complexity(model, (1, 3, 128, 128))

    print("Total params: %.2fM" % (params / 1000000.0))
    print("Total flops: %.2fM" % (flops / 1000000.0))
