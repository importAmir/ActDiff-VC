"""HED edge detector (Xie & Tu, 2015), used as the sketch prior for point selection."""
import numpy as np
import torch


class HEDNetwork(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(
            in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreThr = torch.nn.Conv2d(
            in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFou = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFiv = torch.nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid(),
        )

        self.load_state_dict(
            {
                strKey.replace('module', 'net'): tenWeight
                for strKey, tenWeight in torch.load(model_path).items()
            }
        )

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(
            data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device
        ).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(
            input=tenScoreOne,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False,
        )
        tenScoreTwo = torch.nn.functional.interpolate(
            input=tenScoreTwo,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False,
        )
        tenScoreThr = torch.nn.functional.interpolate(
            input=tenScoreThr,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False,
        )
        tenScoreFou = torch.nn.functional.interpolate(
            input=tenScoreFou,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False,
        )
        tenScoreFiv = torch.nn.functional.interpolate(
            input=tenScoreFiv,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False,
        )

        return self.netCombine(
            torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1)
        )


class HEDDetector:
    def __init__(self, model_path, device):
        self.device = device
        self.network = HEDNetwork(model_path).to(device).eval()

    @torch.no_grad()
    def __call__(self, image):
        """image: HxWx3 float RGB in [0, 1]. Returns the HxW edge map as uint8."""
        bgr = image[:, :, ::-1].copy()
        x = torch.from_numpy(bgr).float().to(self.device).permute(2, 0, 1).unsqueeze(0)
        edge = self.network(x)[0]
        return (edge.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)[0]
