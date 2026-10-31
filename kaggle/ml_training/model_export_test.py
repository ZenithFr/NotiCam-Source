import os
import torch
import torch.nn as nn
import timm

class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding, dilation, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class HybridDeepLabV3PlusDecoder(nn.Module):
    def __init__(self, high_ch, low_ch, decoder_ch=96):
        super().__init__()
        # ASPP Lite without global pooling (to avoid MEAN CPU fallback on NNAPI)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(high_ch, decoder_ch, 1, bias=False),
            nn.BatchNorm2d(decoder_ch),
            nn.ReLU6(inplace=True)
        )
        self.aspp2 = SeparableConv2d(high_ch, decoder_ch, 3, padding=6, dilation=6)
        self.aspp3 = SeparableConv2d(high_ch, decoder_ch, 3, padding=12, dilation=12)

        self.aspp_fuse = nn.Sequential(
            nn.Conv2d(decoder_ch * 3, decoder_ch, 1, bias=False),
            nn.BatchNorm2d(decoder_ch),
            nn.ReLU6(inplace=True)
        )

        # Low-level feature projection
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU6(inplace=True)
        )

        # Fusion layers
        self.fuse = nn.Sequential(
            SeparableConv2d(decoder_ch + 48, decoder_ch, 3, padding=1),
            SeparableConv2d(decoder_ch, decoder_ch, 3, padding=1)
        )

        # Final bilateral slice coefficients: 2 output channels [alpha, depth_guide]
        self.head = nn.Conv2d(decoder_ch, 2, 1)

    def forward(self, f_low, f_high):
        # f_low: [1, low_ch, 96, 128]  (1/4 scale of 384x512)
        # f_high: [1, high_ch, 24, 32] (1/16 scale of 384x512)
        a1 = self.aspp1(f_high)
        a2 = self.aspp2(f_high)
        a3 = self.aspp3(f_high)
        aspp_out = self.aspp_fuse(torch.cat([a1, a2, a3], dim=1))

        # 4x bilinear upsample to 96x128
        aspp_up = nn.functional.interpolate(aspp_out, size=f_low.shape[2:], mode='bilinear', align_corners=False)

        low = self.low_proj(f_low)
        cat = torch.cat([aspp_up, low], dim=1)
        feat = self.fuse(cat)
        coeffs = self.head(feat) # [1, 2, 96, 128]
        return coeffs

class AuxCamPortraitNet(nn.Module):
    def __init__(self, encoder='efficientnet_lite1', decoder_ch=96):
        super().__init__()
        # Use timm to build the EfficientNet-Lite1 backbone (ReLU6, no SE)
        self.backbone = timm.create_model(
            encoder, pretrained=False, features_only=True, out_indices=(1, 3)
        )
        low_ch, high_ch = self.backbone.feature_info.channels()
        print(f"Backbone: {encoder}, low_ch={low_ch}, high_ch={high_ch}")
        self.decoder = HybridDeepLabV3PlusDecoder(high_ch, low_ch, decoder_ch)

    def forward(self, x):
        f_low, f_high = self.backbone(x)
        return self.decoder(f_low, f_high)

def export_test_stub():
    os.makedirs('assets/models', exist_ok=True)
    model = AuxCamPortraitNet(encoder='efficientnet_lite1', decoder_ch=96).eval()
    dummy = torch.randn(1, 3, 384, 512)
    with torch.no_grad():
        out = model(dummy)
        print(f"Model forward output shape: {out.shape}")

    onnx_path = 'assets/models/portrait_test.onnx'
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['rgb'], output_names=['coeffs'],
        opset_version=13, do_constant_folding=True,
        dynamo=False
    )
    print(f"Exported ONNX to {onnx_path}")

if __name__ == '__main__':
    export_test_stub()
