import torch
import torch.nn as nn

from .convnext import convnext_tiny, convnext_small

class PoseDecoder(nn.Module):
    def __init__(self, num_frames_to_predict_for=1):
        super(PoseDecoder, self).__init__()

        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.squeeze = nn.Conv2d(768, 512, 1)
        self.pose_0 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pose_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pose_2 = nn.Conv2d(512, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        x = [self.relu(self.squeeze(f)) for f in last_features]
        x = torch.cat(x, 1)

        x = self.relu(self.pose_0(x))
        x = self.relu(self.pose_1(x))
        x = self.pose_2(x)

        x = x.mean(3).mean(2)
        pose = 0.01 * x.view(-1, 6)

        return pose

class PoseNet(nn.Module):

    def __init__(self, pretrained = True):
        super(PoseNet, self).__init__()
        self.encoder = convnext_small(num_input_images = 2, pretrained=True)
        self.decoder = PoseDecoder()

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], 1)
        features = self.encoder(x)
        pose = self.decoder([features])
        return pose
if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    model = PoseNet().cuda()
    model.train()

    tgt_img = torch.randn(4, 3, 256, 832).cuda()
    ref_imgs = [torch.randn(4, 3, 256, 832).cuda() for i in range(2)]

    pose = model(tgt_img, ref_imgs[0])

    print(pose.size())
