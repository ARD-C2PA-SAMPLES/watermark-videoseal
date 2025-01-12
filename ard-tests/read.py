# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import torchvision
import videoseal

parser = argparse.ArgumentParser(description="Video Watermarking")
parser.add_argument(
    "--input_file", type=str, required=True, help="Path to the input mp4 file"
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the VideoSeal model
video_model = videoseal.load("videoseal")
video_model.eval()
video_model.to(device)
video, audio, info = torchvision.io.read_video(args.input_file, output_format="TCHW")
fps = info["video_fps"]
# Normalize the video frames to the range [0, 1] and trim to 1 second
video = video.float() / 255.0


# Detect watermarks in the video
with torch.no_grad():
    msg_extracted = video_model.extract_message(video)
print(f"Extracted message from video: {msg_extracted}")