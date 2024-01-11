# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import time

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import torch
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from utils import output_to_depth

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Initialize directory prefix and output path
    dir_prefix = "./"
    output_path = dir_prefix + "results"

    with torch.no_grad():

        # Load the pretrained network
        encoder = ResnetEncoder(152, False)
        loaded_dict_enc = torch.load(
            dir_prefix + "ckpts/encoder.pth",
            map_location=device,
        )

        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
        }
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

        # Load checkpoint
        loaded_dict = torch.load(
            dir_prefix + "ckpts/depth.pth",
            map_location=device,
        )
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Open the laptop webcam feed to use as input device
        cap = cv2.VideoCapture(0)

        # Initialize variable for FPS calculation
        frames = 0
        while cap.isOpened():
            # Grab time for FPS calculation
            start_time = time.time()

            # Grab a frame from the webcam
            ret, frame = cap.read()

            # Convert frame to correct shape
            raw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transpose columns of image to achieve the correct shape
            raw_img = raw_img.transpose(2, 0, 1)

            input_image = torch.from_numpy(raw_img).float().to(device)
            input_image = (input_image / 255.0).unsqueeze(0)

            # Resize input image to correct size
            input_image = torch.nn.functional.interpolate(
                input_image, (512, 512), mode="bilinear", align_corners=False
            )
            features = encoder(input_image)
            outputs = depth_decoder(features)

            out = outputs[("out", 0)]

            # Resize to original size
            out_resized = torch.nn.functional.interpolate(
                out, (512, 512), mode="bilinear", align_corners=False
            )

            # Calculate Depth
            depth = output_to_depth(out_resized, 0.1, 10)
            metric_depth = depth.cpu().numpy().squeeze()

            # Please adjust vmax for visualization. 10.0 means 10 meters which is the whole prediction range.
            normalizer = mpl.colors.Normalize(vmin=0.1, vmax=10.0)
            mapper = cm.ScalarMappable(norm=normalizer, cmap="turbo")
            colormapped_im = (mapper.to_rgba(metric_depth)[:, :, :3] * 255).astype(np.uint8)

            output_norm = cv2.flip(colormapped_im[:,:,[2,1,0]], 1) # Flip so image isn't mirrored
            frame = cv2.flip(frame, 1) # Flip so image isn't mirrored

            # Grab end time & calculate FPS
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 1 / elapsed_time

            # Draw FPS on the frame
            cv2.putText(output_norm, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the depth map in a window
            cv2.imshow('Output', output_norm)

            # Display the webcam input in a window
            cv2.imshow('Input', frame)

            # Press 'q' to stop the script and close the windows
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()