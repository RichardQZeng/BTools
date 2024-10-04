import os
import yaml
import torch
import rasterio
import numpy as np
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerConfig, SegformerForSemanticSegmentation

# cd /media/irro/All/LineFootprint/Scripts/predictions

class SeismicLinePredictionFlow:

    environment = 'local'  # Can be overridden when running the script with --environment
    username = 'irina.terenteva'

    def __init__(self):
        # Parameters for the flow
        self.patch_size = 256
        self.overlap_size = None

        self.model_path = r"I:\BERATools\AI_Line_Detection\best_segformer_epoch_14_val_loss_0.0867.pth"
        self.test_image_path = r"I:\BERATools\AI_Line_Detection\CHM.tif"
        self.output_dir = r"I:\BERATools\AI_Line_Detection"

    @property
    def base_dir(self):
        """
        Dynamically set the base directory depending on environment.
        """
        if self.environment == 'hpc':
            return f'/home/{self.username}/LineFootprint/'
        else:
            return 'I:\BERATools\AI_Line_Detection'

    def start(self):
        print(f"Starting Seismic Line Prediction with {self.environment} environment")

        base_dir = self.base_dir
        self.config_path = os.path.join(base_dir, 'config.yaml')
        print(f"Loading configuration from {self.config_path}")

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Use local variables instead of modifying flow-level attributes
        self.test_image_path = self.test_image_path or os.path.join(self.base_dir,
                                                               self.config['prediction_params']['test_image_path'])
        output_dir = self.output_dir or os.path.join(self.base_dir, self.config['prediction_params']['output_dir'])
        # model_path = self.model_path or os.path.join(self.base_dir, self.config['prediction_params']['model_path'])

        # Set patch size and stride from either the config or command-line args
        self.patch_size = self.patch_size or self.config['prediction_params']['patch_size']
        self.overlap_size = self.overlap_size or self.config['prediction_params']['overlap_size']

        # Calculate stride from overlap
        stride = self.patch_size - self.overlap_size

        # Use max_height from config for normalization
        max_height = self.config['project']['max_height']

        # Print key parameters
        print(f"\n*********\nTest Image Path: {self.test_image_path}")
        print(f"Model Path: {self.model_path}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Patch Size: {self.patch_size}")
        print(f"Stride: {stride}")
        print(f"Overlap Size: {self.config['prediction_params']['overlap_size']}")

        # Pass the resolved paths and config values to the next step
        self.resolved_test_image_path = self.test_image_path
        self.resolved_model_path = self.model_path
        self.resolved_output_dir = self.output_dir

        self.resolved_patch_size = self.patch_size
        self.max_height = max_height
        self.stride = stride

        self.predict()

    def predict(self):
        # Use the resolved directories and parameters from the start step
        print(f"Predicting with model: {self.resolved_model_path}")
        print(f"Test image path: {self.resolved_test_image_path}")
        print(f"Output directory: {self.resolved_output_dir}")

        model_config = self.config['model']
        print('\nModel configuration: ', model_config)

        # Config with optional ASPP and ResNet backbone
        config = SegformerConfig(
            num_labels=2,
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=256,
            backbone="resnet50",  # Option to use ResNet backbone
            aspp_ratios=[1, 6, 12, 18],  # ASPP for multi-scale context
            aspp_out_channels=256
        )

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create the SegFormer model with modified input channels
        segformer_model = SegformerForSemanticSegmentation(config)

        # Modify the first conv layer to accept single-channel input (grayscale)
        def modify_first_conv_layer(model):
            model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(
                in_channels=1,  # Single-channel input for grayscale
                out_channels=model.segformer.encoder.patch_embeddings[0].proj.out_channels,
                kernel_size=model.segformer.encoder.patch_embeddings[0].proj.kernel_size,
                stride=model.segformer.encoder.patch_embeddings[0].proj.stride,
                padding=model.segformer.encoder.patch_embeddings[0].proj.padding,
                bias=False
            )
            return model

        # Modify the first layer before loading the weights
        segformer_model = modify_first_conv_layer(segformer_model)

        # Load the model weights
        segformer_model.load_state_dict(torch.load(self.resolved_model_path, map_location=device))

        # Set model to evaluation mode and move to device
        segformer_model.eval()
        segformer_model.to(device)

        self.sliding_window_prediction(self.resolved_test_image_path, segformer_model, self.resolved_output_dir,
                                           self.resolved_patch_size, self.stride, self.max_height)


    def sliding_window_prediction(self, image_path, model, output_dir, patch_size, stride, max_height):
        with rasterio.open(image_path) as src:
            image = src.read(1).astype(np.float32)
            nodata_value = src.nodata
            if nodata_value is not None:
                image[image == nodata_value] = 0  # Handle nodata

            # Clip CHM values and normalize
            image = np.clip(image, 0, max_height)
            image = image / max_height

            # # Apply image processor (similar to training)
            # image = self.image_processor(images=image, return_tensors="pt")['pixel_values']

            height, width = image.shape

            # Initialize an empty array to store the full prediction probabilities
            full_prediction = np.zeros((height, width), dtype=np.float32)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            print('Predictions started')
            # Sliding window loop
            for i in tqdm(range(0, height - patch_size + 1, stride), desc="Sliding Window Row"):
                for j in tqdm(range(0, width - patch_size + 1, stride), desc="Sliding Window Col", leave=False):
                    # Extract the patch
                    patch = image[i:i + patch_size, j:j + patch_size]

                    # Convert the patch to a tensor
                    tensor_patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                    # Make predictions
                    with torch.no_grad():
                        outputs = model(pixel_values=tensor_patch)
                        logits = outputs.logits

                        # Apply softmax to get probabilities
                        probabilities = torch.softmax(logits, dim=1)

                        # Upsample to patch size and take probabilities for class 1 (seismic line)
                        upsampled_probs = torch.nn.functional.interpolate(
                            probabilities[:, 1:2, :, :], size=(patch_size, patch_size), mode="bilinear",
                            align_corners=False
                        )

                        # Convert to numpy array
                        prob_values = upsampled_probs.squeeze(0).squeeze(0).cpu().numpy()

                    # Place the predicted patch in the corresponding location of the full prediction
                    full_prediction[i:i + patch_size, j:j + patch_size] = prob_values



            output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_' + os.path.basename(
                self.resolved_model_path).replace('.pth', '')
            output_path = os.path.join(output_dir, f'{output_filename}.tif')

            with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=rasterio.float32,  # Save as float32 for continuous probabilities
                    crs=src.crs,
                    transform=src.transform
            ) as dst:
                dst.write(full_prediction, 1)

            print(f"Prediction saved to {output_path}", flush=True)

    def end(self):
        print("Prediction flow completed.")


if __name__ == '__main__':
    flow = SeismicLinePredictionFlow()
    flow.start()
