# Image_Preprocessing

# Image Preprocessing with PyTorch and OpenCV

This repository contains a Python script for image preprocessing using PyTorch, Torchvision, OpenCV, and NumPy. The code demonstrates loading images from the CIFAR-10 dataset, applying a series of preprocessing transformations (resize, grayscale conversion, flip, rotation, normalization), and a bonus sharpening filter. It visualizes each step and converts the final images into a PyTorch tensor for potential use in deep learning models.

## Description

The script performs the following key operations:
- **Dataset Loading**: Downloads and loads the CIFAR-10 training dataset using Torchvision.
- **Image Selection**: Selects the first two images from the dataset.
- **Preprocessing Pipeline**: Applies sequential transformations: resizing to 64x64, converting to grayscale, horizontal flipping, 30-degree rotation, normalization to [0,1], and a custom sharpening filter.
- **Visualization**: Displays all preprocessing steps in a single matplotlib figure for comparison.
- **Tensor Conversion**: Stacks the final processed images into a PyTorch tensor with shape [Batch, Channel, Height, Width].

This is a self-contained example script, suitable for educational purposes or as a foundation for computer vision preprocessing pipelines.

## Requirements

- Python 3.7+
- Libraries:
  - `torch` (PyTorch)
  - `torchvision`
  - `numpy`
  - `matplotlib`
  - `opencv-python` (OpenCV)

Install dependencies using pip:
```bash
pip install torch torchvision numpy matplotlib opencv-python
```

Note: PyTorch installation may vary by system (CPU/GPU). Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for details. The script downloads CIFAR-10 automatically on first run.

## Usage

1. Clone or download the script.
2. Run the script in a Python environment (e.g., Jupyter Notebook, Python interpreter, or IDE like VS Code).
   ```bash
   python image_preprocessing.py
   ```
   Replace `image_preprocessing.py` with the actual filename.

3. The script will:
   - Download CIFAR-10 if not present (stored in `./data`).
   - Process the first two images through the pipeline.
   - Print statistics for each step (shape and value range).
   - Display a visualization grid of all steps.
   - Output the final tensor shape and range.

## Code Explanation

### Dataset Loading
- Uses `torchvision.datasets.CIFAR10` to load the training set. Images are PIL images by default.
- Selects the first two images and converts them to NumPy arrays for compatibility with OpenCV.

### Utility Functions
- `print_stats(name, imgs)`: Prints the shape and value range for each image in a list.
- `show_step(title, imgs, step_idx)`: Subplots images in a grid, handling grayscale (2D) and RGB (3D) arrays.

### Preprocessing Steps
1. **Resize**: Uses `torchvision.transforms.Resize` to scale images to 64x64 pixels.
2. **Grayscale**: Converts RGB images to grayscale using `cv2.cvtColor`.
3. **Horizontal Flip**: Flips images horizontally with `cv2.flip`.
4. **Rotation**: Rotates images by 30 degrees around the center using `cv2.warpAffine`.
5. **Normalization**: Scales pixel values to [0, 1] by dividing by 255.
6. **Bonus: Sharpening Filter**: Applies a custom 3x3 sharpening kernel using `cv2.filter2D` to enhance edges.

Each step appends results to a `steps` list for visualization.

### Visualization
- Creates a single figure with subplots for each step and image pair.
- Uses `plt.imshow` with appropriate colormaps (grayscale for 2D, default for RGB).

### Tensor Conversion
- Stacks the final sharpened images into a NumPy array.
- Adds a channel dimension (since images are grayscale, channel=1).
- Converts to a PyTorch tensor with shape [2, 1, 64, 64] (Batch, Channel, Height, Width).

## Output

- **Console Output**: Statistics for each preprocessing step, e.g.,
  ```
  Resize
   Image 1: shape=(64, 64, 3), range=(0, 255)
   Image 2: shape=(64, 64, 3), range=(0, 255)
  ...
  Final Tensor Shape: torch.Size([2, 1, 64, 64])
  Final Tensor Range: 0.0 1.0
  ```
  (Actual ranges depend on the images.)

- **Visualization**: A matplotlib figure showing all steps side-by-side for both images.

## Notes

- CIFAR-10 images are 32x32 RGB by default; resizing to 64x64 is for demonstration.
- The sharpening kernel is a basic Laplacian filter; adjust for different effects.
- For GPU acceleration, move the tensor to a device (e.g., `tensor_output.cuda()`).
- This code processes only two images; extend to full datasets for training.
- OpenCV operations assume images are in HWC format (Height, Width, Channels).

## License

This code is provided as-is for educational purposes. Feel free to modify and distribute under an open-source license.
