#!/usr/bin/env python3
"""
Single Patient Inference Script for A4-Unet
Run inference on a specific BraTS patient folder
"""

import torch
import nibabel as nib
import numpy as np
import logging
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

from a4unet.model.a4unet import create_a4unet_model

# Configuration
CHECKPOINT_PATH = 'a4unet_huggingface.pth'
PATIENT_FOLDER = Path('./data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001')
OUTPUT_FOLDER = Path('./predictions')
INPUT_SIZE = 128

def load_patient_volumes(patient_folder):
    """
    Load all MRI modalities for a patient.
    
    Args:
        patient_folder: Path to patient directory
        
    Returns:
        dict: Dictionary with modality names and their 3D volumes
    """
    modalities = {}
    patient_name = patient_folder.name
    
    # BraTS 2020 naming convention
    for modality in ['t1', 't1ce', 't2', 'flair']:
        # Try both .nii.gz and .nii extensions
        nii_path = patient_folder / f'{patient_name}_{modality}.nii'
        nii_gz_path = patient_folder / f'{patient_name}_{modality}.nii.gz'
        
        if nii_path.exists():
            filepath = nii_path
        elif nii_gz_path.exists():
            filepath = nii_gz_path
        else:
            raise FileNotFoundError(f"Could not find {modality} file in {patient_folder}")
        
        logging.info(f"Loading {modality}: {filepath}")
        nifti = nib.load(str(filepath))
        modalities[modality] = nifti.get_fdata()
    
    return modalities

def normalize_volume(volume):
    """Normalize MRI volume to zero mean and unit variance."""
    mean = volume.mean()
    std = volume.std()
    if std > 0:
        return (volume - mean) / std
    return volume - mean

def run_inference(model, modalities, device, input_size=128):
    """
    Run inference on all slices of a patient volume.
    
    Args:
        model: Trained A4-Unet model
        modalities: Dictionary of 4 MRI modalities
        device: torch device
        input_size: Input size for model (128)
        
    Returns:
        np.ndarray: 3D prediction volume [H, W, D]
    """
    # Get volume dimensions
    h, w, num_slices = modalities['t1'].shape
    
    # Initialize prediction volume (full resolution)
    predictions_full = np.zeros((h, w, num_slices), dtype=np.uint8)
    
    model.eval()
    with torch.no_grad():
        for slice_idx in tqdm(range(num_slices), desc="Processing slices"):
            # Stack all 4 modalities for this slice
            slice_data = np.stack([
                modalities['t1'][:, :, slice_idx],
                modalities['t1ce'][:, :, slice_idx],
                modalities['t2'][:, :, slice_idx],
                modalities['flair'][:, :, slice_idx]
            ], axis=0)  # [4, H, W]
            
            # Normalize each modality separately
            for i in range(4):
                slice_data[i] = normalize_volume(slice_data[i])
            
            # Convert to tensor and add batch dimension
            slice_tensor = torch.from_numpy(slice_data).unsqueeze(0).float()  # [1, 4, H, W]
            slice_tensor = slice_tensor.to(device)
            
            # Resize to model input size
            slice_tensor = F.interpolate(
                slice_tensor, 
                size=(input_size, input_size), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Run inference
            with torch.cuda.amp.autocast(enabled=True):
                output = model(slice_tensor)
            
            # Get prediction
            pred = output.argmax(dim=1).cpu().numpy()[0]  # [128, 128]
            
            # Resize back to original resolution
            pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
            pred_full = F.interpolate(
                pred_tensor,
                size=(h, w),
                mode='nearest'
            ).squeeze().numpy().astype(np.uint8)
            
            predictions_full[:, :, slice_idx] = pred_full
    
    return predictions_full

def save_prediction(prediction_volume, reference_nifti_path, output_path):
    """
    Save prediction as NIfTI file with same affine as reference.
    
    Args:
        prediction_volume: 3D prediction array
        reference_nifti_path: Path to reference NIfTI for affine matrix
        output_path: Where to save prediction
    """
    # Load reference to get affine transform
    reference = nib.load(str(reference_nifti_path))
    
    # Create new NIfTI with prediction data
    prediction_nifti = nib.Nifti1Image(
        prediction_volume.astype(np.uint8),
        reference.affine,
        reference.header
    )
    
    # Save
    nib.save(prediction_nifti, str(output_path))
    logging.info(f"Prediction saved to: {output_path}")

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("="*60)
    logging.info("A4-Unet Single Patient Inference")
    logging.info("="*60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create output directory
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logging.info("Loading A4-Unet model...")
    model = create_a4unet_model(
        image_size=INPUT_SIZE,
        num_channels=128,
        num_res_blocks=2,
        num_classes=2,
        learn_sigma=True,
        in_ch=4
    )
    
    # Load checkpoint
    logging.info(f"Loading checkpoint: {CHECKPOINT_PATH}")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        if 'mask_values' in checkpoint:
            del checkpoint['mask_values']
        model.load_state_dict(checkpoint)
        logging.info("Checkpoint loaded successfully!")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Get model info
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model parameters: {num_params:.1f}M")
    
    # Load patient data
    logging.info(f"\nLoading patient data from: {PATIENT_FOLDER}")
    if not PATIENT_FOLDER.exists():
        logging.error(f"Patient folder not found: {PATIENT_FOLDER}")
        return
    
    try:
        modalities = load_patient_volumes(PATIENT_FOLDER)
        logging.info(f"Loaded volume shape: {modalities['t1'].shape}")
    except Exception as e:
        logging.error(f"Failed to load patient data: {e}")
        return
    
    # Run inference
    logging.info("\nRunning inference on all slices...")
    try:
        predictions = run_inference(model, modalities, device, INPUT_SIZE)
        
        # Calculate statistics
        num_tumor_voxels = (predictions > 0).sum()
        total_voxels = predictions.size
        tumor_percentage = (num_tumor_voxels / total_voxels) * 100
        
        logging.info(f"\nPrediction Statistics:")
        logging.info(f"  Volume shape: {predictions.shape}")
        logging.info(f"  Tumor voxels: {num_tumor_voxels:,}")
        logging.info(f"  Total voxels: {total_voxels:,}")
        logging.info(f"  Tumor percentage: {tumor_percentage:.2f}%")
        
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save prediction
    patient_name = PATIENT_FOLDER.name
    output_path = OUTPUT_FOLDER / f'{patient_name}_prediction.nii.gz'
    
    # Use T1 as reference for affine matrix
    reference_path = None
    for ext in ['.nii', '.nii.gz']:
        ref = PATIENT_FOLDER / f'{patient_name}_t1{ext}'
        if ref.exists():
            reference_path = ref
            break
    
    if reference_path:
        save_prediction(predictions, reference_path, output_path)
    else:
        logging.warning("No reference T1 found, saving with default affine")
        pred_nifti = nib.Nifti1Image(predictions, np.eye(4))
        nib.save(pred_nifti, str(output_path))
        logging.info(f"Prediction saved to: {output_path}")
    
    logging.info("\n" + "="*60)
    logging.info("Inference completed successfully!")
    logging.info("="*60)
    
    # Load ground truth if available
    seg_path = None
    for ext in ['.nii', '.nii.gz']:
        seg = PATIENT_FOLDER / f'{patient_name}_seg{ext}'
        if seg.exists():
            seg_path = seg
            break
    
    if seg_path:
        logging.info("\nGround truth segmentation found!")
        logging.info("You can compare prediction with ground truth using visualization tools.")
        logging.info(f"  Ground truth: {seg_path}")
        logging.info(f"  Prediction:   {output_path}")

if __name__ == '__main__':
    main()
