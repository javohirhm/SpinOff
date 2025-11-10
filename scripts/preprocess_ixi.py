"""
Script to preprocess IXI MRI dataset
Converts NIfTI files to numpy arrays
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import preprocess_pipeline, create_splits
import argparse
from tqdm import tqdm
import json


def main():
    parser = argparse.ArgumentParser(description='Preprocess IXI MRI Dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory with IXI NIfTI files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed files')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256],
                       help='Target size for slices (H W)')
    parser.add_argument('--normalize', type=str, default='minmax',
                       choices=['minmax', 'zscore'],
                       help='Normalization method')
    parser.add_argument('--axis', type=int, default=2,
                       choices=[0, 1, 2],
                       help='Axis along which to slice (0=sagittal, 1=coronal, 2=axial)')
    parser.add_argument('--file_pattern', type=str, default='*.nii.gz',
                       help='File pattern to match')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_size = tuple(args.target_size)
    
    print("="*60)
    print("IXI MRI Dataset Preprocessing")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target size: {target_size}")
    print(f"Normalization: {args.normalize}")
    print(f"Slicing axis: {args.axis}")
    print("="*60)
    
    # Find all NIfTI files
    nifti_files = list(input_dir.rglob(args.file_pattern))
    
    if not nifti_files:
        print(f"❌ No files found matching pattern '{args.file_pattern}' in {input_dir}")
        print("Please check:")
        print("  1. The input directory path is correct")
        print("  2. NIfTI files are present (.nii or .nii.gz)")
        print("  3. The file pattern matches your files")
        return
    
    print(f"\nFound {len(nifti_files)} NIfTI files")
    
    # Process each file
    total_slices = 0
    stats = {
        'total_subjects': len(nifti_files),
        'total_slices': 0,
        'slices_per_subject': [],
        'failed_subjects': []
    }
    
    for nifti_path in tqdm(nifti_files, desc="Processing subjects"):
        try:
            # Extract subject ID from filename
            subject_id = nifti_path.stem.replace('.nii', '')
            
            # Process
            num_slices = preprocess_pipeline(
                str(nifti_path),
                str(output_dir),
                subject_id,
                normalize_method=args.normalize,
                target_size=target_size,
                axis=args.axis
            )
            
            total_slices += num_slices
            stats['slices_per_subject'].append(num_slices)
            
        except Exception as e:
            print(f"\n❌ Error processing {nifti_path.name}: {e}")
            stats['failed_subjects'].append(str(nifti_path))
    
    stats['total_slices'] = total_slices
    
    print("\n" + "="*60)
    print("Preprocessing Summary")
    print("="*60)
    print(f"Total subjects processed: {len(nifti_files) - len(stats['failed_subjects'])}")
    print(f"Total slices created: {total_slices}")
    print(f"Average slices per subject: {total_slices / (len(nifti_files) - len(stats['failed_subjects'])):.1f}")
    
    if stats['failed_subjects']:
        print(f"\n⚠️  Failed subjects: {len(stats['failed_subjects'])}")
        for failed in stats['failed_subjects'][:5]:  # Show first 5
            print(f"  - {failed}")
    
    # Save statistics
    with open(output_dir / 'preprocessing_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print(f"Statistics saved to: {output_dir / 'preprocessing_stats.json'}")
    
    # Create splits
    print("\n" + "="*60)
    print("Creating train/val/test splits...")
    print("="*60)
    
    splits = create_splits(str(output_dir), train_ratio=0.7, val_ratio=0.15)
    
    # Save splits to JSON
    splits_dict = {
        'train': [str(p) for p in splits['train']],
        'val': [str(p) for p in splits['val']],
        'test': [str(p) for p in splits['test']]
    }
    
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits_dict, f, indent=2)
    
    print(f"Splits saved to: {output_dir / 'splits.json'}")
    print("\n✅ All done!")


if __name__ == "__main__":
    main()
