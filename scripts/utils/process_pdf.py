#!/usr/bin/env python3
"""Unified PDF Processing Script
Compresses PDF and embeds comprehensive metadata in one operation.

This script processes the TEP-WB manuscript PDF (Paper 15: "The Wide Binary Anomaly: 
Resolving the Gaia DR3 Controversy via Temporal Screening") by compressing it 
for web distribution and embedding complete academic metadata for proper indexing and citation.

Usage:
    python process_pdf.py <input_pdf> [--quality ebook|printer|prepress|default]
    
Example:
    python process_pdf.py site/public/docs/Smawfield_2026_TEP-WB_v0.1_Kilifi.pdf --quality ebook
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import tempfile


def compress_pdf(input_path, output_path, quality='ebook'):
    """Compress PDF using Ghostscript."""
    quality_settings = {
        'screen': '/screen',      # 72 dpi
        'ebook': '/ebook',        # 150 dpi
        'printer': '/printer',    # 300 dpi
        'prepress': '/prepress',  # 300 dpi, color preserving
        'default': '/default'
    }
    
    if quality not in quality_settings:
        raise ValueError(f"Quality must be one of: {', '.join(quality_settings.keys())}")
    
    gs_quality = quality_settings[quality]
    
    # Get original size
    original_size = os.path.getsize(input_path)
    
    # Compress using Ghostscript
    cmd = [
        'gs',
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        f'-dPDFSETTINGS={gs_quality}',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        f'-sOutputFile={output_path}',
        input_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        compressed_size = os.path.getsize(output_path)
        reduction = ((original_size - compressed_size) / original_size) * 100
        
        return {
            'original_mb': original_size / (1024 * 1024),
            'compressed_mb': compressed_size / (1024 * 1024),
            'reduction_pct': reduction
        }
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ghostscript compression failed: {e.stderr.decode()}")


def embed_metadata(pdf_path, metadata):
    """Embed metadata into PDF using exiftool."""
    cmd = ['exiftool']
    
    # Add all metadata fields
    for key, value in metadata.items():
        cmd.extend([f'-{key}={value}'])
    
    # Overwrite original
    cmd.extend(['-overwrite_original', pdf_path])
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Exiftool metadata embedding failed: {e.stderr.decode()}")


def verify_metadata(pdf_path, expected_fields):
    """Verify metadata was embedded correctly."""
    cmd = ['exiftool'] + [f'-{field}' for field in expected_fields] + [pdf_path]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Compress PDF and embed metadata in one operation'
    )
    parser.add_argument('input_pdf', help='Path to input PDF file')
    parser.add_argument(
        '--quality',
        choices=['screen', 'ebook', 'printer', 'prepress', 'default'],
        default='ebook',
        help='Compression quality (default: ebook)'
    )
    parser.add_argument(
        '--doi',
        default='10.5281/zenodo.19102062',
        help='DOI to embed in metadata'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_pdf).resolve()
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    print(f"Processing: {input_path}")
    print(f"Quality: {args.quality}")
    print()
    
    # Step 1: Compress PDF
    print("Step 1: Compressing PDF...")
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        stats = compress_pdf(str(input_path), tmp_path, args.quality)
        
        # Replace original with compressed version
        os.replace(tmp_path, str(input_path))
        
        print(f"  Original:    {stats['original_mb']:.2f} MB")
        print(f"  Compressed:  {stats['compressed_mb']:.2f} MB")
        print(f"  Reduction:   {stats['reduction_pct']:.1f}%")
        print()
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        print(f"Error during compression: {e}")
        sys.exit(1)
    
    # Step 2: Embed metadata
    print("Step 2: Embedding metadata...")
    
    # Paper metadata - must match manuscript, CITATION.cff, and zenodo.txt
    metadata = {
        # Core identification
        'Title': 'The Wide Binary Anomaly: Resolving the Gaia DR3 Controversy via Temporal Screening',
        'Author': 'Matthew Lukin Smawfield',
        'Creator': 'Matthew Lukin Smawfield',
        
        # Scientific abstract with key results
        'Subject': (
            'The 2022 release of Gaia Data Release 3 (DR3) provided a catalog of more than a million '
            'wide binary stars, creating a powerful test of gravity in the extreme weak-field regime. '
            'Yet the physical interpretation of the signal remains unsettled. Chae (2023, 2024) and '
            'Hernandez (2023) report a significant anomalous velocity boost of ~20% at wide separations, '
            'interpreting it as evidence for Modified Newtonian Dynamics (MOND). By contrast, Banik et al. '
            '(2024) and Pittordis & Sutherland (2025) argue that the same signal may still be explained '
            'within standard General Relativity through hierarchical triple systems. This paper shows that '
            'the Temporal Equivalence Principle (TEP) offers a natural resolution. In TEP, a conformal '
            'scalar field couples to local mass density, so the anomaly appears through density-dependent '
            'screening rather than as a universal, scale-free boost. The Gaia DR3 sample reveals a '
            'characteristic transition radius of Rs = 2646 ± 182 AU, with the lower-density halo '
            'transitioning at smaller radius than the higher-density disk (p = 0.0033). The screening '
            'profile is strongly preferred over both Newtonian and constant-boost alternatives.'
        ),
        
        # Keywords for indexing
        'Keywords': (
            'Wide Binary Stars; Gaia DR3; Modified Newtonian Dynamics; MOND; '
            'Temporal Equivalence Principle; TEP; Chameleon Screening; '
            'Scalar-Tensor Gravity; Modified Gravity; Gravitational Anomalies; '
            'Weak-Field Gravity; Dark Matter; Hierarchical Triples; Stellar Dynamics; '
            'Galactic Halo; Galactic Disk; Density-Dependent Screening'
        ),
        
        # Production metadata
        'Producer': 'TEP-WB Research Project (Paper 15) - Version 0.1 (Kilifi)',
        
        # Rights and identifiers
        'Copyright': 'Creative Commons Attribution 4.0 International License (CC BY 4.0)',
        
        # Dates
        'CreationDate': '2026:03:19 00:00:00',
        'ModifyDate': '2026:03:19 00:00:00',
        
        # XMP Dublin Core metadata (exiftool uses these prefixes)
        'XMP-dc:Creator': 'Matthew Lukin Smawfield',
        'XMP-dc:Title': 'The Wide Binary Anomaly: Resolving the Gaia DR3 Controversy via Temporal Screening',
        'XMP-dc:Description': 'TEP resolution to Gaia DR3 wide-binary controversy via density-dependent temporal screening',
        'XMP-dc:Rights': 'CC BY 4.0',
        'XMP-dc:Identifier': f'doi:{args.doi}',
        'XMP-dc:Source': 'https://matthewsmawfield.github.io/TEP-WB/',
        'XMP-dc:Publisher': 'Zenodo',
        'XMP-dc:Date': '2026-03-19',
        'XMP-dc:Type': 'Preprint',
        'XMP-dc:Format': 'application/pdf',
        'XMP-dc:Language': 'en',
        
        # PRISM (Publishing Requirements for Industry Standard Metadata)
        'XMP-prism:DOI': args.doi,
        'XMP-prism:URL': 'https://matthewsmawfield.github.io/TEP-WB/',
        'XMP-prism:VersionIdentifier': '0.1',
        'XMP-prism:PublicationName': 'TEP Research Series',
        
        # PDF/A metadata
        'XMP-pdfaid:Part': '1',
        'XMP-pdfaid:Conformance': 'B'
    }
    
    try:
        embed_metadata(str(input_path), metadata)
        print("  Metadata embedded successfully")
        print()
        
    except Exception as e:
        print(f"Error during metadata embedding: {e}")
        sys.exit(1)
    
    # Step 3: Verify
    print("Step 3: Verifying metadata...")
    verification = verify_metadata(
        str(input_path),
        ['Title', 'Author', 'Subject', 'Keywords', 'Creator', 'Copyright']
    )
    
    if verification:
        print("  ✓ Metadata verified")
        print()
        print("Verification output:")
        print(verification)
    else:
        print("  ⚠ Could not verify metadata")
    
    print()
    print(f"✓ Processing complete: {input_path}")
    print(f"  Final size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")


if __name__ == '__main__':
    main()
