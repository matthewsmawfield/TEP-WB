#!/usr/bin/env python3
"""Unified PDF Processing Script
Compresses PDF and embeds comprehensive metadata in one operation.

This script processes the TEP-WB manuscript PDF (Paper 13: "The Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries") by compressing it 
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
        'Title': 'The Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries',
        'Author': 'Matthew Lukin Smawfield',
        'Creator': 'Matthew Lukin Smawfield',
        
        # Scientific abstract with key results
        'Subject': (
            'The Gaia DR3 catalog of over one million wide binaries opens a precise window onto gravity in the weak-field regime (a ≲ 10^-10 m/s^2), yet whether the observed velocity excess reflects modified gravity or unresolved systematics remains contested. This paper shows that the Temporal Equivalence Principle (TEP), in which a conformal scalar field modulates matter proper time as dτ/dt ≈ A(φ)^{1/2}, addresses that tension through density-dependent screening. From 341,315 high-purity systems, the analysis identifies a screening transition at R_s = 2,646 ± 182 AU (statistical; ±609 AU total), strongly preferred over both a flat Newtonian profile (Δχ² = 14,845) and a constant boost (Δχ² = 3,583). At large separation the profile saturates at α_sat = 0.366 ± 0.012, roughly 35-40% above the Keplerian baseline. The signal also shows the environmental ordering required by TEP. After metallicity-dependent mass corrections and bootstrap uncertainty estimation, the lower-density high-|Z| population transitions at smaller radius than the higher-density midplane (R_s = 4,662 ± 196 versus 7,131 ± 1,341 AU), confirmed by a solar-track control (R_s = 4,145 ± 276 versus 6,856 ± 920 AU; permutation p < 10^-4 for the full sample and p < 10^-3 for the solar track). Scrambling tests fail to reproduce the observed screening preference (p < 10^-4). The wide-binary anomaly is therefore not a generic low-acceleration excess but a structured, environmentally modulated screening transition whose morphology, onset scale, and density dependence are quantitatively consistent with the conformal scalar field of TEP and are not reproduced by MOND with or without the External Field Effect.',
        ),
        
        # Keywords for indexing
        'Keywords': (
            'Temporal Equivalence Principle; TEP; density-dependent screening; '
            'chameleon gravity; conformal scalar field; Gaia DR3 wide binaries; '
            'weak-field gravity tests; MOND; modified Newtonian dynamics; '
            'velocity excess anomaly; environmental screening; self-screening; '
            'mass-luminosity relation; metallicity correction; hierarchical triples; '
            'RUWE astrometry; screening transition radius; saturation amplitude; '
            'Galactic disk stratification; universal critical density; two-metric geometry; '
            'proper time modulation; external field effect; gravitational anomaly; '
            'Keplerian dynamics; astrometric precision; bootstrap uncertainty; '
            'permutation testing; SPARC rotation curves; cross-scale predictions; '
            'dark matter alternatives; fifth-force constraints; Ratra-Peeples potential; '
            'Galactic height dependence; vertical disk dynamics; injection-recovery validation; '
            'scalar-tensor gravity; modified gravity theories; dark matter; '
            'gravitational anomalies; stellar dynamics; galactic halo'
        ),
        
        # Production metadata
        'Producer': 'TEP-WB Research Project (Paper 13) - Version 0.3 (Kilifi)',
        
        # Rights and identifiers
        'Copyright': 'Creative Commons Attribution 4.0 International License (CC BY 4.0)',
        
        # Dates
        'CreationDate': '2026:04:27 00:00:00',
        'ModifyDate': '2026:04:27 00:00:00',
        
        # XMP Dublin Core metadata (exiftool uses these prefixes)
        'XMP-dc:Creator': 'Matthew Lukin Smawfield',
        'XMP-dc:Title': 'The Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries',
        'XMP-dc:Description': 'TEP resolution to Gaia DR3 wide-binary controversy via density-dependent temporal screening',
        'XMP-dc:Rights': 'CC BY 4.0',
        'XMP-dc:Identifier': f'doi:{args.doi}',
        'XMP-dc:Source': 'https://matthewsmawfield.github.io/TEP-WB/',
        'XMP-dc:Publisher': 'Zenodo',
        'XMP-dc:Date': '2026-04-27',
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
