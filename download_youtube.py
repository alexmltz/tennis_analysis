#!/usr/bin/env python3
"""
YouTube Video Downloader Script

This script downloads YouTube videos in high quality MP4 format using yt-dlp.
It provides options for different quality settings and saves videos to the input_videos directory.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


def check_yt_dlp():
    """Check if yt-dlp is installed, if not, install it using uv."""
    try:
        subprocess.run(['yt-dlp', '--version'], check=True, capture_output=True)
        print("✓ yt-dlp is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("yt-dlp not found. Installing with uv...")
        try:
            subprocess.run(['uv', 'add', 'yt-dlp'], check=True)
            print("✓ yt-dlp installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install yt-dlp: {e}")
            return False


def download_youtube_video(url, output_dir="input_videos", quality="best"):
    """
    Download a YouTube video in high quality MP4 format.
    
    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to save the video
        quality (str): Quality setting ('best', 'worst', or specific format)
    
    Returns:
        bool: True if download successful, False otherwise
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # yt-dlp command with highest quality settings
    if quality == "best":
        # Use the best quality format available, prioritizing higher resolutions
        format_selector = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best'
    elif quality == "1080p":
        format_selector = 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best[height<=1080]'
    elif quality == "720p":
        format_selector = 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]'
    elif quality == "480p":
        format_selector = 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]'
    else:
        format_selector = f'{quality}[ext=mp4]/best[ext=mp4]/best'
    
    cmd = [
        'yt-dlp',
        '--format', format_selector,                           # High quality format selection
        '--output', str(output_path / '%(title)s.%(ext)s'),    # Save with video title as filename
        '--no-playlist',                                        # Download single video only
        '--merge-output-format', 'mp4',                        # Ensure final output is MP4
        '--embed-subs',                                         # Embed subtitles if available
        '--no-write-info-json',                                # Don't save video metadata
        '--no-write-thumbnail',                                # Don't save video thumbnail
        url
    ]
    
    try:
        print(f"Downloading video from: {url}")
        print(f"Output directory: {output_path.absolute()}")
        print(f"Quality setting: {quality}")
        print("-" * 50)
        
        # Run yt-dlp command
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        
        print("-" * 50)
        print("✓ Download completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def get_video_info(url):
    """Get information about the YouTube video without downloading."""
    cmd = [
        'yt-dlp',
        '--dump-json',
        '--no-playlist',
        url
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        import json
        info = json.loads(result.stdout)
        
        print(f"Title: {info.get('title', 'N/A')}")
        print(f"Duration: {info.get('duration_string', 'N/A')}")
        print(f"Uploader: {info.get('uploader', 'N/A')}")
        print(f"View count: {info.get('view_count', 'N/A')}")
        print(f"Upload date: {info.get('upload_date', 'N/A')}")
        
        return info
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to get video info: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None


def list_available_formats(url):
    """List all available formats for the video."""
    cmd = [
        'yt-dlp',
        '--list-formats',
        '--no-playlist',
        url
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to list formats: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download YouTube videos in high quality MP4 format')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('-o', '--output', default='input_videos', 
                       help='Output directory (default: input_videos)')
    parser.add_argument('-q', '--quality', default='best',
                       help='Quality setting: best, 1080p, 720p, 480p, worst, or specific format ID (default: best)')
    parser.add_argument('--info', action='store_true',
                       help='Show video information without downloading')
    parser.add_argument('--list-formats', action='store_true',
                       help='List available formats without downloading')
    
    args = parser.parse_args()
    
    # Check if yt-dlp is installed
    if not check_yt_dlp():
        sys.exit(1)
    
    # Show video info only
    if args.info:
        get_video_info(args.url)
        return
    
    # List available formats only
    if args.list_formats:
        list_available_formats(args.url)
        return
    
    # Download the video
    success = download_youtube_video(args.url, args.output, args.quality)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 