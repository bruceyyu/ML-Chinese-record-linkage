#!/usr/bin/env python3
"""
Simple Dataverse File Downloader and Converter

Downloads a specific .dta file from Dataverse.
"""

import requests
import os
import sys
import pandas as pd
from pathlib import Path

SERVER_URL = "https://dataverse.harvard.edu"  # Change this if using a different server
FILE_ID = 6177054
FILE_NAME = "CGED-Q 1850-1864.dta"

def download_dataverse_file(server_url, file_id):
    """
    Download a file from Dataverse using file ID.
    
    Args:
        server_url (str): Base URL of the Dataverse server
        file_id (str/int): The numeric ID of the file
    
    Returns:
        str: Path to the downloaded file
    """
    # Construct the URL
    url = f"{server_url}/api/access/datafile/{file_id}"
    
    # Parameters to get the original .dta file
    params = {'format': 'original'}
    
    try:
        print(f"Downloading from: {url}")
        print(f"Parameters: {params}")
        
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()
        
        # Get filename from Content-Disposition header
        filename = FILE_NAME
        
        print(f"Saving as: {filename}")
        
        # Download the file
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(filename)
        print(f"Download completed! File size: {file_size:,} bytes")
        return filename
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("Error: File not found. Please check the file ID.")
        else:
            print(f"HTTP Error {e.response.status_code}: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def main():
    # Configuration
    
    print("=== Dataverse File Downloader and Converter ===")
    print(f"Server: {SERVER_URL}")
    print(f"File ID: {FILE_ID}")
    print()
    
    try:
        # Download the .dta file
        dta_file = download_dataverse_file(SERVER_URL, FILE_ID)
        
        print(f"\n✅ Success!")
        print(f"📁 Downloaded: {dta_file}")
            
    except KeyboardInterrupt:
        print("\n❌ Process cancelled by user")
        sys.exit(1)


if __name__ == '__main__':
    main()
