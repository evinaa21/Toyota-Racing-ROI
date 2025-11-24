import zipfile
import os

zip_path = r'c:\Users\User\Desktop\Toyota-Racing-ROI\data\Sonoma\Race 1\sonoma.zip'

if os.path.exists(zip_path):
    print(f"Found {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print("Contents of zip:")
            for name in zip_ref.namelist():
                print(f" - {name}")
    except Exception as e:
        print(f"Error reading zip: {e}")
else:
    print(f"File not found: {zip_path}")
