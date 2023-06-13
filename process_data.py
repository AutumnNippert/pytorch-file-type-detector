import os
import base64
from PIL import Image
import pandas as pd

file_types = [
    '.py', '.c', '.cpp', '.h', '.java', '.js', '.html', '.css', '.xml', '.json',
    '.txt', '.csv', '.md', '.pdf', '.doc', '.xls', '.ppt', '.zip', '.rar', '.tar',
    '.gz', '.7z', '.bmp', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.mp3', '.wav',
    '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.aac', '.psd', '.ai', '.eps',
    '.svg', '.ogg', '.ico', '.exe', '.dll', '.obj', '.class', '.jar', '.apk',
    '.pdb', '.ini', '.cfg', '.bat', '.sh', '.ps1', '.sql', '.bak', '.log', '.bak',
    '.docx', '.xlsx', '.pptx', '.odt', '.ods', '.odp', '.dwg', '.dxf', '.max',
    '.tif', '.tiff', '.mov', '.m4a', '.srt', '.sub', '.ass', '.ttf', '.woff',
    '.woff2', '.eot', '.otf', '.swf', '.ttf', '.lnk', '.url', '.dat', '.db',
    '.sqlite', '.sqlite3', '.dbf', '.ps', '.rtf', '.tex', '.chm', '.csv', '.tsv',
    '.yaml', '.yml', '.bak', '.tmp', '.swp', '.ics', '.iso', '.img', '.rpm',
    '.deb', '.pkg'
]


def file_to_image(file_path, image_path, x_dim, y_dim):
    with open(file_path, 'rb') as file:
        file_data = file.read()
        encoded_data = base64.b64encode(file_data)

        pixel_count = x_dim * y_dim

        # Pad
        while len(encoded_data) < pixel_count:
            encoded_data += chr(0).encode('utf-8')

        # Create a new image
        image = Image.new('L', (x_dim, y_dim))

        # Iterate over the encoded data and set pixel values
        for i in range(pixel_count):
            pixel_value = encoded_data[i]
            x = i % x_dim
            y = i // x_dim
            image.putpixel((x, y), pixel_value)

        image.save(image_path)

def dirToFiles(input_dir_path, output_dir_path, x_dim, y_dim):
    # create output directory
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # get total size
    total_kb = 0
    for file in os.listdir(input_dir_path):
        file_path = os.path.join(input_dir_path, file)
        total_kb += os.path.getsize(file_path) / 1000.0

    running_total_kb = 0

    files = []
    labels = []
    for file in os.listdir(input_dir_path):
        file_path = os.path.join(input_dir_path, file)
        # get extension
        extension = os.path.splitext(file_path)[-1]

        image_name = file + '.png'
        image_path = os.path.join(output_dir_path, image_name)

        files.append(image_name)
        labels.append(extension)
        file_to_image(file_path, image_path, x_dim, y_dim)
        # flush output
        progress = (len(files) / len(os.listdir(input_dir_path))) * 100
        progress_str = '=' * int(progress / 2)
        # pad with spaces
        progress_str = progress_str.ljust(50)

        curr_kb = os.path.getsize(image_path) / 1000.0
        running_total_kb += curr_kb

        print(f'Processing {output_dir_path} [{progress_str}] {progress:.2f}% [{running_total_kb:.3f}kB / {total_kb:.3f}kB]', end='\r', flush=True)
    print()
    print(f'Processed {len(files)} files')
    
    # create csv file
    df = pd.DataFrame({'file': files, 'label': labels})
    df.to_csv(output_dir_path + '_dataset.csv', index=False)

dirToFiles('train_data_unp', 'train_data', 256, 256)
dirToFiles('test_data_unp', 'test_data', 256, 256)