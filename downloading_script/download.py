import pandas as pd
import os
from tqdm import tqdm
from urllib.request import urlretrieve
import urllib

opener = urllib.request.build_opener()
header = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.0.0'
opener.addheaders = [('User-Agent', header)]
urllib.request.install_opener(opener)


def get_pics(urls, names):
    output_dir = 'images'
    os.makedirs(output_dir, exist_ok=True)
    for idx, url in enumerate(tqdm(urls)):
        output_path = os.path.join(output_dir, names[idx])
        try:
            urlretrieve(url, output_path)
        except Exception:
            print(url)
            continue


if __name__ == '__main__':
    df = pd.read_csv('downloading_script/image_list.csv')
    urls = df['image_link'].values.tolist()
    names = df['image_name'].values.tolist()

    get_pics(urls, names)

