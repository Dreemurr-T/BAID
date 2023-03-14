import requests
import re
import os
from time import time
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
import time
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
from tqdm import tqdm
import csv

# proxies = {
#     'http': 'http://127.0.0.1:7890',
#     'https': 'http://127.0.0.1:7890'
# }

# chrome_options = Options()
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--disable-gpu')
# chrome_options.add_argument('--proxy-server=http://127.0.0.1:7980')

csv_file = open('record.csv', 'a+', newline='')
writer = csv.writer(csv_file)

# s = Service(r"./chromedriver.exe")
headers = {
    'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.134 Safari/537.36 Edg/103.0.1264.71'}
# driver = webdriver.Chrome(service=s, options=chrome_options)
count = 24857

def get_pics(url):
    response = requests.get(url, headers=headers)
    # driver.get(url)
    html = response.text
    # html = driver.page_source
    urls = re.findall('<td><a href="(.*?)" alt=".*?" title=".*?">.*?</a></td>', html)
    # print(urls)
    for url in tqdm(urls):
        # driver.switch_to.default_content()
        url = "https:" + url
        # driver.implicitly_wait(10)
        # driver.get(url)
        # time.sleep(5)
        # driver.switch_to_default_content()
        # html1 = driver.page_source
        response = requests.get(url,headers=headers)
        html1 = response.text

        # frame = driver.find_elements(By.TAG_NAME, "iframe")[2]
        # driver.switch_to.frame(frame)
        # html2 = driver.page_source

        dir_name = 'images'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        download_pics(html1, dir_name)


def download_pics(html, dir_name):
    global count

    pic = re.findall('<a href="(.*?jpg)" title=".*?">.*?</a>', html)
    date = re.findall('<a href=".*?jpg" title="(.*?)">.*?</a>', html)
    month = re.findall(r"Entered in:&nbsp;<b>(.*)</b>",html)
    entrynum = re.findall(r"Entry Number:&nbsp;(.*)<br>", html)
    votes = re.findall('<span style=".*">(.*)</span>', html)
    name = re.findall('<h2>(.*?)</h2>', html)
    artist = re.findall('<i>.*<a href=".*">(.*?)</a></i>', html)
    material = re.findall(r"(.*)&nbsp;.*<br><br>", html)
    size = re.findall(r".*&nbsp;(.*)<br><br>", html)
    category = re.findall(r"Category:&nbsp;<b>(.*?)</b>", html)

    if size:
        size[0] = size[0].replace('-', '')
        size[0] = size[0].replace('/', '')

    # print(pic)
    # print(date)
    # print(entrynum)
    # print(votes)
    # print(name)
    # print(artist)
    # print(material)
    # print(size)
    # print(category)

    # likes = re.findall(r'<span>(.*) people like this.</span>', html1)
    # like_person = re.findall(r'<span>One person likes this.</span>', html1)
    if len(pic):
        try:
            img = requests.get(pic[0], headers=headers).content
        except requests.exceptions.SSLError as e:
            print(e)
        except requests.exceptions.ConnectionError as e:
            print(e)
        except requests.exceptions as e:
            print(e)
        except Exception as e:
            print(e)
        else:
            count += 1
            file_name = os.path.join(dir_name, entrynum[0]) + '.jpg'
            f = open(file_name, 'wb')
            f.write(img)
            f.close()
            writer.writerow([count, entrynum[0], month[0], date[0], int(votes[0]), name[0], artist[0], material[0], size[0],
                         category[0]])
    # if (len(likes)):
    #     writer.writerow([count, entrynum[0], date[0], int(votes[0]), int(likes[0])])
    # elif (len(like_person)):
    #     writer.writerow([count, entrynum[0], date[0], int(votes[0]), 1])
    # else:
    #     writer.writerow([count, entrynum[0], date[0], int(votes[0]), 0])


if __name__ == '__main__':
    # head = ['id', 'entry', 'month', 'date', 'vote', 'name', 'artist', 'material', 'size', 'category']
    # writer.writerow(head)
    i = 174
    while i >= 1:
        url = 'https://faso.com/boldbrush/popular/' + str(i)
        get_pics(url)
        i -= 1

    csv_file.close()
