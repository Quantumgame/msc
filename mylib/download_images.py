# -*- coding=UTF-8 -*-

"""

File name : download_images.py

Creation date : 20-07-2016

Last modified :

Created by :

Purpose :

    Downloads image from google image search.

Usage :

    python download_images.py keyword directory quantity

Observations :

    A new folder named keyword will be created if there is none under directory.

"""

from icrawler.builtin import GoogleImageCrawler
import os
import sys

def download_images(keyword, directory, quantity):

    os.chdir(directory)

    if keyword not in os.listdir():

        os.mkdir(keyword)

    os.chdir(keyword)

    google_crawler = GoogleImageCrawler(directory + '/' + keyword)
    google_crawler.crawl(keyword=keyword, offset=0, max_num=quantity,
                         date_min=None, date_max=None, feeder_thr_num=1,
                         parser_thr_num=1, downloader_thr_num=4,
                         #min_size=(200,200), max_size=None)
                         min_size=None, max_size=None)

    # adding keyword in the beginning of filename
    command = 'perl-rename \'s/(.*)/%s_$1/\' *' %(keyword)
    os.system(command)


if __name__ == '__main__':

    keyword = sys.argv[1]
    directory = sys.argv[2]
    quantity = int(sys.argv[3])

    download_images(keyword, directory, quantity)
