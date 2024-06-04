
import time
import random
from pathlib import Path

import bs4
import chardet
import requests
from simple_loggers import SimpleLogger

from . import util




class WebRequest(object):
    """Simple Web Requests Wrapper

    >>> from webrequests import WebRequest
    >>> 
    >>> # request an url
    >>> url = 'http://output.nsfc.gov.cn/captcha/defaultCaptcha'
    >>> resp = WebRequest.get_response(url)
    >>> resp = WebRequest.get_response(url, method='POST', max_try=5, timeout=5)
    >>> print(resp.headers)
    >>> 
    >>> # download file from an url
    >>> url = 'https://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz'
    >>> WebRequest.download(url)
    >>> WebRequest.download(url, 'out.jpg', max_try=5, timeout=10)
    >>> 
    >>> # request with session
    >>> session = requests.session()
    >>> resp = WebRequest.get_response('http://www.cip.cc/', session=session)
    >>> print(resp.cookies)
    >>> print(session.cookies)
    >>> 
    >>> # get a soup
    >>> soup = WebRequest.get_soup('http://www.cip.cc/')
    >>> print(soup.select_one('.kq-well pre').text.strip())
    """
    logger = SimpleLogger(name='WebRequest', level='info')

    MAX_CONTENT_SIZE_FOR_DETECTION  = 2 * 1024 * 1024

    @classmethod
    def random_ua(cls):
        ua_list = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95',
            'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)',
            'Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50',
            'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
        ]
        return {'User-Agent': random.choice(ua_list)}


    @classmethod
    def get_response(cls, url, method='GET', session=None, max_try=10, allowed_codes=[200], encoding=None, **kwargs):
        """
            params
                - allowed_codes: the allowed status_code

            Return a response object
        """
        if 'headers' not in kwargs:
            kwargs['headers'] = cls.random_ua()
        elif 'User-Agent' not in kwargs['headers']:
            kwargs['headers'].update(cls.random_ua())

        for n in range(max_try):
            try:
                r = session or requests
                resp = r.request(method, url, **kwargs)

                # only detect encoding when Content-Length < 2M
                content_length = int(resp.headers.get('Content-Length', 0))
                if not encoding and content_length <= cls.MAX_CONTENT_SIZE_FOR_DETECTION:
                    result = chardet.detect(resp.content)
                    if result['confidence'] > 0.9:
                        resp.encoding = result['encoding']
                elif encoding:
                    resp.encoding = encoding

                if resp.status_code in allowed_codes:
                    return resp
                cls.logger.warning('{}st time bad status code: {} [{}]'.format(n + 1, resp.status_code, resp.text))
            except Exception as e:
                cls.logger.debug('{}st time failed for url: {}, as {}'.format(n + 1, url, e))
            time.sleep(random.randint(2, 6))
        
        cls.logger.error('failed requests for url: {}'.format(url))
        return None
        
    @classmethod
    def download(cls, url, outfile=None, chunk_size=4096, show_progress=True, **kwargs):
        """
            Download file from an url
        """
        cls.logger.debug(f'download from url: {url}')

        resp = cls.get_response(url, stream=True, **kwargs)
        if not resp:
            return False

        if show_progress:
            total = int(resp.headers.get('Content-Length', 0))
            bar = util.file_download_bar(total=total, desc='downloading')

        outfile = outfile or Path(url).name
        with util.safe_open(outfile, 'wb') as out:
            for chunk in resp.iter_content(chunk_size):
                out.write(chunk)
                if show_progress:
                    bar.update(len(chunk))

        cls.logger.info('save file: {}'.format(outfile))

    @classmethod
    def get_soup(cls, url, features='html.parser', **kwargs):
        """
            features: html.parser, lxml

            Return: a BeautifulSoup object
        """
        resp = cls.get_response(url, **kwargs)
        if not resp:
            return False
        soup = bs4.BeautifulSoup(resp.text, features=features)
        return soup
