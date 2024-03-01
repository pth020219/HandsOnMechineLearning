#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 스팸 분류기를 만들어 보세요.


# In[25]:


import os
import tarfile
import urllib
import requests
from bs4 import BeautifulSoup
import re
import shutil


# In[46]:


# 데이터셋 다운로드

download_root = "https://spamassassin.apache.org/old/publiccorpus/"  # 파일이 위치해있는 URL

response = requests.get(download_root)    # requests 를 통한 URL 에 있는 'a' tag 추출
soup = BeautifulSoup(response.text, 'html.parser') 
tags = soup.select('a')

for i, tag in enumerate(tags):    # 'a' 태그 텍스트 추출
    tags[i] = tag.get_text()
    
# 정규표현식을 활용한 파일 이름 추출

regex = re.compile(r"^[0-9]{8}.*")
file_names = []

for i in range(len(tags)):
    result = regex.search(tags[i])
    if result:
        file_names.append(result.group())
                
def fetch_mail_data(file_names=file_names, download_root=download_root):
    os.makedirs("datasets/hams", exist_ok=True)
    os.makedirs('datasets/spams', exist_ok=True)
    
    for name in file_names:
        file_url = download_root + name
        bz2_path = os.path.join('datasets', name)
        urllib.request.urlretrieve(file_url, bz2_path)
        
        if 'ham' in name:    #bz2 파일이 ham 인지 spam 인지 구분 후, 추후 들어갈 디렉토리 정의
            ham_or_spam = 'hams'
        else:
            ham_or_spam = 'spams'
            
        with tarfile.open(bz2_path, 'r') as tar:
            tar.extractall(path='datasets')    # datasets 디렉토리에 바로 압축 해제
            middle = re.search('_(\w+).tar.bz2', name).group(1)    
            # 압축 해제시, 20021010_hard_ham.tar.bz 를 예시로 들면, hard_ham 디렉토리가 생성되고, 그 안에 데이터가 존재함.
            # 데이터를 꺼내 hams or spams 디렉토리에 넣기 위해, hard_ham 같은 middle 문자열을 추출
            files = os.listdir(os.path.join('datasets', middle))
            
            for file in files:
                shutil.move(os.path.join('datasets', middle, file), os.path.join('datasets', ham_or_spam, file))
            shutil.rmtree(os.path.join('datasets', middle))


# In[47]:


fetch_mail_data(file_names)


# In[45]:


# shutil.rmtree('datasets')

