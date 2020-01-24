#一个想法：
#可以按照发型的类型来进行区分获取数据

import time
from selenium import webdriver
import re
import os
import pandas as pd
import random
import requests


def search_product(name):
#使用python打开特定网页
    
    input_div=driver.find_element_by_id("q")
    input_div.send_keys(name)
    driver.find_element_by_xpath('//*[@id="J_TSearchForm"]/div[1]/button').click()
#手动登录延迟
    time.sleep(20)
#找到一共有多少页
    token=driver.find_element_by_xpath('//*[@id="mainsrp-pager"]/div/div/div/div[1]').text
    token=int(re.compile('(\d+)').search(token).group(1))
    return token

#拉动滑条
def drop_down():
    for x in range(1,11,2):
        time.sleep(1.5)
        j=x/10
        js='document.documentElement.scrollTop=document.documentElement.scrollHeight*%f' % j
        driver.execute_script(js)

#正则表达式定义
def get_products():

    html = driver.page_source
    use = re.compile('[\s]|(\\n)*')
    t = use.sub('', html)
    
    pattern = re.compile('alt="(.*?)".*?<\/span><strong>(\d*.\d*).*?"deal-cnt">(\d*).*?"location">(.*?)<\/div>')
    info = re.findall(pattern, t)
    return info

    
    
# 下一页
def next_page(name):
    
    
    token=search_product(name)
        
    pages=sorted(random.sample(range(0, token+1), 20))
    
    info_lst = []
    for num in pages:  ### beta,(token)
        try:
            url=r'https://s.taobao.com/search?q='+name+'&s={}'. format(44* num)
            driver.implicitly_wait(10)
            driver.get(url)
            drop_down()
            info_lst += get_products()
            num+=1
            print('success')
            
        except:
            
            print("connection lost")
            
#            driver.quit()
            break
            
    df = pd.DataFrame(info_lst)
    df.to_csv(name+'3.csv', encoding='utf-8')   ### 分多次爬取，改变存储文件名
#    df_temp.to_csv(name+' temp'+'.csv')
           
        
def internet_on():   ### check connection status
    url = 'http://www.baidu.com'
    try:
        html = requests.get(url, timeout=2)
        print('Internet Connected')
    except:
        print('Please check your connection')
        
if __name__=='__main__':
    
    internet_on()
    
    driver = webdriver.Chrome(os.getcwd()+r'\chromedriver')
    driver.maximize_window()
    driver.get(r'https://www.taobao.com/')
      
    next_page('假发')   ### change parameters to get different products
    