# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:05:22 2019

@author: Chens
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba 
from wordcloud import WordCloud
from scipy.misc import imread
import os
from os import path
import string 
import seaborn as sns
import matplotlib as mpl
import pylab as pl
import numpy as np
from sklearn import (manifold, metrics, preprocessing)
from pyecharts import Map, Bar
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 

#%%
### 合并不同产品数据+数据清理

file_lst = []
for i in os.listdir():
    if ('.csv' in i):
        file_lst.append(i)
        
data={'num':[1], 'title':['1'], 'price':[1.00], 'quant':[1], 'place':['1']}
data=pd.DataFrame(data)

# 数据合并
def clean(file, n):
    
    df = pd.read_csv(open(file, encoding='utf-8'))
    df.columns=['num', 'title','price', 'quant','place']
    df.drop('num', axis=1)
    df['product'] = [int(n)]*len(df)
    return df

  
for file in file_lst:
    try:
        if "假发" in file:
            n=1
        else:
            n=2
            
        data=data.append(clean(file, n), sort=False)
    except UnicodeEncodeError:
        print('coding error!')

data = data.drop_duplicates() 
data=data.reset_index().drop(['num','index'], axis=1)
data = data.drop(0)
    
data['sale']= data.apply(lambda x: x['price'] * x['quant'], axis=1)
data['province']=data.place.apply(lambda x:x[0:2])

#data.to_csv('data.csv')



#%%
### title process
#标题分词

namelst = []
for item in data['title']:
    namelst.append(str(item))

stopwords = string.ascii_letters + ',:%'+ '\n' + '"' + '1234567890'

#指定产品
'''
namelst = []
for item in data[data['product']==1]:
    namelst.append(str(item))
'''

#drop symbol
with open('name.txt', 'w+', encoding='utf-8') as f:
    for i in namelst:
        n = jieba.lcut(i)
        for j in n:
            for k in stopwords:
                if j.find(k) == 0:
                    j=''
            f.write(j+'\n')
            
def wcloud(name):
    ### 词云绘制
    with open(name, encoding='utf-8')as file:
        #1.读取文本内容
        text=file.read()
    
        #2.设置词云的背景颜色、宽高、字数
        d = path.dirname('mask_image.jpg')
        mask_pic= imread(path.join(d, "mask_image.jpg"))   ### 读出词云图的底板图
    #    mask_pic=np.array(Image.open("mask_image.jpg"))
        wordcloud = WordCloud(font_path="C:/Windows/Fonts/simfang.ttf",
                              background_color='white',max_words=2000,max_font_size=50,
                             min_font_size=5, mask=mask_pic).generate(text)
        image=wordcloud.to_image()
        image.save('title.jpeg')
        image.show()


wcloud('name.txt')

#%%
###       descriptive statistics(假发)
    
    
# load data
    
#a=pd.read_csv(open('data.csv',encoding='utf-8'))
    
a = data[data['product']==1]   #假发产品
a = a.reset_index()
a.drop(['index'],axis=1)

#商品价格和销量分布

plt.figure(figsize=(15, 10))
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

price_pic = sns.violinplot(x="price",data=a)
price_pic.get_figure().savefig('price_box.jpeg')
quant_pic = sns.violinplot(x='quant',data=a)
quant_pic.get_figure().savefig('quant_box.jpeg')

#%%
#类型转换和价格极端值处理

a.price=a.price.astype('float')
a.quant=a.quant.astype('float')
de_extreme=a.query('price<873')

#%%
#商品价格和销量分布曲线图
#设定画布的大小
plt.figure(figsize=(15, 10))
sns.distplot(de_extreme.price, bins=10, hist=False, rug=True)
price_pic.get_figure().savefig('price_hist.jpeg')
sns.distplot(de_extreme.quant, bins=10, hist=False, rug=True)
quant_pic.get_figure().savefig('quant_hist.jpeg')

#%%
# 不同价格区间的商品的平均销量分布

#设定画布的大小
plt.figure(figsize=(15, 10))

listBins = [x for x in range(0,860,10)]
listLabels = ['{}-{}'.format(x,x+10) for x in range(0,850,10)]
de_extreme['price_2'] = pd.cut(de_extreme['price'], bins=listBins, labels=listLabels, include_lowest=True)
d=de_extreme.groupby('price_2').quant.mean().plot()
plt.xlabel('price section')
plt.ylabel('mean')
plt.savefig('section.jpeg')

#商品价格对销量的影响
sns.lmplot(x='price', y='quant', data=de_extreme)

#商品价格对销售额的影响
de_extreme['sale']= de_extreme.apply(lambda x: x['price'] * x['quant'], axis=1)
sns.lmplot(x='price', y='sale', data=de_extreme)

#不同省份商品数量分布
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

de_extreme.groupby('province').quant.mean().plot(kind='bar',figsize=(20,4))
pl.xticks(rotation=360)
de_extreme.groupby('province').sale.mean().plot(kind='bar',figsize=(20,4))
pl.xticks(rotation=360)

#%%
#图上显示分布

list1=[]
list2=[]

#城市名的list
for i,j in a.groupby('province').province:
    list1.append(i)
list1
k=a.groupby('province').sale.mean()

#城市对应销量

list_v = list(k.values)
list2 = []
for i in list_v:
    list2.append(round(i,2))
    
list1 = list(k.index)


m = Map("各地区销售额地图", width=1200, height=600)
m.add('销售额', list1, list2, visual_range=[min(list2), max(list2)],  maptype='china', 
      is_visualmap=True, visual_text_color='#000', is_label_show=True)
      
m.show_config()
m.render(path="中国地图.html")

#%%
##1.标题名和销售额分布
# 需重置索引
namelist=['刘海','长发','短发','补发','真发','长卷发','网红','卷发','大波浪','直发','梨花','丸子','微卷','黑发']

tsum=[]
for w in namelist:
    i=0
    s_list=[]
    for t in a.title:
        if w in t:
#            s_list.append(1)    #可显示出现频数
            s_list.append(a.sale[i])
        i+=1
    tsum.append(sum(s_list))
    
df_w_sum=pd.DataFrame({'tsum':tsum})
name=pd.DataFrame({'word':namelist})
df_sum=pd.concat([name,df_w_sum],axis=1,ignore_index=True)
df_sum.columns=['word','w_s_sum']
bar=Bar('标题与销售额')
bar.add('总',df_sum['word'],df_sum['w_s_sum'],is_stack=True,xaxis_rotate=30)
bar.render('标题和销售额.html')

#%%
##2.标题名和售价分布
m_price=[]
for w in namelist:
    i=0
    lst=[]
    for t in a.title:
        if w in t:
            lst.append(a.price[i])
        i+=1
    m_price.append(round(np.mean(lst),2))

df_m_price=pd.DataFrame({'mprice':m_price})
nlist=pd.DataFrame({'word':namelist})
df_sum=pd.concat([nlist,df_m_price],axis=1,ignore_index=True)
df_sum.columns=['word','price']
bar=Bar('标题与售价')
bar.add('',df_sum['word'],df_sum['price'],is_stack=True,xaxis_rotate=30)
bar.render('标题和售价.html')

#%%
#sex ratio

data['sex'] = data.apply(lambda x: 1 if '女' in x.title else 0, axis=1)

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('quant~sex',data).fit()
anovat = anova_lm(model)
print(round(anovat,3))

# area difference

model = ols('quant~province',data).fit()
anovat = anova_lm(model)
print(round(anovat,3))

#%%

###load data
# word frequency
# 筛选出现频率最高的假发类型分类词作为分类指标

data = a

namelst = []
for item in data['title']:
    namelst.append(str(item))

stopwords = string.ascii_letters + ',:%|'+ '\n' + '"' + '1234567890'


with open('title.txt', 'w+', encoding='utf-8') as f:
    for i in namelst:
        n = jieba.lcut(i)
        for j in n:
            for k in stopwords:
                if j.find(k) == 0:
                    j=''
            f.write(j+'\n')

with open('title.txt', 'r', encoding='utf-8') as file:
    title=file.readlines()


title_l = []
for i in title:
    title_l.append(i.strip('\n'))
    
t_dict={}
for item in title_l:
    if item not in t_dict:
        t_dict[item] = 1
    else:
        t_dict[item] += 1

count=0
t_dict_s = sorted(t_dict)

for item in t_dict_s:
    count += 1
    print(item)
    
    if count >= 30:
        break
'''       
lst = ["!", "#", "&", "'", "(", ")", "*", "+", "-", ".", "/", ";", "[", "]","\\", "~", "←", "│", "◊", "⭐", "、", "", "。", "《", "》", "「", "」", "『", "』", "【", "】", "の", "㊙", "一"]
for i in lst:
    t_dict.pop(i)
'''

#%%

# clustering analysis
# 仅对假发类进行分析
  
data['long'] = data.apply(lambda x: 1 if '长发' in x.title else 0, axis=1)
data['short'] = data.apply(lambda x: 1 if '短发' in x.title else 0, axis=1)
data['curve'] = data.apply(lambda x: 1 if '卷发' in x.title else 0, axis=1)
data['straight'] = data.apply(lambda x: 1 if '直发' in x.title else 0, axis=1)
data['wave'] = data.apply(lambda x: 1 if '波浪' in x.title else 0, axis=1)
data['black'] = data.apply(lambda x: 1 if '黑发' in x.title else 0, axis=1)
data = data.drop(['index'], axis=1)

data_cluster = data[['price', 'quant','long', 'short', 'curve', 'straight', 'wave', 'black']]

#data cleaning - standardized
data_cluster['price'] = preprocessing.scale(data_cluster['price'])
data_cluster['quant'] = preprocessing.scale(data_cluster['quant'])


## cluster elbow
K = range(2,9) # 假设可能聚成 2~8 类
lst = []
ss=[]
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_cluster)
    
    lst.append(sum(np.min(cdist(data_cluster, kmeans.cluster_centers_,'euclidean'), axis=1)) / data_cluster.shape[0])
    ss.append(metrics.silhouette_score(data_cluster, kmeans.labels_, metric = 'euclidean'))
    

plt.figure(figsize=(15, 10))
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

plt.plot(K, lst, 'bo-')
plt.title('elbow method')
plt.xlabel("K")
plt.ylabel("Cost function")
plt.show()       

### manifold

### cluster labeling & reducing data dimension
kmeans = KMeans(n_clusters=3)
kmeans.fit(data_cluster)
t = kmeans.labels_

tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
data_tsne = tsne.fit_transform(data_cluster)

plt.figure(figsize=(15, 10))
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

clster1 = pd.concat([pd.DataFrame(data_tsne), pd.Series(t)], axis=1)
clster1.columns = ['y', 'x', 't']
x1 = clster1[clster1['t']==0]
plt.scatter(x1.x, x1.y, color='red')
x2 = clster1[clster1['t']==1]
plt.scatter(x2.x, x2.y, color='blue')
x3 = clster1[clster1['t']==2]
plt.scatter(x3.x, x3.y, color='green')












