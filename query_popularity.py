
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('C:\\Users\\Vick\\Desktop\\results.csv')


# In[3]:


df.to_dict()


# In[4]:


new_dict = {}
old_dict = df.to_dict()
for key in old_dict['count']:
    new_dict[old_dict['query'][key]] = old_dict['count'][key]
new_dict


# In[5]:


def popularTerms(query):
    termsList = query.split()
    for term in termsList:
        if term in new_dict.keys():
            new_dict[term] += 1
        else:
            new_dict[term] = 1

    DescList = sorted(new_dict, key=new_dict.get, reverse=True)
    return DescList 


# In[8]:


updated_list = popularTerms('hello hello what is up')
updated_list


# In[9]:


for i in range (0,5):
    updated_dict[updated_list[i]] = new_dict[updated_list[i]]
updated_dict

