from bs4 import BeautifulSoup
from urllib.request import Request
from urllib.request import urlopen

time_info = []
link_list = []
raw_doc_list = []

for i in range(1, 2):  # (1,2) means to crawl the first page only
    site_url = f"https://www.semiconductors.org/news-events/latest-news/?fwp_paged={i}"
    
    # retrieve news page links
    with urlopen(Request(site_url, headers={'User-Agent': 'Chrome/119.0.0.0'})) as response:
        soup = BeautifulSoup(response, "html.parser")
        news_div_list = soup.find_all("div", {"class": "col-sm-8"})
        
        for news_div in news_div_list:
            time_info += [time_div.text for time_div 
                              in news_div.find_all("div", {"class": "resource-item-meta"})]
            link_list += [a_tag["href"] for a_tag in news_div.find_all("a", href=True)]
    
    # look into news pages
    for news_url in link_list:
        with urlopen(Request(news_url, headers={'User-Agent': 'Chrome/119.0.0.0'})) as response:
            soup = BeautifulSoup(response, "html.parser")
            main = soup.find("main")
            if not main: continue
            
            h1_tag = main.find("h1")
            raw_doc = h1_tag.text + ' '  # add the title to string
            
            for p in main.find_all("p"):  # add paragraphs to string
                raw_doc += p.text
            raw_doc_list.append(raw_doc)

# process time_info
for i in range(len(time_info)):
    time_info[i] = time_info[i][time_info[i].index(':') + 2 : ]
    time_seg = time_info[i].split('/')
    time_info[i] = f"20{time_seg[2]}/{time_seg[0]}/{time_seg[1]}"
print(time_info)
print(len(raw_doc_list), len(time_info))

# save the documents and dates
start_index = 1
for i in range(len(raw_doc_list)):
    with open(f"./semiconductor/content/{start_index + i}.txt", 'w', encoding='UTF-8') as content_file:
        content_file.write(raw_doc_list[i])
    with open(f"./semiconductor/date/{start_index + i}.txt", 'w', encoding='UTF-8') as date_file:
        date_file.write(time_info[i])