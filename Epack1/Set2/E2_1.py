
# Get the text content of the page
def getpagetext(parsedpage):
    # Remove HTML elements that are scripts
    scriptelements=parsedpage.find_all('script')
    # Concatenate the text content from all table cells
    for scriptelement in scriptelements:
        # Extract this script element from the page.
        # This changes the page given to this function!
        scriptelement.extract()
        
    pagetext=parsedpage.get_text()
    return(pagetext)

#%%
import requests

mywebpage_url = 'https://www.sis.uta.fi/~tojape/'
#mywebpage_url='https://www.tuni.fi/en/'
mywebpage_html = requests.get(mywebpage_url)

#%%
import bs4
# Parse the HTML content using beautifulsoup
mywebpage_parsed = bs4.BeautifulSoup(mywebpage_html.content,'html.parser')

mywebpage_text=getpagetext(mywebpage_parsed)
print(mywebpage_text)

#%%
"""
If necessary, beautifulsoup allows to search for
individual cells. Be careful to avoid duplicating text:
contents of nested cells are also listed in their
parents!
Example:
"""
# Find HTML elements that are table cells or 'div' cells
parsedpage = mywebpage_parsed
tablecells=parsedpage.find_all(['td','div'])
# Concatenate the text content from all table or div cells
pagetext=''
for tablecell in tablecells:
    pagetext=pagetext+'\n'+tablecell.text.strip()

#%%
""" 
To crawl further pages, we analyze links on the page we already crawled:
Example:
"""
# Find linked pages in Finnish sites, but not PDF or PS files
def getpageurls(webpage_parsed):
    # Find elements that are hyperlinks
    pagelinkelements = webpage_parsed.find_all('a')
    pageurls = []
    for pagelink in pagelinkelements:
        pageurl_isok=1
        try:
            pageurl=pagelink['href']
        except:
            pageurl_isok=0
        if(pageurl_isok):
            # Check that the url does NOT contain these strings
            if (pageurl.find('.pdf')!=-1) | (pageurl.find('.ps')!=-1):
                pageurl_isok=0
            # Check that the url DOES contain these strings
            if (pageurl.find('http')==-1)|(pageurl.find('.fi')==-1):
                pageurl_isok=0

        if(pageurl_isok):
            pageurls.append(pageurl)

    return(pageurls)

#%%
mywebpage_urls=getpageurls(mywebpage_parsed)
print(mywebpage_urls)

#%%
"""
Basic crawling procedure: start from a seed page, crawl until there are enough
"""
# Basic web crawler
def basicwebcrawler(seedpage_url, maxpages):
    # Store URLs crawled and their text content
    crawled_urls = []
    crawled_texts = []
    num_pages_crawled = 0

    # Remaining pages to crawl: start from a seed page URL
    pagestocrawl = [seedpage_url]
    # Process remaining pages until a desired number of pages have been found
    while (num_pages_crawled<maxpages) & (len(pagestocrawl)>0):
        # Retrieve the topmost remaining page and parse it
        pagetocrawl_url = pagestocrawl[0]
        print('Getting page:\n', pagetocrawl_url)
        pagetocrawl_html = requests.get(pagetocrawl_url)
        pagetocrawl_parsed = bs4.BeautifulSoup(pagetocrawl_html.content,'html.parser')
        
        # Get the text and URLs of the page
        pagetocrawl_text = getpagetext(pagetocrawl_parsed)
        pagetocrawl_urls = getpageurls(pagetocrawl_parsed)

        # Store the URL and content of the processed page
        num_pages_crawled += 1
        crawled_urls.append(pagetocrawl_url)
        crawled_texts.append(pagetocrawl_text)
        
        # Remove the processed page from remaining pages, but add the new URLs
        pagestocrawl=pagestocrawl[1:len(pagestocrawl)]
        pagestocrawl.extend(pagetocrawl_urls)

    return(crawled_urls,crawled_texts)

#%%
mycrawled_urls_and_texts = basicwebcrawler(mywebpage_url, 3)
mycrawled_urls = mycrawled_urls_and_texts[0]
mycrawled_texts = mycrawled_urls_and_texts[1]

#%%
import random
"""
Basic crawling procedure: start from a seed page, crawl until there are enough

Updated version asked in Exercise 2.1.
"""
# Basic web crawler
def basicwebcrawler_updated(seedpage_url, maxpages):
    # Store URLs crawled and their text content
    crawled_urls = []
    crawled_texts = []
    num_pages_crawled = 0

    # Remaining pages to crawl: start from a seed page URL
    pagestocrawl = [seedpage_url]
    # Process remaining pages until a desired number of pages have been found
    while (num_pages_crawled<maxpages) & (len(pagestocrawl)>0):
        print()
        # Retrieve random from remaining pages with some probability ...
        # Exercise 2.1, Solution to task 2.
        pagetocrawl_url = pagestocrawl[0]
        if random.random() < 0.7:
            pagetocrawl_url = random.choice(pagestocrawl)
        
        # ... and parse it.
        print('Length:', len(pagestocrawl))
        print('Getting page:\n', pagetocrawl_url)
        pagetocrawl_html = requests.get(pagetocrawl_url)
        pagetocrawl_parsed = bs4.BeautifulSoup(pagetocrawl_html.content,'html.parser')
        print('Page Getted.')
        
        # Get the text and URLs of the page
        pagetocrawl_text = getpagetext(pagetocrawl_parsed)
        pagetocrawl_urls = getpageurls(pagetocrawl_parsed)

        # Store the URL and content of the processed page
        num_pages_crawled += 1
        crawled_urls.append(pagetocrawl_url)
        crawled_texts.append(pagetocrawl_text)
        
        # Remove the processed page from remaining pages, but add the new URLs
        # Exercise 2.1. Solution to task 1.
        pagestocrawl.remove(pagetocrawl_url)
        pagetocrawl_urls = list(set(pagetocrawl_urls) - set(crawled_urls))
        pagestocrawl.extend(pagetocrawl_urls)
        print('num_pages_crawled=',num_pages_crawled)
        
        
    return(crawled_urls,crawled_texts)

#%% Run by using updated function
mycrawled_urls_and_texts2 = basicwebcrawler_updated(mywebpage_url, 3)
mycrawled_urls2 = mycrawled_urls_and_texts2[0]
mycrawled_texts2 = mycrawled_urls_and_texts2[1]













