from bs4 import BeautifulSoup
import requests

# Scrape function
def scrape(site):
    urls = []

    def scrape_helper(current_site):
        nonlocal urls
        r = requests.get(current_site)
        s = BeautifulSoup(r.text, "html.parser")
        print(s.find_all("a"))
        for i in s.find_all("a"):
            if "href" in i.attrs:
                href = i.attrs["href"]

                if href.startswitch("/") or href.startswitch("#"):
                    full_url = site + href 

                    if full_url not in urls:
                        urls.append(full_url)
                        scrape_helper(full_url)

    scrape_helper(site)
    return urls
