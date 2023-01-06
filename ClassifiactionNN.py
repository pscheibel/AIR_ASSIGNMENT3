def get_pdfs(my_url):
    print("process: "+ str(my_url))
    links = []
    html  = urllib.request.urlopen(my_url).read()
    soup = BeautifulSoup(html)
    for link in soup.findAll('a'):
        if "/pdf/" in str(link.get('href')):
            links.append("https://arxiv.org"+str(link.get('href'))+".pdf")

    #for link in links:
    #        try:
    #            #print(link)
    #            ign=0
    #            #wget.download(link)
    #            #input  = urllib.request.urlopen(link).read()
    #            #input = PdfFileReader(file(link, "rb"))
    #            #print(input)
    #        except:
    #            print(" \n \n Unable to Download A File \n")
    return links
