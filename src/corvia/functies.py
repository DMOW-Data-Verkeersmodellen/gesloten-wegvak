
from klasseDefinities import *


def findStartLinks(linkList):
    startLinks = []

    for link in linkList:
        if len(link.prevLinkList) != 1:
            startLinks.append(link)
        
        if len(link.prevLinkList) == 1:
            prevLink = link.prevLinkList[0]
            if len(prevLink.nextLinkList) > 1:
                startLinks.append(link)

    return startLinks


def createLinkKetens(startLinks):
    linkketenList = []

    ## If there are no start links, return an empty list    
    ## Loop through the start links and create chains
    for startLink in startLinks:
        currentLink = startLink
        keten = [currentLink]

        ## Loop through the next links until there are no more next links or more than one next link
        while currentLink.nextLinkList and len(currentLink.nextLinkList) == 1:

            currentLink = currentLink.nextLinkList[0]

            if len(currentLink.prevLinkList) != 1:
                break
            
            keten.append(currentLink)
            
            # # Delete the current link from the list of start links to avoid duplicates
            # if currentLink in startLinks:
            #     startLinks.remove(currentLink)

        linkketenList.append(Linkketen(keten))
    return linkketenList
