
## Definieer klasse 'Link'
class Link:
    def __init__(self, startNode, endNode, linkNo):
        self.startNode = startNode
        self.endNode = endNode
        self.linkNo = linkNo
        self.tellingenList = []
        self.intensiteiten = {"PW": None, "VR": None, "Totaal": None}
        self.nextLinkList = []
        self.prevLinkList = []

    ## Definieer een methode om de link te printen
    def __repr__(self):
        return f"""Deze link met id '{self.linkNo}' heeft start node '{self.startNode}' en eind node '{self.endNode}'.
Deze link heeft {len(self.tellingenList)} gelinkte locposten, {len(self.prevLinkList)} vorige links en {len(self.nextLinkList)} volgende links."""

    ## Definieer een methode om naar de previous link te zoeken in een lijst van links
    def findPrevLinks(self, linkList):
        for link in linkList:
            if self.startNode == link.endNode and self.linkNo != link.linkNo:
                self.prevLinkList.append(link)

    ## Definieer een methode om naar de next link te zoeken in een lijst van links
    def findNextLinks(self, linkList):
        for link in linkList:
            if self.endNode == link.startNode and self.linkNo != link.linkNo:
                self.nextLinkList.append(link)
    
    ## Definieer een methode om de locposten van deze link te vinden uit een dictionary
    def findLocposten(self, locpost_dict, tellingenList):
        if (self.linkNo, self.startNode, self.endNode) in locpost_dict:
            locpostList = locpost_dict[(self.linkNo, self.startNode, self.endNode)]
            self.tellingenList = [telling for locpost in locpostList for telling in tellingenList if telling.locpost == locpost]
        else:
            self.tellingenList = []

    def berekenIntensiteiten(self):
        ## Neem het gemiddelde van de tellingen voor deze link (indien 1 telling dan is het de telling zelf)
        ## Indien we dit op meerdere manieren willen berekenen dan voegen we een extra laag toe aan de self.intensiteiten dictionary
        if len(self.tellingenList) > 0:
            self.intensiteiten["PW"] = sum(telling.tellingPW for telling in self.tellingenList) / len(self.tellingenList)
            self.intensiteiten["VR"] = sum(telling.tellingVR for telling in self.tellingenList) / len(self.tellingenList)
            self.intensiteiten["Totaal"] = sum(telling.tellingTotaal for telling in self.tellingenList) / len(self.tellingenList)


class Linkketen:
    def __init__(self, links):
        self.links = links
        self.numberOfLinks = len(links)
        self.startNode = links[0].startNode ## startnode van de eerste link in de keten
        self.endNode = links[-1].endNode ## eindnode van de laatste link in de keten
        self.bemeten = False
        self.extrapolatieIn = False
        self.extrapolatieUit = False
        self.intensiteiten = {"0-init":{"PW": None, "VR": None, "Totaal": None}} 
        self.nextLinkketenList = []
        self.prevLinkketenList = []

    def __repr__(self):
        return f"""Linkketen met {self.numberOfLinks} links, start node: {self.startNode}, eind node: {self.endNode}, 
        bemetingsgraad: {self.bemeten}, intensiteiten: {self.intensiteiten}, 
        {len(self.prevLinkketenList)} vorige linkketens, {len(self.nextLinkketenList)} volgende linkketens."""

    def bepaalBemeting(self):
        if self.numberOfLinks > 0 and any(len(link.tellingenList) > 0 for link in self.links):
            self.bemeten = True
        else:
            self.bemeten = False 

    def berekenInitieleTelling(self):
        if self.bemeten:
            links_met_tellingen = [link for link in self.links if len(link.tellingenList) > 0]
            self.intensiteiten["0-init"]["PW"] = sum(link.intensiteiten["PW"] for link in links_met_tellingen) / len(links_met_tellingen)
            self.intensiteiten["0-init"]["VR"] = sum(link.intensiteiten["VR"] for link in links_met_tellingen) / len(links_met_tellingen)
            self.intensiteiten["0-init"]["Totaal"] = sum(link.intensiteiten["Totaal"] for link in links_met_tellingen) / len(links_met_tellingen)
            self.intensiteiten["0-init"]["Berekening"] = "Waarde 1 link" if len(links_met_tellingen) == 1 else "Gemiddelde meerdere links"

    def berekenInkomendeIntensiteit(self, voertuigType):
        
        # Eerste keuze is om de intensiteiten te berekenen obv "0-init" intensiteiten
        # Wat als slechts een deel van de prevLinkketens bemeten is? Dan werken met geextrapoleerde waarden
        # To do: eerste if statement verwijderen en een break toevoegen in de for loop wanneer een linkketen geen intensiteiten heeft
        
        if all(linkketen.bemeten for linkketen in self.prevLinkketenList):
            # Sum the intensiteit[voertuigtype] for each linkketen in prevLinkketenList
            inkomende_intensiteit = 0
            
            for linkketen in self.prevLinkketenList:
                if linkketen.intensiteiten.get("0-init", {}).get(voertuigType) is not None:
                    inkomende_intensiteit += linkketen.intensiteiten["0-init"][voertuigType]
                    
                    # Subtract the intensiteit[voertuigtype] for each linkketen in the nextLinkketenList 
                    # of the linkketens of prevLinkketenList that are not the self linkketen
                    for next_linkketen in linkketen.nextLinkketenList:
                        if next_linkketen != self and next_linkketen.intensiteiten.get("0-init", {}).get(voertuigType) is not None:
                            inkomende_intensiteit -= next_linkketen.intensiteiten["0-init"][voertuigType]
            
            return inkomende_intensiteit
        
        return None

#     Initiele bemeting is geweten voor de loop, dus enkel extrapolities moeten gecheckt worden

# If extrapolatieIn is False: 
# 	berekenInkomendeIntensiteiten()
	
# If extrapolatieUit is False:
# berekenUitgaandeIntensiteiten()
    
    # Fout corrigeren: indien lengte van prevLinkketenList = 1 maar die linkketen heeft meer dan 1 
    # nextlink, dan is de berekening van de nieuwe intensiteit niet correct want met de uitstroom van de vorige linke wordt geen rekening gehouden
    def extrapoleerIntensiteiten(self, iteratie):
        if self.bemeten == False:
            if all(linkketen.bemeten for linkketen in self.nextLinkketenList):
                self.intensiteiten["PW"] = sum(linkketen.intensiteiten["PW"] for linkketen in self.nextLinkketenList)
                self.intensiteiten["VR"] = sum(linkketen.intensiteiten["VR"] for linkketen in self.nextLinkketenList)
                self.intensiteiten["Totaal"] = sum(linkketen.intensiteiten["Totaal"] for linkketen in self.nextLinkketenList)
                self.intensiteiten["Invulling"] = True
                self.bemeten = True

                return 1

            elif all(linkketen.bemeten for linkketen in self.prevLinkketenList):
                self.intensiteiten["PW"] = sum(linkketen.intensiteiten["PW"] for linkketen in self.prevLinkketenList)
                self.intensiteiten["VR"] = sum(linkketen.intensiteiten["VR"] for linkketen in self.prevLinkketenList)
                self.intensiteiten["Totaal"] = sum(linkketen.intensiteiten["Totaal"] for linkketen in self.prevLinkketenList)
                self.intensiteiten["Invulling"] = True
                self.bemeten = True

                return 1
            else:
                return 0
        else:
            return 0

    ## Definieer een methode om naar de vorige linkketen te zoeken in een lijst van linkketens
    def findPrevLinkketens(self, linkketenList):
        for linkketen in linkketenList:
            ## Zoek naar vorige linkketens obv hun eindnode. Linkketens met dezelfde elementen worden uitgesloten
            ## om dezelfde linkketen in omgekeerde rijriching te voorkomen.
            if self.startNode == linkketen.endNode and set(self.links) != set(linkketen.links): ## 
                self.prevLinkketenList.append(linkketen)

    ## Definieer een methode om naar de volgende linkketen te zoeken in een lijst van linkketens
    def findNextLinkketens(self, linkketenList):
        for linkketen in linkketenList:
            if self.endNode == linkketen.startNode and set(self.links) != set(linkketen.links):
                self.nextLinkketenList.append(linkketen)


class Telling:
    def __init__(self, locpost, tellingPW, tellingVR):
        self.locpost = locpost
        self.tellingPW = tellingPW
        self.tellingVR = tellingVR
        self.tellingTotaal = tellingPW + tellingVR
 


