#%% Import libraries and classes
import pandas as pd 
import geopandas as gpd

from klasseDefinities import *
from functies import findStartLinks, createLinkKetens

#%% IMPORTEER DATA
root = r"E:/v423_ONTW_SPM/04 Diverse onderzoeken/34 Gesloten wegvakprobleem/"

## Data van knopen
nodes = gpd.read_file(root + r'Masternetw/MASTER_2022_20240808_ACT_node.SHP')
nodes = nodes.drop(columns='geometry')
nodes = nodes[nodes['STUDIEGE~3'].isin([1, 2])] # enkel Vlaanderen behouden
nodes = nodes[['NO']] 

## Data van linken
links = gpd.read_file(root + r'Masternetw/MASTER_2022_20240808_ACT_link.SHP')
links = links.drop(columns='geometry')
links = links[links['FROMNODENO'].isin(nodes['NO']) & links['TONODENO'].isin(nodes['NO'])]
links = links[['FROMNODENO', 'TONODENO', 'NO']]

## Data van locposten met linkid
count_loc = gpd.read_file(root + r'Masternetw/MASTER_2022_20240808_ACT_countlocation.SHP')
count_loc = count_loc.drop(columns='geometry')
count_loc = count_loc[['CODE', 'FROMNODENO', 'TONODENO', 'LINKNO']]

## Data van tellingen 
tellingen = pd.read_csv(root + r'teldatabank_v423 test.csv', sep=',', encoding='utf-8')
tellingen['TEL_PW_TOTAAL'] = tellingen[[f'TEL_PW_{str(i).zfill(2)}' for i in range(24)]].sum(axis=1)
tellingen['TEL_VR_TOTAAL'] = tellingen[[f'TEL_VR_{str(i).zfill(2)}' for i in range(24)]].sum(axis=1)

print(nodes.shape, links.shape, count_loc.shape, tellingen.shape)


# %% LINK OBJECTEN AANMAKEN EN ATTRIBUTEN INVULLEN

## Link dataframe omzetten naar een lijst van Link objecten
linkList = [Link(row.FROMNODENO, row.TONODENO, row.NO) for row in links.itertuples(index=False)]

## Tellingen dataframe omzetten naar een lijst van Telling objecten
tellingenList = [Telling(row.Code, row.TEL_PW_TOTAAL, row.TEL_VR_TOTAAL) for row in tellingen.itertuples(index=False)]

## Maak obv de locpostdata een dictionary dat linkid's (key) mapt naar (een lijst van) locpost codes (value)
locpost_dict = count_loc.groupby(['LINKNO','FROMNODENO','TONODENO'])['CODE'].apply(list).to_dict()

## Lijst met vorige en volgende links en lijst met locposten aanmaken voor elk link object
for link in linkList:
    link.findPrevLinks(linkList)
    link.findNextLinks(linkList)
    link.findLocposten(locpost_dict, tellingenList)
    link.berekenIntensiteiten()
#findLocposten nog aanpassen want nu wordt geen rekening gehouden met de rijrichting

# %% LINKKETENS AANMAKEN: aaneenschakeling van links die elkaar opvolgen tot er een volgende knoop is 
    
## Maak een subset van links aan die fungeren als de start van een linkketen omdat ze aan de volgende voorwaarden voldoen:
## - Ze hebben geen vorige link of meer dan 1 vorige link
## - OF: Ze hebben 1 vorige link, maar die vorige link heeft meer dan 1 volgende link (bv op het einde van vorige link is er een afrit)
startLinks = findStartLinks(linkList)

## Maak een lijst van linkketen objecten aan 
linkketenList = createLinkKetens(startLinks)

## Vul de attributen van de linkketens in 
for linkketen in linkketenList:
    linkketen.bepaalBemeting() # Maak boolean die aangeeft om de linkketen minstens 1 link met tellingen bevat
    linkketen.findPrevLinkketens(linkketenList) # Zoek de vorige linkketens
    linkketen.findNextLinkketens(linkketenList) # Zoek de volgende linkketens
    linkketen.berekenInitieleTelling() # Bereken de initiële tellingen obv de tellingen van de links in de linkketen



# %% Tel het tottaal aantal links in alle linkketens
total_links = sum(linkketen.numberOfLinks for linkketen in linkketenList)
print(f"Totaal aantal linkketens: {len(linkketenList)}")
print(f"Totaal aantal links in link ketens: {total_links}")

## Aanmaken van lijst met links die niet voorkomen in de lijst met linkketens
links_in_ketens = {link for keten in linkketenList for link in keten.links}
links_not_in_ketens = [link for link in linkList if link not in links_in_ketens]
#print(links_not_in_ketens)
# Dit zijn linken die in een cirkel zitten (bijv. A -> B -> C -> A)
# Is dit een fout in de data of is dit gewoon onderdeel van het netwerk?


# %% LOOP OM DE INTENSITEITEN VAN DE LINKKETENS TE EXTRAPOLEREN 
## De loop stopt wanneer er geen updates meer zijn
aantalUpdates = 1
iteratie = 0

while aantalUpdates != 0:
    iteratie += 1
    aantalUpdates = 0
    for linkketen in linkketenList: 
        aantalUpdates += linkketen.extrapoleerIntensiteiten(iteratie)
    print(f"Aantal updates: {aantalUpdates}")

## Print het aantal linkketens die zijn bijgewerkt
print("Totaal aantal linkketens:")
print(len(linkketenList))

print("Totaal aantal linkketens die bemeten zijn (al dan niet na invulling):")
print(sum(1 for linkketen in linkketenList if linkketen.bemeten))

print("Totaal aantal linkketens die zijn ingevuld:")
print(sum(1 for linkketen in linkketenList if linkketen.intensiteiten["Invulling"]))


# %% Test dooor enkele linkketens te printen
for linkketen in linkketenList[0:20]: 
    print(linkketen)
    print("Deze linkketen bevat de volgende links:")
    for link in linkketen.links:
        print(link)
