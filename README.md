# Word embedding

(c) 2025 Hogeschool Utrecht  
Auteurs: Brian van der Bijl, Tijmen Muller (tijmen.muller@hu.nl)

- Studentnummer:
- Naam:
- Datum:

Werk onderstaande opdracht uit in Python. Lever een leesbaar verslag in een Jupyter Notebook op, maar codeer de benodigde functies in een losse module om het overzichtelijk te houden.


## Inleiding

Een _word embedding_ is een representatie van een woord met een hoogdimensionale vector, zodat woorden met een vergelijkbare betekenis dichtbij elkaar liggen in deze vectorruimte, terwijl woorden met weinig overeenkomsten juist een grote afstand hebben. 

In deze opdracht gaan we onderzoeken hoe we deze modellen kunnen gebruiken om te voorspellen welke woorden en zinnen een vergelijkbare betekenis hebben. In deel I onderzoeken we dit met een kleine dataset van landen en hoofdsteden, in deel II kijken we naar een grotere tekst als dataset, namelijk de verzamelde werken van Sherlock Holmes.


## Deel I: Vector space models

1. Lees de dataset `capitals.txt` in en bestudeer deze; ze bevat combinaties van landen en hoofdsteden.

2. Lees het bestand `en_embeddings.p` in met onderstaande code. Deze dataset is een selectie van de complete [Google News](https://code.google.com/archive/p/word2vec) word embedding dataset, die we hebben verkleind voor deze opdracht. Verken deze dataset en licht kort toe wat erin zit.

   ```py
   import pickle

   with open('data/en_embeddings.p', 'rb') as file:
      word_embeddings = pickle.load(file)
   ```

### Afstandsmaat

Voor het bepalen van de afstand tussen twee vectoren maken we gebruik van de _cosine similarity_. Deze is gegeven met de volgende formule:

$$\cos(\theta)=\frac{\langle a \mid b \rangle}{\|a\|\|b\|}=\frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2}\sqrt{\sum_{i=1}^{n} b_i^2}}$$

Hierbij is $n$ het aantal dimensies van beide vectoren.

Kenmerken:
- Als vectoren $\vec{a}$ en $\vec{b}$ identiek zijn, oftewel $\vec{a} $ = $ \vec{b}$, dan geldt $\cos(\theta) = 1$.
- Als vectoren $\vec{a}$ en $\vec{b}$  tegengesteld zijn, oftewel $\vec{a}$ = $- \vec{b}$, dan geldt $\cos(\theta) = -1$.
- Als vectoren $\vec{a}$ en $\vec{b}$ orthogonaal zijn (met andere woorden, ze staan haaks op elkaar), dan geldt $\cos(\theta) = 0$.
- Er geldt dus dat waarden tussen 0 en 1 _gelijkenis_ aangeven en waarden tussen −1 en 0 _niet-gelijkenis_ aangeven.


3. Schrijf een functie `cos_similarity(a, b)` die de cosine similarity tussen twee (woord)vectoren berekent.

   Input:
   - `a`: numpy array van een word-vector  
   - `b`: numpy array van een word-vector

   Output:
   - `cos`: getal dat de cosine similarity tussen $\vec{a}$ en $\vec{b}$ aangeeft

   Test je functie door de embeddings van de woorden `king` en `queen` te vergelijken. De verwachte cosine similarity van deze twee woorden is ongeveer 0.65 bij deze dataset.

### Analogieën voorspellen

Met de gegeven _word embeddings_ en een afstandsmaat kunnen we de betekenis van woorden met elkaar vergelijken. We doen dit eerst aan de hand van het concept '[analogie](https://nl.wikipedia.org/wiki/Analogie)'. Hierbij zoeken we het woord dat een relatie met een gegeven woord, die overeenkomt met de relatie tussen twee andere woorden. Bijvoorbeeld: 'koning' staat tot 'man' (de relatie), zoals 'vrouw' (het gegeven woord) staat tot 'koningin' (het gezochte woord).  
Als 'berekening' ziet dat er als volgt uit:

   ```
   king - man + woman -> queen
   ```

4. Schrijf een functie `find_analogy(word1, word2, word3)` die gegeven twee woorden met een relatie de analogie bij het gegeven derde woord vindt. 

   Input:
   - `word1`: string, bijvoorbeeld de hoofdstad van `country1` (bv. Athens)
   - `word2`: string, bijvoorbeeld het land van `city1` (bv. Greece)
   - `word3`: string, hoofdstad van een ander land (bv. Cairo)
   - `embeddings`: de word embeddings

   Output:
   - `word4`: de gevonden analogie bij `word3`

   Test je functie eerst met de genoemde voorbeelden:  
   - De analogie bij `('man', 'king', 'woman')` zou `queen` moeten zijn met een similarity van ~0.73.  
   - De analogie bij `('Athens', 'Greece', 'Cairo')` zou `Egypt` moeten zijn met een similarity van ~0.76.
   
   Vul deze tests aan met een aantal aansprekende analogieën die je zelf hebt gevonden.

5. Bereken de _accuracy_ van je model door voor alle rijen van de dataset een kolom te voorspellen aan de hand van de andere drie kolommen. Je zou tot een heel aardige score moeten komen.  
Als het niet lukt om binnen een redelijk tijd tot een goede score te komen, probeer dan te verklaren waarom hiet niet lukt. Je kan terugvallen op de word embeddings in `en_embeddings_capitals.p`, waar alleen de embeddings van de landen en steden uit de dataset in zijn opgenomen.

### Locality-sensitive hashing

Het zoeken in een hoogdimensionale vectorruimte met een groot aantal vectorrepresentaties is een complexe operatie. Een manier om dit te versimpelen is door gebruik te maken van _locality sensitive hashing_ (LSH), waarbij de ruimte middels een hash wordt opgedeeld in _buckets_ en alleen in één van die buckets wordt gezocht naar nabijgelegen _neighbours_. De winst in snelheid gaat ten koste van de nauwkeurigheid, want het beste antwoord hoeft niet in het bezochte deelgebied te liggen.

In de module `lshash.py` is een eenvoudige implementatie van LSH gegeven. Met `vector_hash()` wordt een hash gemaakt van een vector. De functie `make_hash_table()` maakt een _hash table_ van een gegeven lijst vectoren:

   Input:
   - `vecs`: list van vectoren met gelijke dimensie
   - `n_buckets`: het aantal buckets dat de hash table moet bevatten, een veelvoud van 2

   Output:
   - `buckets`: de buckets met daarin een deel van de vectoren uit `vecs` -- de aanname is dat deze vectoren nabij elkaar liggen
   - `lookup`: de index die de vectoren in `buckets` relateerd aan de positie in de originele lijst met vectoren `vecs`
   - `planes`: de lijst met vectoren die de vectorruimte opdeeld, zodat er `n_buckets` deelruimtes ontstaan -- hiervoor zijn $\log_{2}(\mathtt{n\_buckets})$ van deze vectoren nodig

6. Pas locality-sensitive hashing toe bij het voorspellen van de landen-hoofdsteden-combinaties. Welke winst in snelheid behaal je en wat lever je in qua _accuracy_?


## Deel II: Language models

In dit tweede deel van opdracht gaan we proberen om zinnen te vinden die _lijken_ op een gezochte zin. Dit gaan we doen door de embeddings van alle woordstammen per zin bij elkaar op te tellen. Vervolgens vergelijken we deze sommaties om de zinnen met de grootste overeenkomst in betekenis te vinden.

We maken gebruik van het boek [_The Adventures of Sherlock Holmes_](https://www.gutenberg.org/ebooks/48320) als dataset. De opdracht is om het CRISP-DM proces te volgen om tot een voorspellen model te komen.


### Data understanding & preparation

In deze stap moet je elke zin uit de originele tekst omzetten in een lijst met woordstammen. Deze woordstammen gaan we in de volgende stap omzetten in een _sentence embedding_, zodat we gelijkenis kunnen voorspellen. 

Ter illustratie, het boek begint met de volgende zinnen:

```
To Sherlock Holmes she is always _the_ woman. I have seldom heard him
mention her under any other name. In his eyes she eclipses and
predominates the whole of her sex.
```

#### Understanding

1. Analyseer de orinele tekst in `data/holmes.txt`. Bedenk welke data bruikbaar is, welke bijzonderheden er zijn in de tekst en wat er opgeschoond moet worden.

#### Cleaning

2. Laad de tekst en schoon deze op: verwijder onbruikbare delen en vervang woorden en leestekens waar nodig.

#### Transformation

3. Splits nu de tekst in zinnen. Het is handig om hier een reguliere expressie voor te gebruiken, bijvoorbeeld met [`nltk.tokenize.RegexpTokenizer()`](https://www.nltk.org/api/nltk.tokenize.RegexpTokenizer.html).  

   Ter illustratie, de eerste zinnen van het boek leveren dit op:

   ```py
   ['To Sherlock Holmes she is always the woman.',
    'I have seldom heard him mention her under any other name.',
    'In his eyes she eclipses and predominates the whole of her sex.']
   ```

   Zet de zinnen vervolgens om in een lijst met woordstammen. Hiertoe hak je de zinnen op in een lijst met woorden; een goede reguliere expressie hiervoor is `r'\w+|\$[\d\.]+|\S+'`. Vervolgens gebruik je `nltk.stem.PorterStemmer()` om de woorden te stammen. Dit is ook een goed moment om stopwoorden te verwijderen.

   Ter illustratie, de eerste zinnen van het boek hebben nu als eindresultataat deze lijst met woordstammen:
   
   ```py
   [['to', 'sherlock', 'holm', 'alway', 'woman'],
    ['i', 'seldom', 'heard', 'mention', 'name'],
    ['in', 'eye', 'eclips', 'predomin', 'whole', 'sex']]
   ```


### Modeling

We gaan nu een voorspellend model maken om vergelijkbare zinnen in het boek te vinden gegeven een 'zoekzin'. Hiertoe zetten we elke zin uit het boek via de lijst met woordstammen uit de vorige stap om in een _sentence embedding_. Een sentence embedding is hier niks meer dan de sommatie van alle word embeddings van de woordstammen uit die zin.

4. Zet alle lijsten met woordstammen om in een _sentence embedding_ door weer gebruik te maken van de Google News word embedding dataset in  `data/en_embeddings.p`.  
Let op! De woorden in deze embedding dataset zijn _niet_ gestamd!

   Ter illustratie, de eerste drie zinnen van het boek resulteren dus in een matrix met shape (3, 300): 3 vectoren (één per zin) met een dimensie van 300.

5. Gegeven een 'zoekzin', bepaal met behulp van de _cosine similarity_ welke zin uit het boek het meest hiermee overeenkomt.

   Ter illustratie, gegeven de zoekzin `He rarely knows my age.`, de similarities van de eerste drie zinnen uit het boek zijn:

   - `To Sherlock Holmes she is always the woman.`: $\cos(\theta) = 0.318$
   - `I have seldom heard him mention her under any other name.`: $\cos(\theta) = 0.312$
   - `In his eyes she eclipses and predominates the whole of her sex.`: $\cos(\theta) = 0.290$


### Evaluation

6. Onderzoek de prestaties van je model in ieder geval door een variatie aan zoekzinnen te proberen. Vind je de resultaten logisch en waarom (niet)? 

Voer naar eigen inzicht verbeteringen door in je implementatie. Zorg dat je inlevering bestaat uit een goed leesbaar verslag in Jupyter Notebook en begrijpelijk code.
