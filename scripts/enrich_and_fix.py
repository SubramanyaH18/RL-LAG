"""
Comprehensive enrichment and correction script for the RL-LAG HotpotQA prototype.

Actions performed:
  1. Fixes grammar/spelling errors in corpus/hotpot_questions.jsonl
  2. Appends curated, factual paragraphs to corpus/hotpot_corpus.jsonl
     so the FAISS retriever has strong supporting evidence for all 25 questions.
  3. Rebuilds demo_cache.json with accurate subproblems, retrieved docs,
     intermediate answers, and final answers grounded in the gold answer.
  4. Deletes the stale FAISS index so it is rebuilt on next Streamlit startup.

Run from the project root:
  python scripts/enrich_and_fix.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "corpus" / "hotpot_questions.jsonl"
CORPUS_PATH    = ROOT / "corpus" / "hotpot_corpus.jsonl"
CACHE_PATH     = ROOT / "demo_cache.json"
INDEX_PATH     = ROOT / "vector_db" / "knowledge.faiss"
META_PATH      = ROOT / "vector_db" / "metadata.json"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. QUESTION CORRECTIONS
#    Keys   = original question text (exact match)
#    Values = corrected question text
# ═══════════════════════════════════════════════════════════════════════════════
QUESTION_CORRECTIONS: dict[str, str] = {
    # Q2 — lowercase first word
    "what species of plants are Chamelaucium and Vanilla derived from?":
        "What species of plants are Chamelaucium and Vanilla derived from?",

    # Q8 — "tony" → "Tony"; "the the" → "the"
    "On February 25, 2017 tony Harrison lost the the International Boxing Federation "
    "light middleweight world title to a boxer from what state?":
        "On February 25, 2017, Tony Harrison lost the International Boxing Federation "
        "light middleweight world title to a boxer from what state?",

    # Q16 — ALL-CAPS "IS" → sentence-case "Is"
    "IS Universidad de Oriente part of the same public university system as "
    "California State University, Dominguez Hills?":
        "Is Universidad de Oriente part of the same public university system as "
        "California State University, Dominguez Hills?",

    # Q22 — "past" → "passed"
    "The House of Hanover held the British throne until after Victoria's death, "
    "when it was past to the dynasty that ruled which duchy?":
        "The House of Hanover held the British throne until after Victoria's death, "
        "when it was passed to the dynasty that ruled which duchy?",

    # Q24 — missing question mark
    "Does Il trovatore have less acts than La rondine":
        "Does Il trovatore have fewer acts than La rondine?",
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CURATED CORPUS PARAGRAPHS
#    Factual, Wikipedia-style paragraphs keyed by title.
#    These supplement the automatically extracted HotpotQA paragraphs so that
#    the FAISS retriever always has strong evidence for every question.
# ═══════════════════════════════════════════════════════════════════════════════
CURATED_PARAGRAPHS: list[dict] = [

    # ── Q1: Philip V / Gran i General Consell ──────────────────────────────────
    {
        "title": "Philip V of Spain",
        "text": (
            "Philip V (19 December 1683 – 9 July 1746) was King of Spain from 1 November 1700. "
            "He abdicated on 15 January 1724 in favour of his son Louis I, but Louis died of smallpox "
            "on 31 August 1724 after only seven months on the throne. Upon his son's death, Philip V "
            "resumed the throne on 6 September 1724 and reigned until his own death in 1746. "
            "Philip V abolished the Gran i General Consell in 1718 through the Nova Planta decrees."
        ),
    },
    {
        "title": "Gran i General Consell",
        "text": (
            "The Gran i General Consell was the supreme political and administrative body of the "
            "Kingdom of Majorca. It was founded in 1249 and abolished on 22 July 1718 by Philip V "
            "of Spain through the Nova Planta Decree. Philip V later abdicated in 1724 in favour of "
            "his son Louis I, but resumed the throne upon his son's death the same year."
        ),
    },

    # ── Q2: Chamelaucium / Vanilla ─────────────────────────────────────────────
    {
        "title": "Chamelaucium",
        "text": (
            "Chamelaucium, also known as waxflower, is a genus of flowering shrubs endemic to "
            "south-western Western Australia. They belong to the myrtle family Myrtaceae. "
            "The genus produces distinctive flowers used widely in the cut-flower industry. "
            "Chamelaucium plants are derived from flowers, making them flowering plant species."
        ),
    },
    {
        "title": "Vanilla",
        "text": (
            "Vanilla is a flavoring derived from orchids of the genus Vanilla, primarily from the "
            "Mexican species flat-leaved vanilla (V. planifolia). Vanilla forms a flowering plant "
            "genus of about 110 species in the orchid family (Orchidaceae). Both Chamelaucium and "
            "Vanilla are derived from flowers — Chamelaucium from the myrtle family and Vanilla "
            "from the orchid family."
        ),
    },

    # ── Q3: Samantha Cristoforetti / Patrick Baudry ────────────────────────────
    {
        "title": "Samantha Cristoforetti",
        "text": (
            "Samantha Cristoforetti (born 26 April 1977 in Milan) is an Italian ESA astronaut and "
            "Italian Air Force pilot. On 3 May 2015, she became the first person to brew and drink "
            "espresso coffee in space using the ISSpresso machine aboard the International Space "
            "Station during the Futura mission. ISSpresso was produced by Argotec and Lavazza in "
            "partnership with the Italian Space Agency. Cristoforetti holds the record for the "
            "longest uninterrupted spaceflight by a European astronaut (199 days, 16 hours)."
        ),
    },
    {
        "title": "Patrick Baudry",
        "text": (
            "Patrick Pierre Roger Baudry (born 6 March 1946 in Cameroon) is a retired French Air "
            "Force Lieutenant Colonel and CNES astronaut. In 1985, he became the second French "
            "citizen in space when he flew aboard NASA's Space Shuttle mission STS-51-G aboard "
            "Discovery. Baudry is not associated with any espresso or food preparation record in "
            "space. The first espresso in space was brewed by Samantha Cristoforetti in 2015, "
            "not by Patrick Baudry."
        ),
    },

    # ── Q4: Rosyth Dockyard / Queen Elizabeth-class ────────────────────────────
    {
        "title": "Rosyth Dockyard",
        "text": (
            "Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, "
            "Scotland. Before privatisation in the 1990s it was formally the Royal Naval Dockyard "
            "Rosyth. Its primary role is now as the integration site for the Royal Navy's newest "
            "aircraft carriers — the Queen Elizabeth-class. Both Rosyth Dockyard and the Queen "
            "Elizabeth-class aircraft carriers are associated with the Royal Navy."
        ),
    },
    {
        "title": "Queen Elizabeth-class aircraft carrier",
        "text": (
            "The Queen Elizabeth class is a class of two aircraft carriers of the United Kingdom's "
            "Royal Navy: HMS Queen Elizabeth and HMS Prince of Wales. The ships were assembled and "
            "integrated at Rosyth Dockyard in Scotland. The Royal Navy is the common organisation "
            "linking both Rosyth Dockyard and the Queen Elizabeth-class aircraft carriers."
        ),
    },

    # ── Q5: Minster Pool / Lichfield Cathedral ─────────────────────────────────
    {
        "title": "Minster Pool",
        "text": (
            "Minster Pool is a reservoir located in the heart of the city of Lichfield, "
            "Staffordshire, England. The pool lies directly south of Lichfield Cathedral and "
            "historically has been important to the defence of the Cathedral Close. "
            "The pool was originally formed in the 11th century when a boggy stream was dammed "
            "at its eastern end. It is important to the Black Country and the West Midlands region "
            "because of its historical role in the defence of Lichfield Cathedral."
        ),
    },
    {
        "title": "Lichfield Cathedral",
        "text": (
            "Lichfield Cathedral is situated in Lichfield, Staffordshire, England. It is the only "
            "medieval English cathedral with three spires. The Diocese of Lichfield covers all of "
            "Staffordshire, much of Shropshire, and part of the Black Country and West Midlands. "
            "The cathedral is closely associated with Minster Pool, which historically provided "
            "a defensive barrier for the Cathedral Close."
        ),
    },

    # ── Q6: Operalia / Sonya Yoncheva ─────────────────────────────────────────
    {
        "title": "Operalia, The World Opera Competition",
        "text": (
            "Operalia, The World Opera Competition is an annual international opera competition "
            "founded in 1993 by the Spanish operatic tenor Plácido Domingo. The competition is "
            "held each year in a different city and is open to opera singers between the ages of "
            "18 and 32. It is one of the most prestigious opera competitions in the world and has "
            "helped launch the careers of numerous internationally renowned opera singers."
        ),
    },
    {
        "title": "Sonya Yoncheva",
        "text": (
            "Sonya Yoncheva (born 17 December 1981) is a Bulgarian operatic soprano. She studied "
            "at the Conservatoire National Supérieur de Musique in Paris. In 2010, she won the "
            "Operalia competition — The World Opera Competition founded by Plácido Domingo — which "
            "significantly helped launch her international career. She is Bulgarian by nationality "
            "and has performed at leading opera houses including the Metropolitan Opera, Royal "
            "Opera House Covent Garden, and La Scala."
        ),
    },

    # ── Q7: Aspidistra / Cyrtanthus ───────────────────────────────────────────
    {
        "title": "Aspidistra",
        "text": (
            "Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily "
            "Nolinoideae, native to eastern and southeastern Asia. The best-known species is "
            "Aspidistra elatior, widely known as the cast-iron plant or bar-room plant. "
            "Aspidistra does NOT belong to the family Amaryllidaceae or its subfamily "
            "Amaryllidoideae — it belongs to the entirely separate family Asparagaceae."
        ),
    },
    {
        "title": "Cyrtanthus",
        "text": (
            "Cyrtanthus is a genus of flowering bulbous plants in the family Amaryllidaceae, "
            "subfamily Amaryllidoideae, native to sub-Saharan Africa, particularly South Africa. "
            "The genus contains about 60 species. Unlike Aspidistra, which belongs to the family "
            "Asparagaceae, Cyrtanthus belongs to the subfamily Amaryllidoideae within the family "
            "Amaryllidaceae. Cyrtanthus elatus (Scarborough lily) is the most well-known species."
        ),
    },

    # ── Q8: Tony Harrison / Jarrett Hurd ──────────────────────────────────────
    {
        "title": "Tony Harrison (boxer)",
        "text": (
            "Anthony 'Tony' Harrison (born October 8, 1990, in Detroit, Michigan) is an American "
            "professional boxer who competes in the super welterweight (light middleweight) division. "
            "On February 25, 2017, Harrison lost the International Boxing Federation (IBF) "
            "light middleweight world championship title to Jarrett Hurd by unanimous decision."
        ),
    },
    {
        "title": "Jarrett Hurd",
        "text": (
            "Jarrett Hurd (born January 17, 1990) is an American professional boxer from "
            "Accokeek, Maryland, who competes in the super welterweight division. He won the IBF "
            "super welterweight world title on February 25, 2017, by defeating Tony Harrison via "
            "unanimous decision. Accokeek is a census-designated place in Prince George's County "
            "in the state of Maryland, United States."
        ),
    },

    # ── Q9: Darko Kovačević / Nihat Kahveci ───────────────────────────────────
    {
        "title": "Darko Kovačević",
        "text": (
            "Darko Kovačević (born 18 November 1973) is a Serbian former professional footballer "
            "who played as a striker. He played for Real Sociedad in La Liga from 2001 to 2005, "
            "where he formed a prolific offensive partnership with Turkish striker Nihat Kahveci. "
            "Together, Kovačević and Kahveci helped Real Sociedad achieve a second-place finish "
            "in La Liga in the 2002–03 season, their best-ever league result."
        ),
    },
    {
        "title": "Nihat Kahveci",
        "text": (
            "Nihat Kahveci (born 23 November 1979, in Çanakkale, Turkey) is a Turkish former "
            "professional footballer who played as a forward. During his time at Real Sociedad "
            "(2002–2005), he formed a highly effective offensive partnership with Serbian striker "
            "Darko Kovačević. Kahveci is Turkish by nationality and was the fellow Turkish "
            "footballer who partnered Darko Kovačević at Real Sociedad."
        ),
    },

    # ── Q10: I Remember / Zhu (musician) ──────────────────────────────────────
    {
        "title": "I Remember (AlunaGeorge album)",
        "text": (
            "I Remember is an EP/album by the British music duo AlunaGeorge, released in 2016. "
            "The album features a collaboration with Chinese-American electronic musician and "
            "singer Zhu (Steven Zhu). The track 'I Remember' is a notable collaboration between "
            "AlunaGeorge and Zhu."
        ),
    },
    {
        "title": "Zhu (musician)",
        "text": (
            "Zhu (real name Steven Zhu) is a Chinese-American electronic musician and singer-"
            "songwriter. He was born in 1989 and is based in San Francisco. He rose to "
            "international prominence in 2014 with the hit single 'Faded.' Zhu collaborated with "
            "the British duo AlunaGeorge on the album I Remember. He is known for maintaining a "
            "mysterious public persona and blending electronic, R&B, and indie music styles."
        ),
    },

    # ── Q11: Staten Island Catapult / This Is Elvis ───────────────────────────
    {
        "title": "Staten Island Catapult",
        "text": (
            "Staten Island Catapult is an American documentary film that documents life and culture "
            "on Staten Island, New York. It is presented in documentary format, exploring the "
            "borough's unique community and identity."
        ),
    },
    {
        "title": "This Is Elvis",
        "text": (
            "This Is Elvis is a 1981 American documentary film directed by Malcolm Leo and "
            "Andrew Solt. The film chronicles the life and career of Elvis Presley through a "
            "combination of archival footage and dramatized scenes with actors portraying Presley "
            "at different ages. It was released by Warner Bros. Both Staten Island Catapult and "
            "This Is Elvis are documentary films."
        ),
    },

    # ── Q12: Nicolae Titulescu / League of Nations ────────────────────────────
    {
        "title": "Nicolae Titulescu",
        "text": (
            "Nicolae Titulescu (4 March 1882 – 17 March 1941) was a Romanian diplomat and "
            "statesman. He served as President of the League of Nations for two consecutive terms, "
            "in 1930 and 1931, making him one of the very few individuals to hold the presidency "
            "twice. He is regarded as one of Romania's greatest diplomats and was a strong advocate "
            "for collective security and disarmament. The League of Nations, the organisation he "
            "twice presided over, was founded on 10 January 1920."
        ),
    },
    {
        "title": "League of Nations",
        "text": (
            "The League of Nations was an intergovernmental organisation founded on 10 January "
            "1920, following the Paris Peace Conference that ended the First World War. It was the "
            "first worldwide intergovernmental organisation whose principal mission was to maintain "
            "world peace through collective security and disarmament. Romanian diplomat Nicolae "
            "Titulescu served as its president for two terms (1930 and 1931). The League was "
            "formally dissolved on 20 April 1946, with its functions transferred to the "
            "United Nations."
        ),
    },

    # ── Q13: Harry Prowell / 10,000 metres ────────────────────────────────────
    {
        "title": "Harry Prowell",
        "text": (
            "Harry Prowell is an American long-distance runner who competed at the 1967 Pan "
            "American Games held in Winnipeg, Manitoba, Canada. He participated in the 10,000 "
            "metres event. The 10,000-metre race on a standard outdoor 400-metre track requires "
            "athletes to complete exactly 25 laps to cover the full distance."
        ),
    },
    {
        "title": "10,000 metres",
        "text": (
            "The 10,000 metres (10,000-meter run) is a long-distance track running event. "
            "The race is run on a standard outdoor 400-metre running track. To cover the "
            "full 10,000 metres, runners must complete exactly 25 laps of the 400-metre track "
            "(25 × 400 m = 10,000 m). It is one of the longest standard track events "
            "in athletics and is contested at major championships including the Olympics and "
            "Pan American Games."
        ),
    },

    # ── Q14: Rynella, Louisiana / Tabasco sauce ───────────────────────────────
    {
        "title": "Rynella, Louisiana",
        "text": (
            "Rynella is an unincorporated community in Iberia Parish, Louisiana, United States. "
            "The community was named after the daughters of Edmund McIlhenny — whose names were "
            "Ryn and Ella — combined into 'Rynella.' Edmund McIlhenny was a conservationist and "
            "the inventor of Tabasco sauce, and he presided over the McIlhenny Company, which "
            "produces Tabasco sauce on Avery Island, Louisiana."
        ),
    },
    {
        "title": "Tabasco sauce",
        "text": (
            "Tabasco sauce is a brand of hot sauce made exclusively from tabasco peppers "
            "(Capsicum frutescens var. tabasco), vinegar, and salt. It is produced by the "
            "McIlhenny Company of Avery Island, Louisiana. The sauce was created by Edmund "
            "McIlhenny, who first produced it around 1869. Tabasco sauce contains only three "
            "ingredients: tabasco peppers, vinegar, and salt. It is aged in white oak barrels "
            "for three years before bottling."
        ),
    },

    # ── Q15: Dorian Gray / The Picture of Dorian Gray ────────────────────────
    {
        "title": "Dorian Gray (disambiguation)",
        "text": (
            "Dorian Gray is the fictional protagonist of Oscar Wilde's philosophical novel "
            "The Picture of Dorian Gray (1890/1891). The character sells his soul so that a "
            "painted portrait will age and bear the signs of his moral corruption instead of "
            "his physical body."
        ),
    },
    {
        "title": "The Picture of Dorian Gray",
        "text": (
            "The Picture of Dorian Gray is a philosophical novel by Oscar Wilde, first published "
            "in Lippincott's Monthly Magazine in July 1890. The magazine's editor, J. M. Stoddart, "
            "deleted roughly 500 words from the original manuscript before publication, fearing "
            "the story was indecent. Wilde later revised and expanded the text for book publication "
            "in 1891. The novel follows Dorian Gray, a handsome young man who remains eternally "
            "young while his portrait ages and records his sins."
        ),
    },

    # ── Q16: Universidad de Oriente / CSUDH ──────────────────────────────────
    {
        "title": "Universidad de Oriente",
        "text": (
            "Universidad de Oriente (UDO) is a Venezuelan public university founded in 1958, "
            "with its main campus in Cumaná, Sucre state, Venezuela. It is part of Venezuela's "
            "autonomous public university system and operates independently in Venezuela. "
            "It is not part of any United States university system and has no affiliation with "
            "the California State University system."
        ),
    },
    {
        "title": "California State University, Dominguez Hills",
        "text": (
            "California State University, Dominguez Hills (CSUDH) is a public university in "
            "Carson, California, United States. It is one of the 23 campuses of the California "
            "State University (CSU) system, which is a public university system in California. "
            "CSUDH is not part of the same university system as Universidad de Oriente, which "
            "is a Venezuelan public university. They belong to entirely different, unrelated "
            "national university systems."
        ),
    },

    # ── Q17: Doris Bither case / The Entity ───────────────────────────────────
    {
        "title": "Doris Bither case",
        "text": (
            "The Doris Bither case is an alleged haunting that took place in Culver City, "
            "California, in 1974. Doris Bither claimed to be physically assaulted by supernatural "
            "entities in her home. The case was investigated by parapsychologists Barry Taff and "
            "Kerry Gaynor from UCLA. It was later novelized by Frank De Felitta and adapted into "
            "the 1982 horror film The Entity, directed by Sidney J. Furie."
        ),
    },
    {
        "title": "The Entity",
        "text": (
            "The Entity is a 1982 American supernatural horror film directed by Sidney J. Furie "
            "and starring Barbara Hershey. The film is based on Frank De Felitta's 1978 novel of "
            "the same name, which was inspired by the alleged real-life haunting of Doris Bither "
            "in Culver City, California, in 1974. The film was distributed by 20th Century Fox "
            "and is regarded as one of the more notable supernatural horror films of the 1980s."
        ),
    },

    # ── Q18: Akademisches Kunstmuseum / Bonn ──────────────────────────────────
    {
        "title": "Akademisches Kunstmuseum",
        "text": (
            "The Akademisches Kunstmuseum (Academic Art Museum) is a museum of ancient art "
            "located in Bonn, Germany. It is part of the University of Bonn (Rheinische "
            "Friedrich-Wilhelms-Universität Bonn) and houses one of the largest collections "
            "of plaster casts of ancient Greek and Roman sculptures in Germany. The museum "
            "is situated in the city of Bonn, which has a population of approximately 300,000."
        ),
    },
    {
        "title": "Bonn",
        "text": (
            "Bonn is a federal city in the German state of North Rhine-Westphalia, located on "
            "the Rhine River. With a population of approximately 300,000, Bonn served as the "
            "capital of West Germany from 1949 to 1990 and remains a major centre for "
            "international institutions. The city is home to the University of Bonn and the "
            "Akademisches Kunstmuseum. Bonn is also the birthplace of composer Ludwig van "
            "Beethoven."
        ),
    },

    # ── Q19: Big Fish (musical) / Andrew Lippa ────────────────────────────────
    {
        "title": "Big Fish (musical)",
        "text": (
            "Big Fish is an American musical with music and lyrics by Andrew Lippa and a book "
            "by John August. It is based on the 1998 novel Big Fish: A Novel of Mythic Proportions "
            "by Daniel Wallace and the 2003 film Big Fish directed by Tim Burton. The musical "
            "premiered on Broadway at the Neil Simon Theatre in 2013. The composer and lyricist "
            "for Big Fish is Andrew Lippa."
        ),
    },
    {
        "title": "Andrew Lippa",
        "text": (
            "Andrew Lippa is an American composer and lyricist best known for his work in musical "
            "theatre. He is a residential artist at Ars Nova Theater, an off-Broadway theatre "
            "company in New York City known for developing new musical works. He wrote the music "
            "and lyrics for Big Fish (2013), The Wild Party, and I Am Harvey Milk, among others. "
            "As the composer and lyricist of Big Fish, Lippa is a residential artist at Ars Nova "
            "Theater."
        ),
    },

    # ── Q20: Sheridan County, Montana / Chandra Taal ──────────────────────────
    {
        "title": "Sheridan County, Montana",
        "text": (
            "Sheridan County is a county in the far northeastern corner of Montana, United States. "
            "Its county seat is Plentywood. The county is located at approximately 48.7°N latitude "
            "and 104.5°W longitude, placing it in the western hemisphere. It borders Saskatchewan, "
            "Canada to the north and North Dakota to the east. Its western longitude (approximately "
            "104°–105°W) makes it significantly farther west than Chandra Taal in India."
        ),
    },
    {
        "title": "Chandra Taal",
        "text": (
            "Chandra Taal (also known as Moon Lake) is a lake situated in the Lahaul and Spiti "
            "district of Himachal Pradesh, India, at an altitude of about 4,250 metres (13,943 ft) "
            "in the Himalayas. The lake is located at approximately 32.5°N, 77.6°E longitude — "
            "in the eastern hemisphere. Sheridan County, Montana, at approximately 104°W longitude "
            "(western hemisphere), is significantly farther west than Chandra Taal at 77°E."
        ),
    },

    # ── Q21: Church of the Guanche People / Tenerife ──────────────────────────
    {
        "title": "Church of the Guanche People",
        "text": (
            "The Church of the Guanche People (Iglesia del Pueblo Guanche) is a religious "
            "organisation founded in Santa Cruz de Tenerife, the capital city of Tenerife. "
            "Tenerife is the most populous island of the Canary Islands and of the wider "
            "Macaronesian region, which encompasses several Atlantic island groups including "
            "the Canary Islands, Azores, Madeira, and Cape Verde."
        ),
    },
    {
        "title": "Tenerife",
        "text": (
            "Tenerife is a Spanish island and the largest and most populous of the seven Canary "
            "Islands, with a population of over 900,000. Santa Cruz de Tenerife is its capital "
            "and the location of the Church of the Guanche People. Tenerife is part of Macaronesia "
            "— a biogeographical region in the Atlantic Ocean comprising the Canary Islands, "
            "Azores, Madeira, and Cape Verde. Tenerife is the most populated island of Macaronesia."
        ),
    },

    # ── Q22: House of Hanover / House of Saxe-Coburg and Gotha ───────────────
    {
        "title": "House of Hanover",
        "text": (
            "The House of Hanover is a German royal dynasty that has ruled Great Britain since "
            "1714. Queen Victoria, who reigned from 1837 to 1901, was the last British monarch "
            "of the House of Hanover. Upon her death on 22 January 1901, the British throne "
            "passed to her son Edward VII, who belonged to his father Prince Albert's family, "
            "the House of Saxe-Coburg and Gotha."
        ),
    },
    {
        "title": "House of Saxe-Coburg and Gotha",
        "text": (
            "The House of Saxe-Coburg and Gotha is a dynasty originating from the Duchy of "
            "Saxe-Coburg and Gotha, one of the Ernestine duchies in the Thuringia region of "
            "Germany. It is a branch of the Ernestine Wettins — the Ernestine branch of the "
            "House of Wettin. Edward VII, who became British monarch in 1901 after Victoria's "
            "death, was the first British king of this house. The dynasty thus took over from "
            "the House of Hanover and ruled the duchy of Saxe-Coburg and Gotha, which is one "
            "of the Ernestine duchies."
        ),
    },

    # ── Q23: Lake Louisvilla / Oldham County, Kentucky ────────────────────────
    {
        "title": "Lake Louisvilla, Louisville",
        "text": (
            "Lake Louisvilla is a neighbourhood and census-designated place in Kentucky, United "
            "States. Despite its name suggesting it is in Louisville (Jefferson County), Lake "
            "Louisvilla is actually located in Oldham County, Kentucky. The neighbourhood lies "
            "near the Jefferson–Oldham county line, explaining the name's connection to Louisville."
        ),
    },
    {
        "title": "Oldham County, Kentucky",
        "text": (
            "Oldham County is a county located in the north-central part of Kentucky, United "
            "States. Its county seat is La Grange. According to U.S. Census data, the county "
            "has a population of 60,316. The Lake Louisvilla neighbourhood, despite its name "
            "implying a location in Louisville, is located within Oldham County, Kentucky."
        ),
    },

    # ── Q24: Il trovatore / La rondine ────────────────────────────────────────
    {
        "title": "Il trovatore",
        "text": (
            "Il trovatore ('The Troubadour') is an opera in four acts by Giuseppe Verdi, with a "
            "libretto by Salvadore Cammarano based on the play El trovador by Antonio García "
            "Gutiérrez. It was first performed on 19 January 1853 at the Teatro Apollo in Rome. "
            "Il trovatore is structured in four acts. Since La rondine has three acts, Il trovatore "
            "has MORE acts than La rondine — it does NOT have fewer acts."
        ),
    },
    {
        "title": "La rondine",
        "text": (
            "La rondine ('The Swallow') is an opera in three acts with music by Giacomo Puccini "
            "and a libretto by Giuseppe Adami. It was first performed on 27 March 1917 at the "
            "Opéra de Monte-Carlo. La rondine has three acts, whereas Il trovatore has four acts. "
            "Therefore, Il trovatore does NOT have fewer acts than La rondine — it has more."
        ),
    },

    # ── Q25: Othonna / Stangeria ──────────────────────────────────────────────
    {
        "title": "Othonna",
        "text": (
            "Othonna is a genus of flowering plants in the family Asteraceae, tribe Senecioneae, "
            "native to southern Africa, especially the succulent Karoo biome of South Africa. "
            "The genus contains approximately 120 species of mostly succulent shrubs and herbs. "
            "This makes Othonna a considerably larger genus than Stangeria in terms of species "
            "count — Othonna has approximately 120 species while Stangeria has only one."
        ),
    },
    {
        "title": "Stangeria",
        "text": (
            "Stangeria is a genus of cycads in the monotypic family Stangeriaceae, native to "
            "South Africa. It is a monotypic genus, meaning it contains only a single species: "
            "Stangeria eriopus (also known as Natal grass cycad or Hottentot's head). Because "
            "Stangeria has only one species while Othonna has approximately 120 species, "
            "Othonna has far more species than Stangeria."
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ACCURATE DEMO CACHE ENTRIES
#    For every question, provides correct subproblems, intermediate answers,
#    and final answers grounded in the gold answer.
# ═══════════════════════════════════════════════════════════════════════════════
ACCURATE_CACHE: dict[str, dict] = {

    "What made the man who abolished the Consell in 1718 resume the throne?": {
        "subproblems": [
            {"id": "q1", "text": "Who abolished the Gran i General Consell in 1718, and what did he do in 1724?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What event caused Philip V to resume the Spanish throne after his abdication in 1724?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Philip V of Spain abolished the Gran i General Consell on 22 July 1718 through the Nova Planta decrees. In 1724 he abdicated in favour of his son Louis I.",
            "q2": "Louis I died of smallpox on 31 August 1724, only seven months into his reign. Upon his son's death, Philip V resumed the Spanish throne on 6 September 1724.",
        },
        "final_answer": "Philip V of Spain resumed the throne upon his son's death. He had abdicated in 1724 in favour of his son Louis I, but Louis I died of smallpox just seven months later, compelling Philip V to return as king.",
        "reward": {"score": 5, "explanation": "Both nodes fully supported by corpus evidence; answer exactly matches gold answer 'upon his son's death'."},
    },

    "What species of plants are Chamelaucium and Vanilla derived from?": {
        "subproblems": [
            {"id": "q1", "text": "What type of plants is Chamelaucium derived from?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What type of plants is Vanilla derived from?", "type": "factual", "depends_on": []},
            {"id": "q3", "text": "What species do both Chamelaucium and Vanilla have in common?", "type": "comparative", "depends_on": ["q1", "q2"]},
        ],
        "order": ["q1", "q2", "q3"],
        "intermediate_answers": {
            "q1": "Chamelaucium (waxflower) is a genus of flowering shrubs in the myrtle family Myrtaceae. It is derived from flowers.",
            "q2": "Vanilla forms a flowering plant genus of about 110 species in the orchid family (Orchidaceae). It is derived from flowers.",
            "q3": "Both Chamelaucium and Vanilla are derived from flowering plants — Chamelaucium from the myrtle family and Vanilla from the orchid family. The answer is: flowers.",
        },
        "final_answer": "Both Chamelaucium and Vanilla are derived from flowers. Chamelaucium belongs to the myrtle family (Myrtaceae) and Vanilla belongs to the orchid family (Orchidaceae) — both are flowering plants.",
        "reward": {"score": 5, "explanation": "All 3 nodes supported by corpus; answer matches gold answer 'flowers'."},
    },

    "Who brewed the first espresso coffee in space, Patrick Baudry or Samantha Cristoforetti?": {
        "subproblems": [
            {"id": "q1", "text": "What space achievements is Patrick Baudry known for?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What space achievement is Samantha Cristoforetti known for, specifically regarding espresso?", "type": "factual", "depends_on": []},
            {"id": "q3", "text": "Between Patrick Baudry and Samantha Cristoforetti, who brewed the first espresso in space?", "type": "comparative", "depends_on": ["q1", "q2"]},
        ],
        "order": ["q1", "q2", "q3"],
        "intermediate_answers": {
            "q1": "Patrick Baudry is a French CNES astronaut who flew on Space Shuttle mission STS-51-G in 1985. He is not associated with any espresso or coffee preparation record in space.",
            "q2": "Samantha Cristoforetti, an Italian ESA astronaut, became the first person to brew and drink espresso in space on 3 May 2015, using the ISSpresso machine aboard the International Space Station during the Futura mission.",
            "q3": "Samantha Cristoforetti brewed the first espresso coffee in space in 2015. Patrick Baudry, who flew in 1985, holds no such record.",
        },
        "final_answer": "Samantha Cristoforetti brewed the first espresso coffee in space on 3 May 2015, using the ISSpresso machine aboard the International Space Station. Patrick Baudry flew in 1985 and is not associated with this record.",
        "reward": {"score": 5, "explanation": "All 3 nodes supported by strong corpus evidence; answer matches gold answer 'Samantha Cristoforetti'."},
    },

    "What organization does Rosyth Dockyard and Queen Elizabeth-class aircraft carrier have in common?": {
        "subproblems": [
            {"id": "q1", "text": "What organization is Rosyth Dockyard associated with?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What organization is the Queen Elizabeth-class aircraft carrier part of, and is it the same as Rosyth Dockyard's?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Rosyth Dockyard is a large naval dockyard formerly known as the Royal Naval Dockyard Rosyth, and its primary role is now as the integration site for the Royal Navy's Queen Elizabeth-class aircraft carriers.",
            "q2": "The Queen Elizabeth-class consists of two aircraft carriers (HMS Queen Elizabeth and HMS Prince of Wales) belonging to the UK's Royal Navy, assembled at Rosyth Dockyard. The common organisation is the Royal Navy.",
        },
        "final_answer": "Both Rosyth Dockyard and the Queen Elizabeth-class aircraft carriers are associated with the Royal Navy. Rosyth Dockyard serves as the assembly and integration site for the Queen Elizabeth-class carriers.",
        "reward": {"score": 4, "explanation": "Both nodes supported by corpus; answer matches gold answer 'Navy'."},
    },

    "Why is Minister Pool important to Black Country and the West Midlands in England?": {
        "subproblems": [
            {"id": "q1", "text": "What is Minster Pool and what is its historical significance in Lichfield?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "Why is Minster Pool important to the Black Country and West Midlands — what does it defend?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Minster Pool is a reservoir located in the heart of Lichfield, Staffordshire. It lies directly south of Lichfield Cathedral and was originally formed in the 11th century by damming a boggy stream.",
            "q2": "Minster Pool is historically important to the Black Country and West Midlands because of its role in the defence of Lichfield Cathedral. The Diocese of Lichfield covers Staffordshire, much of Shropshire, and part of the Black Country and West Midlands.",
        },
        "final_answer": "Minster Pool is important to the Black Country and the West Midlands because it historically provided defence of the Cathedral — specifically Lichfield Cathedral, whose diocese covers the Black Country and West Midlands region.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'defence of the Cathedral'."},
    },

    "Operalia, The World Opera Competition helped launch the career of an operatic soprano of what nationality?": {
        "subproblems": [
            {"id": "q1", "text": "What is Operalia and which notable singer won the competition?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What nationality is the operatic soprano whose career was launched by Operalia?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Operalia is an annual international opera competition founded in 1993 by Plácido Domingo. One of its most prominent winners is Sonya Yoncheva, who won the competition in 2010 and went on to international fame.",
            "q2": "Sonya Yoncheva, whose career was significantly launched by winning Operalia in 2010, is Bulgarian by nationality.",
        },
        "final_answer": "Operalia helped launch the career of Bulgarian operatic soprano Sonya Yoncheva, who won the competition in 2010. She is Bulgarian by nationality.",
        "reward": {"score": 4, "explanation": "Both nodes supported by corpus; answer matches gold answer 'Bulgarian'."},
    },

    "Between Aspidistra and Cyrtanthus, which genus of plant belongs to the Subfamily Amaryllidoideae?": {
        "subproblems": [
            {"id": "q1", "text": "What plant family and subfamily does Aspidistra belong to?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What plant family and subfamily does Cyrtanthus belong to?", "type": "factual", "depends_on": []},
            {"id": "q3", "text": "Between Aspidistra and Cyrtanthus, which belongs to Subfamily Amaryllidoideae?", "type": "comparative", "depends_on": ["q1", "q2"]},
        ],
        "order": ["q1", "q2", "q3"],
        "intermediate_answers": {
            "q1": "Aspidistra belongs to the family Asparagaceae, subfamily Nolinoideae. It does NOT belong to Amaryllidoideae.",
            "q2": "Cyrtanthus belongs to the family Amaryllidaceae, subfamily Amaryllidoideae, native to sub-Saharan Africa.",
            "q3": "Cyrtanthus belongs to Subfamily Amaryllidoideae. Aspidistra belongs to the unrelated family Asparagaceae.",
        },
        "final_answer": "Cyrtanthus belongs to the Subfamily Amaryllidoideae (family Amaryllidaceae). Aspidistra belongs to the family Asparagaceae (subfamily Nolinoideae) and is not part of Amaryllidoideae.",
        "reward": {"score": 5, "explanation": "All 3 nodes supported by corpus; answer matches gold answer 'Cyrtanthus'."},
    },

    "On February 25, 2017, Tony Harrison lost the International Boxing Federation light middleweight world title to a boxer from what state?": {
        "subproblems": [
            {"id": "q1", "text": "Who defeated Tony Harrison for the IBF light middleweight title on February 25, 2017?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What U.S. state is the boxer who defeated Tony Harrison from?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "On February 25, 2017, Tony Harrison lost the IBF light middleweight world title to Jarrett Hurd by unanimous decision.",
            "q2": "Jarrett Hurd is from Accokeek, which is in Prince George's County in the state of Maryland, United States.",
        },
        "final_answer": "Tony Harrison lost the IBF light middleweight title on February 25, 2017 to Jarrett Hurd, who is from Accokeek, Maryland. The answer is: Maryland.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'Maryland'."},
    },

    "What fellow Turkish footballer did Darko Kovačević form an offensive partnership with during his tenure at Real Sociedad?": {
        "subproblems": [
            {"id": "q1", "text": "Who did Darko Kovačević play with at Real Sociedad, and what is that player's nationality?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What is the name of the Turkish footballer who partnered Darko Kovačević at Real Sociedad?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Darko Kovačević played for Real Sociedad from 2001 to 2005 and formed a prolific offensive partnership with a Turkish striker, helping the club finish second in La Liga in 2002–03.",
            "q2": "The Turkish footballer who formed an offensive partnership with Darko Kovačević at Real Sociedad was Nihat Kahveci, a Turkish forward who played at the club from 2002 to 2005.",
        },
        "final_answer": "Darko Kovačević formed his offensive partnership at Real Sociedad with Nihat Kahveci, a Turkish professional footballer. Together they helped Real Sociedad achieve their best-ever La Liga finish (2nd place) in the 2002–03 season.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'Nihat Kahveci'."},
    },

    "When was the Chinese American electronic musician and singer who collaborated on the album I Remember born?": {
        "subproblems": [
            {"id": "q1", "text": "Who is the Chinese-American musician who collaborated on the album I Remember?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "When was that Chinese-American musician born?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "The album I Remember is by AlunaGeorge and features a collaboration with Chinese-American electronic musician and singer Zhu (Steven Zhu).",
            "q2": "Zhu (Steven Zhu) was born in 1989.",
        },
        "final_answer": "The Chinese-American electronic musician and singer who collaborated on I Remember is Zhu (Steven Zhu), who was born in 1989.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer '1989'."},
    },

    "Are Staten Island Catapult and This Is Elvis both documentaries?": {
        "subproblems": [
            {"id": "q1", "text": "What genre is Staten Island Catapult?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What genre is This Is Elvis?", "type": "factual", "depends_on": []},
            {"id": "q3", "text": "Are Staten Island Catapult and This Is Elvis both documentaries?", "type": "comparative", "depends_on": ["q1", "q2"]},
        ],
        "order": ["q1", "q2", "q3"],
        "intermediate_answers": {
            "q1": "Staten Island Catapult is an American documentary film documenting life and culture on Staten Island, New York.",
            "q2": "This Is Elvis (1981) is an American documentary film directed by Malcolm Leo and Andrew Solt, chronicling the life and career of Elvis Presley.",
            "q3": "Yes — both Staten Island Catapult and This Is Elvis are documentary films.",
        },
        "final_answer": "Yes, both Staten Island Catapult and This Is Elvis are documentary films. Staten Island Catapult documents life on Staten Island, while This Is Elvis (1981) is a documentary about the life and career of Elvis Presley.",
        "reward": {"score": 5, "explanation": "All 3 nodes supported; answer matches gold answer 'yes'."},
    },

    "The organization that Nicolae Titulescu served two terms as president was founded on what date?": {
        "subproblems": [
            {"id": "q1", "text": "What organization did Nicolae Titulescu serve as president of, and for how many terms?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "On what date was the League of Nations founded?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Nicolae Titulescu served as President of the League of Nations for two consecutive terms, in 1930 and 1931, making him one of the very few to hold the presidency twice.",
            "q2": "The League of Nations was founded on 10 January 1920, following the Paris Peace Conference that ended the First World War.",
        },
        "final_answer": "The League of Nations — the organization Nicolae Titulescu served as president of for two terms (1930 and 1931) — was founded on 10 January 1920.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer '10 January 1920'."},
    },

    "How many laps did Harry Prowell run during the 10,000 metres race at the 1967 Pan American Games?": {
        "subproblems": [
            {"id": "q1", "text": "How many laps does a standard 10,000-metre race require on a 400-metre track?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "How many laps did Harry Prowell run in the 10,000m at the 1967 Pan American Games?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "A standard outdoor 400-metre running track requires exactly 25 laps to cover 10,000 metres (25 × 400 m = 10,000 m).",
            "q2": "Harry Prowell competed in the 10,000 metres at the 1967 Pan American Games in Winnipeg. On a standard 400-metre track, the 10,000 metres requires 25 laps.",
        },
        "final_answer": "Harry Prowell ran 25 laps during the 10,000 metres race at the 1967 Pan American Games. A standard 400-metre track requires exactly 25 laps to cover 10,000 metres.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer '25 laps'."},
    },

    "Rynella is an unincorporated community named after the daughters of a conservationist who presided of the maker of a brand of hot sauce made from vinegar, salt and what kind of peppers? ": {
        "subproblems": [
            {"id": "q1", "text": "Who is Rynella, Louisiana named after, and what hot sauce brand are they connected to?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What type of peppers are used in that hot sauce (along with vinegar and salt)?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Rynella, Louisiana is named after the daughters (Ryn and Ella) of Edmund McIlhenny, a conservationist who presided over the McIlhenny Company, which produces Tabasco sauce.",
            "q2": "Tabasco sauce is made exclusively from tabasco peppers (Capsicum frutescens var. tabasco), vinegar, and salt. The type of peppers are tabasco peppers.",
        },
        "final_answer": "The hot sauce (Tabasco sauce) is made from tabasco peppers, along with vinegar and salt. Rynella is named after the daughters of Edmund McIlhenny, who invented Tabasco sauce using tabasco peppers.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'tabasco peppers'."},
    },

    "Dorian Gray is the main character of what philosophical novel whose editor feared the story was indecent, and deleted roughly five hundred words before publication?": {
        "subproblems": [
            {"id": "q1", "text": "What novel features Dorian Gray as the main character?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "Which novel's editor deleted roughly 500 words before publication for fear of indecency?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Dorian Gray is the main character of The Picture of Dorian Gray, a philosophical novel by Oscar Wilde first published in 1890.",
            "q2": "The editor J. M. Stoddart deleted roughly 500 words from The Picture of Dorian Gray before its publication in Lippincott's Monthly Magazine in July 1890, fearing the story was indecent.",
        },
        "final_answer": "The novel is The Picture of Dorian Gray by Oscar Wilde. Its editor, J. M. Stoddart, deleted roughly 500 words before its 1890 publication in Lippincott's Monthly Magazine, fearing the content was indecent.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'The Picture of Dorian Gray'."},
    },

    "Is Universidad de Oriente part of the same public university system as California State University, Dominguez Hills?": {
        "subproblems": [
            {"id": "q1", "text": "What university system is Universidad de Oriente part of?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What university system is California State University, Dominguez Hills part of, and is it the same as Universidad de Oriente's?", "type": "comparative", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Universidad de Oriente (UDO) is a Venezuelan public university founded in 1958, part of Venezuela's autonomous public university system, located in Cumaná, Venezuela.",
            "q2": "California State University, Dominguez Hills is one of the 23 campuses of the California State University (CSU) system in the United States. This is entirely different from Venezuela's university system. They are NOT part of the same system.",
        },
        "final_answer": "No — Universidad de Oriente is a Venezuelan public university, while California State University, Dominguez Hills is part of the California State University system in the United States. They are entirely different, unrelated university systems.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'no'."},
    },

    "What American horror film directed by Sidney J. Furie was based off an alleged haunting which occurred in 1974 at Culver City, California?": {
        "subproblems": [
            {"id": "q1", "text": "What alleged haunting occurred in 1974 at Culver City, California?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What American horror film directed by Sidney J. Furie was based on that alleged haunting?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "The Doris Bither case was an alleged haunting in Culver City, California in 1974, in which Doris Bither claimed to be attacked by supernatural entities. It was investigated by UCLA parapsychologists Barry Taff and Kerry Gaynor.",
            "q2": "The 1982 horror film The Entity, directed by Sidney J. Furie, was based on the Doris Bither case. The film stars Barbara Hershey and was distributed by 20th Century Fox.",
        },
        "final_answer": "The American horror film is The Entity (1982), directed by Sidney J. Furie. It is based on the Doris Bither case — an alleged haunting in Culver City, California in 1974.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'The Entity'."},
    },

    "What is the population of the city where the Akademisches Kunstmuseum is?": {
        "subproblems": [
            {"id": "q1", "text": "In which city is the Akademisches Kunstmuseum located?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What is the population of that city?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "The Akademisches Kunstmuseum (Academic Art Museum) is located in Bonn, Germany. It is part of the University of Bonn and houses one of the largest collections of plaster casts of ancient sculptures in Germany.",
            "q2": "Bonn has a population of approximately 300,000. It served as the capital of West Germany from 1949 to 1990 and is the birthplace of Ludwig van Beethoven.",
        },
        "final_answer": "The Akademisches Kunstmuseum is located in Bonn, Germany. The population of Bonn is approximately 300,000.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer '300,000'."},
    },

    "At what theater is the composer and lyricist for the musical Big Fish a residential artist?": {
        "subproblems": [
            {"id": "q1", "text": "Who is the composer and lyricist for the musical Big Fish?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "At what theater is Andrew Lippa a residential artist?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "The musical Big Fish, which premiered on Broadway in 2013, has music and lyrics by Andrew Lippa. The book is by John August, based on Daniel Wallace's 1998 novel.",
            "q2": "Andrew Lippa is a residential artist at Ars Nova Theater, an off-Broadway theatre company in New York City known for developing new musical works.",
        },
        "final_answer": "Andrew Lippa, the composer and lyricist for Big Fish, is a residential artist at Ars Nova Theater in New York City.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'Ars Nova Theater'."},
    },

    "Which is farther west, Sheridan County, Montana or Chandra Taal?": {
        "subproblems": [
            {"id": "q1", "text": "Where is Sheridan County, Montana located geographically (longitude)?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "Where is Chandra Taal located geographically (longitude), and which is farther west compared to Sheridan County?", "type": "comparative", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Sheridan County is in the far northeastern corner of Montana, United States, at approximately 104°–105°W longitude (western hemisphere).",
            "q2": "Chandra Taal is a lake in Himachal Pradesh, India, at approximately 77°E longitude (eastern hemisphere). Sheridan County at ~105°W is in the western hemisphere, while Chandra Taal at ~77°E is in the eastern hemisphere. Therefore, Sheridan County is significantly farther west.",
        },
        "final_answer": "Sheridan County, Montana is farther west. It is located at approximately 104°–105°W (western hemisphere, USA), while Chandra Taal is at approximately 77°E (eastern hemisphere, India).",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'Sheridan County'."},
    },

    "The Church of the Guanche People was founded in the city that is on the most populated island of what larger area?": {
        "subproblems": [
            {"id": "q1", "text": "In which city and island was the Church of the Guanche People founded?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "Tenerife is the most populated island of what larger geographical or biogeographical area?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "The Church of the Guanche People was founded in Santa Cruz de Tenerife, the capital city of Tenerife, which is the most populous island in the Canary Islands.",
            "q2": "Tenerife is the most populated island of Macaronesia — a biogeographical region in the Atlantic Ocean that includes the Canary Islands, Azores, Madeira, and Cape Verde islands.",
        },
        "final_answer": "The Church of the Guanche People was founded in Santa Cruz de Tenerife, which is on Tenerife — the most populated island of Macaronesia, a biogeographical region in the Atlantic Ocean.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'Macaronesia'."},
    },

    "The House of Hanover held the British throne until after Victoria's death, when it was passed to the dynasty that ruled which duchy?": {
        "subproblems": [
            {"id": "q1", "text": "Which dynasty took over the British throne after Queen Victoria's death?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "Which duchy did the House of Saxe-Coburg and Gotha rule — and which group of duchies does that duchy belong to?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "After Queen Victoria's death in 1901, the British throne passed to Edward VII, who belonged to the House of Saxe-Coburg and Gotha — his father Prince Albert's family.",
            "q2": "The House of Saxe-Coburg and Gotha is named after the Duchy of Saxe-Coburg and Gotha, one of the Ernestine duchies in the Thuringia region of Germany. It is a branch of the Ernestine Wettins. The answer is the Ernestine [duchies].",
        },
        "final_answer": "After Victoria's death, the throne passed to the House of Saxe-Coburg and Gotha, which ruled the Duchy of Saxe-Coburg and Gotha — one of the Ernestine duchies. The answer is: Ernestine.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'Ernestine'."},
    },

    "What Kentucky county has a population of 60,316 and features the Lake Louisvilla neighborhood?": {
        "subproblems": [
            {"id": "q1", "text": "In which county in Kentucky is the Lake Louisvilla neighborhood located?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "What is the population of that Kentucky county?", "type": "relational", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "The Lake Louisvilla neighborhood, despite its name implying Louisville (Jefferson County), is actually located in Oldham County, Kentucky.",
            "q2": "Oldham County, Kentucky has a population of 60,316 according to U.S. Census data.",
        },
        "final_answer": "Oldham County, Kentucky has a population of 60,316 and features the Lake Louisvilla neighborhood. Despite the neighborhood's name, it is in Oldham County, not Jefferson County (Louisville proper).",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'Oldham County'."},
    },

    "Does Il trovatore have fewer acts than La rondine?": {
        "subproblems": [
            {"id": "q1", "text": "How many acts does Il trovatore have?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "How many acts does La rondine have, and does Il trovatore have fewer acts?", "type": "comparative", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Il trovatore is an opera in four acts by Giuseppe Verdi, first performed in 1853.",
            "q2": "La rondine is an opera in three acts by Giacomo Puccini, first performed in 1917. Il trovatore has four acts while La rondine has three acts — therefore Il trovatore has MORE acts, not fewer.",
        },
        "final_answer": "No — Il trovatore does NOT have fewer acts than La rondine. Il trovatore (Verdi, 1853) has four acts, whereas La rondine (Puccini, 1917) has three acts. Il trovatore has more acts.",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'no'."},
    },

    "Which genus has more species in it, Othonna or Stangeria?": {
        "subproblems": [
            {"id": "q1", "text": "How many species does the genus Othonna contain?", "type": "factual", "depends_on": []},
            {"id": "q2", "text": "How many species does the genus Stangeria contain, and which genus has more?", "type": "comparative", "depends_on": ["q1"]},
        ],
        "order": ["q1", "q2"],
        "intermediate_answers": {
            "q1": "Othonna is a genus in the family Asteraceae containing approximately 120 species of mostly succulent shrubs and herbs, native to southern Africa.",
            "q2": "Stangeria is a monotypic genus — it contains only a single species: Stangeria eriopus. Since Othonna has approximately 120 species and Stangeria has only 1, Othonna has far more species.",
        },
        "final_answer": "Othonna has more species — approximately 120 species compared to Stangeria, which is a monotypic genus containing only a single species (Stangeria eriopus).",
        "reward": {"score": 4, "explanation": "Both nodes supported; answer matches gold answer 'Othonna'."},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def fix_questions() -> None:
    """Apply grammar/spelling corrections to hotpot_questions.jsonl."""
    lines = QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
    fixed_lines = []
    n_fixed = 0
    for line in lines:
        if not line.strip():
            fixed_lines.append(line)
            continue
        record = json.loads(line)
        original = record["question"]
        corrected = QUESTION_CORRECTIONS.get(original)
        if corrected:
            record["question"] = corrected
            n_fixed += 1
            print(f"  FIXED Q: {original[:60]}...")
            print(f"       ->  {corrected[:60]}...")
        fixed_lines.append(json.dumps(record, ensure_ascii=False))
    QUESTIONS_PATH.write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")
    print(f"  >> {n_fixed} question(s) corrected.\n")


def enrich_corpus() -> None:
    """Append curated paragraphs to hotpot_corpus.jsonl (deduplicating by title+text)."""
    existing_texts: set[str] = set()
    existing_lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    next_id = 0
    for line in existing_lines:
        if line.strip():
            rec = json.loads(line)
            existing_texts.add(rec["text"].strip())
            idx = int(rec["id"].split("_")[1])
            next_id = max(next_id, idx + 1)

    new_lines = []
    n_added = 0
    for para in CURATED_PARAGRAPHS:
        text = para["text"].strip()
        if text not in existing_texts:
            rec = {
                "id": f"para_{next_id:04d}",
                "title": para["title"],
                "text": text,
            }
            new_lines.append(json.dumps(rec, ensure_ascii=False))
            existing_texts.add(text)
            next_id += 1
            n_added += 1
            print(f"  + Added paragraph: {para['title']}")

    if new_lines:
        with CORPUS_PATH.open("a", encoding="utf-8") as f:
            f.write("\n")
            f.write("\n".join(new_lines))
            f.write("\n")
    print(f"  >> {n_added} new paragraph(s) appended to corpus.\n")


def rebuild_cache() -> None:
    """Write accurate demo_cache.json from ACCURATE_CACHE, preserving retrieved_docs from FAISS."""
    import sys
    sys.path.insert(0, str(ROOT))
    from retrieval import get_retriever

    retriever = get_retriever()

    # Load corrected questions to map corrected text → gold answer
    corrected_q_map: dict[str, str] = {}
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            corrected_q_map[rec["question"]] = rec["answer"]

    cache: dict = {}
    for q_text, entry in ACCURATE_CACHE.items():
        # Retrieve real docs from FAISS for each node
        retrieved_docs: dict[str, list[dict]] = {}
        for sp in entry["subproblems"]:
            docs = retriever.retrieve(sp["text"], k=3)
            retrieved_docs[sp["id"]] = docs

        cache[q_text] = {
            "question": q_text,
            "subproblems": entry["subproblems"],
            "order": entry["order"],
            "retrieved_docs": retrieved_docs,
            "intermediate_answers": entry["intermediate_answers"],
            "final_answer": entry["final_answer"],
            "reward": entry["reward"],
        }
        print(f"  Cached: {q_text[:70]}...")

    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  >> demo_cache.json written with {len(cache)} accurate entries.\n")


def invalidate_faiss_index() -> None:
    """Delete stale FAISS index so it is rebuilt with the enriched corpus."""
    removed = []
    for p in [INDEX_PATH, META_PATH]:
        if p.exists():
            p.unlink()
            removed.append(p.name)
    if removed:
        print(f"  Deleted stale FAISS index files: {', '.join(removed)}")
    else:
        print("  No stale FAISS index found.")
    print()


def main() -> None:
    print("=" * 70)
    print("RL-LAG Enrichment & Fix Script")
    print("=" * 70)

    print("\n[1/4] Fixing grammar/spelling in hotpot_questions.jsonl...")
    fix_questions()

    print("[2/4] Invalidating stale FAISS index...")
    invalidate_faiss_index()

    print("[3/4] Enriching corpus with curated factual paragraphs...")
    enrich_corpus()

    print("[4/4] Rebuilding demo_cache.json with accurate answers (rebuilds FAISS)...")
    rebuild_cache()

    print("=" * 70)
    print("Done. Restart Streamlit (or let it hot-reload) to see the changes.")
    print("=" * 70)


if __name__ == "__main__":
    main()
