import spacy
from spacy.matcher import Matcher, PhraseMatcher
import json
import random
from pathlib import Path


FANTASY_LABELS = ["PERSONAJE", "FACCION", "LUGAR", "ARTEFACTO_MAGICO", "RAZA"]

FANTASY_ENTITIES = {
    "PERSONAJE": [
        "Gandalf", "Aragorn", "Legolas", "Gimli", "Frodo", "Sauron", "Saruman",
        "Galadriel", "Elrond", "Arwen", "Boromir", "Faramir", "Theoden", "Eomer",
        "Eowyn", "Gollum", "Bilbo", "Sam", "Merry", "Pippin", "Treebeard",
        "Radagast", "Glorfindel", "Celeborn", "Thranduil", "Tauriel", "Azog",
        "Bolgo", "Gothmog", "Witch-King", "Denethor", "Imrahil", "Beregond",
        "Grishnakh", "Shagrat", "Gorbag", "Ugluk", "Lurtz", "Haldir",
        "Elendil", "Isildur", "Anarion", "Gil-galad", "Finrod", "Beren",
        "Luthien", "Turin", "Earendil", "Maedhros", "Fingolfin", "Thingol",
        "Melian", "Curunir", "Grima", "Wormtongue", "Bill Ferny", "Tom Bombadil",
        "Goldberry", "Barliman", "Butterbur", "Rosie Cotton", "Lobelia",
    ],
    "LUGAR": [
        "Gondor", "Mordor", "Rivendel", "Rohan", "la Comarca", "Minas Tirith",
        "Isengard", "Moria", "Khazad-dum", "Lothlorien", "Dol Guldur",
        "Osgiliath", "Helm", "Abismo de Helm", "Edoras", "Bree",
        "Mount Doom", "Orodruin", "Fangorn", "Mirkwood", "Bosque Negro",
        "Valinor", "Aman", "Numenor", "Atlantis", "Harad", "Rhun",
        "Eriador", "Arnor", "Dale", "Lago Largo", "Montanas Nubladas",
        "Pass of Caradhras", "Cirith Ungol", "Minas Morgul", "Pelennor",
        "Anduin", "Gran Rio", "Puerto de los Cisnes", "Alqualonde",
        "Tirion", "Gondolin", "Nargothrond", "Doriath", "Beleriand",
        "Tierras Imperecederas", "Tierras Medias", "Shire", "Hobbiton",
        "Crickhollow", "Buckland", "Toma", "Archet", "Combe",
    ],
    "RAZA": [
        "Elfo", "Enano", "Hobbit", "Humano", "Orco", "Uruk-hai", "Mayar",
        "Valar", "Ent", "Troll", "Balrog", "Nazgul", "Espectro del Anillo",
        "Medio-Elfo", "Dunedain", "Rohirrim", "Hombre del Norte",
        "Haradrim", "Easterling", "Druida", "Hechicero", "Mago",
        "Elfo Oscuro", "Elfo Gris", "Alto Elfo", "Elfo Silvano",
        "Enano de Hierro", "Enano de Montaña", "Orco de Montaña",
        "Goblin", "Huargo", "Dragon", "Ents", "Hombres de Dunland",
    ],
    "ARTEFACTO_MAGICO": [
        "Anduril", "Glamdring", "el Anillo Unico", "Narya", "Nenya", "Vilya",
        "Mithril", "Palantir", "Orthanc", "Bastion de Orthanc",
        "la Llama de Anor", "Furia de Elendil", "Espada de Sauron",
        "Arco de Galadriel", "Capa Elfica", "Frasco de Galadriel",
        "Phial de Galadriel", "Bastion de Gandalf", "Gandalfs Staff",
        "Orb de Saruman", "Corona de Gondor", "Sello de Numenor",
        "Anillo de Barahir", "Cuerno de Gondor", "Cuerno de Rohan",
        "Arco de los Galadhrim", "Daga de los Hobbits", "Sting",
        "Orcrist", "Anglachel", "Gurthang", "Aeglos", "Dramborleg",
        "Silmaril", "Arpa de Daeron", "Martillo de Aule",
    ],
    "FACCION": [
        "la Comunidad del Anillo", "los Nazgul", "la Guardia Blanca",
        "los Rohirrim", "la Compania del Anillo", "el Consejo Elrond",
        "los Elfos de Rivendel", "los Enanos de Erebor", "los Hobbits de la Comarca",
        "los Orcos de Mordor", "los Uruk-hai de Isengard", "la Mano Blanca",
        "el Ojo de Sauron", "los Dunledinos", "los Corsarios de Umbar",
        "la Guardia de la Ciudadela", "los Caballeros de Dol Amroth",
        "los Arqueros de Ithilien", "la Orden de los Magos",
        "los Istari", "los Valar", "los Maiar", "la Casa de Fingolfin",
        "la Casa de Feanor", "los Noldor", "los Sindar", "los Teleri",
        "los Vanyar", "la Hermandad del Lobulo Rojo",
        "los Jinetes de Rohan", "los Escudos de Gondor",
    ],
}

SENTENCE_TEMPLATES = [
    "{PERSONAJE} viajó a {LUGAR} portando {ARTEFACTO_MAGICO}.",
    "El {RAZA} {PERSONAJE} lideró a {FACCION} en la batalla de {LUGAR}.",
    "{PERSONAJE} encontró {ARTEFACTO_MAGICO} en las ruinas de {LUGAR}.",
    "Los {RAZA} de {FACCION} marcharon hacia {LUGAR} bajo el mando de {PERSONAJE}.",
    "{PERSONAJE}, un {RAZA} de {LUGAR}, blandía {ARTEFACTO_MAGICO} con destreza.",
    "En {LUGAR}, {PERSONAJE} se unió a {FACCION} para luchar contra los {RAZA}.",
    "{ARTEFACTO_MAGICO} fue forjado por los {RAZA} en {LUGAR} para {PERSONAJE}.",
    "{FACCION} defendió {LUGAR} contra el ataque de los {RAZA} liderados por {PERSONAJE}.",
    "El {RAZA} {PERSONAJE} cruzó {LUGAR} en busca de {ARTEFACTO_MAGICO}.",
    "{PERSONAJE} entregó {ARTEFACTO_MAGICO} a {FACCION} en {LUGAR}.",
    "Los espías de {FACCION} informaron que {PERSONAJE} se dirigía a {LUGAR}.",
    "{PERSONAJE} del linaje de los {RAZA} gobernó {LUGAR} durante siglos.",
    "En las profundidades de {LUGAR}, {PERSONAJE} halló {ARTEFACTO_MAGICO} custodiado por {RAZA}.",
    "{FACCION} y {PERSONAJE} sellaron un pacto en {LUGAR} usando {ARTEFACTO_MAGICO}.",
    "El {ARTEFACTO_MAGICO} brilló cuando {PERSONAJE} lo alzó frente a los {RAZA} en {LUGAR}.",
    "{PERSONAJE} nació en {LUGAR} y se unió a {FACCION} siendo joven {RAZA}.",
    "Las crónicas de {LUGAR} narran cómo {PERSONAJE} destruyó {ARTEFACTO_MAGICO} traicionado por {RAZA}.",
    "{FACCION} envió a {PERSONAJE} a explorar {LUGAR} en busca de aliados {RAZA}.",
    "El trono de {LUGAR} fue ocupado por {PERSONAJE}, último {RAZA} de su linaje.",
    "{PERSONAJE} forjó {ARTEFACTO_MAGICO} con la ayuda de {FACCION} en {LUGAR}.",
]

COMPLEX_SENTENCES = [
    "Tras días de marcha, {PERSONAJE} y su compañia de {RAZA} llegaron a las puertas de {LUGAR}, donde {FACCION} los esperaba con {ARTEFACTO_MAGICO} en mano.",
    "Las leyendas de {LUGAR} cuentan que {PERSONAJE}, el más sabio de los {RAZA}, escondió {ARTEFACTO_MAGICO} para que {FACCION} nunca lo encontrara.",
    "Cuando {PERSONAJE} alzó {ARTEFACTO_MAGICO} sobre las murallas de {LUGAR}, los {RAZA} de {FACCION} prorrumpieron en vítores de guerra.",
    "El consejo de {FACCION} se reunió en {LUGAR} para decidir el destino de {ARTEFACTO_MAGICO}, con {PERSONAJE} como único {RAZA} en votar en contra.",
    "{PERSONAJE} juró ante {ARTEFACTO_MAGICO} que defendería {LUGAR} hasta el último aliento, cumpliendo la promesa hecha a {FACCION} hace eras.",
]

EXTRA_TEMPLATES = [
    "{PERSONAJE} era conocido en {LUGAR} por su valentia como {RAZA} de {FACCION}.",
    "Los archivos de {LUGAR} mencionan a {PERSONAJE} como el portador de {ARTEFACTO_MAGICO}.",
    "{FACCION} protegia {LUGAR} con {ARTEFACTO_MAGICO} bajo el liderazgo de {PERSONAJE}.",
    "El {RAZA} {PERSONAJE} fue exiliado de {LUGAR} y se unio a {FACCION} portando {ARTEFACTO_MAGICO}.",
    "En {LUGAR}, los {RAZA} de {FACCION} celebraron la victoria de {PERSONAJE} con {ARTEFACTO_MAGICO}.",
]


def build_matcher(nlp):
    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    for label, entities in FANTASY_ENTITIES.items():
        patterns = []
        for entity in entities:
            doc = nlp(entity)
            patterns.append(doc)
        phrase_matcher.add(label, patterns)

    return phrase_matcher


def weak_label_text(text, nlp, phrase_matcher):
    doc = nlp(text)
    matches = phrase_matcher(doc)
    entities = []
    seen = set()

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        key = (span.start_char, span.end_char)
        if key not in seen:
            seen.add(key)
            entities.append((span.start_char, span.end_char, label))

    entities.sort(key=lambda x: x[0])
    return doc, entities


def generate_synthetic_texts(count=500):
    texts = []
    for _ in range(count):
        if random.random() < 0.15:
            template = random.choice(COMPLEX_SENTENCES)
        else:
            template = random.choice(SENTENCE_TEMPLATES)

        text = template
        for label in ["PERSONAJE", "LUGAR", "RAZA", "ARTEFACTO_MAGICO", "FACCION"]:
            if "{" + label + "}" in text:
                text = text.replace("{" + label + "}", random.choice(FANTASY_ENTITIES[label]), 1)
        texts.append(text)

    for _ in range(count // 5):
        template = random.choice(EXTRA_TEMPLATES)
        text = template
        for label in ["PERSONAJE", "LUGAR", "RAZA", "ARTEFACTO_MAGICO", "FACCION"]:
            if "{" + label + "}" in text:
                text = text.replace("{" + label + "}", random.choice(FANTASY_ENTITIES[label]), 1)
        texts.append(text)

    return texts


def create_dataset_from_texts(texts, output_path="data/annotations/weak_labeled.json"):
    nlp = spacy.load("en_core_web_sm")
    phrase_matcher = build_matcher(nlp)

    dataset = []
    skipped = 0
    for text in texts:
        doc, entities = weak_label_text(text, nlp, phrase_matcher)
        if entities:
            dataset.append({
                "text": text,
                "entities": entities,
            })
        else:
            skipped += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    label_counts = {}
    for item in dataset:
        for _, _, label in item["entities"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"Dataset creado: {len(dataset)} textos etiquetados -> {output_path}")
    print(f"Textos sin entidades detectadas: {skipped}")
    print(f"\nDistribucion por etiqueta:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    return dataset


if __name__ == "__main__":
    random.seed(42)
    print("Generando textos sinteticos de fantasia...")
    synthetic_texts = generate_synthetic_texts(800)
    print(f"Total de textos generados: {len(synthetic_texts)}\n")

    print("Aplicando weak labeling con spaCy...")
    create_dataset_from_texts(synthetic_texts)
