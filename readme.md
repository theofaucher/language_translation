# API & Site web de traduction

Projet réalisé dans le cadre de mes études

## I - Modules utilisés:

> ### Pour l'intelligence artificiele
>
> Tensorflow@2.0

> ### Pour l'API
>
> Flask

> ### Pour le site web
>
> Bootstrap - JQuerry

## II - Données d'exemple

http://www.manythings.org/anki/

## III - L'inteligence artificielle

### 1 - Modèle:

![Réseau de neurone récurrent à encoder et decoder GRU avec un attention layer](attention_mechanism.jpg)

Réseau de neurone récurrent à encoder et decoder GRU avec un attention layer

### 2 - Utilisation (Entraînement et test):

```
usage: main.py [-h] [--mode MODE] [--config-name]
               [--batch-size INT] [--epoch INT] [--embedding-dim INT]
               [--units INT] [--sentences-size]

optional arguments:
  -h, --help            Montre ceci
  --mode MODE           Train ou test
  --config-name FILE Chemin pour enregistrer/charger les checkpoints et le config.json
  --batch-size INT      Taille des batchs <default: 32>
  --epoch INT           Nombre epoch <default: 10>
  --embedding-dim INT   Dimension de l'embedding <default: 256>
  --units INT           units <default: 512>
  --sentences-size INT  nombre de phrases prises dans la dataset<default: 60000>
```

### 4 - Exemple:

```
python main.py --epoch 20 --sentences-size 60000 gemelestest
...(Apprentissage)

python main.py --mode test gemelestest

Ecrivez une phrase (Entrer pour quitter):  : What is your name?
<start> What is your name ? <end>
<start> Quelle est ton prénom <end>

Ecrivez une phrase (Entrer pour quitter):  : What is your name?
<start> My name is Théo <end>
<start> Mon nom est <unk> <end>
```

## API:

> ### Requête type
>
> `<URL>:5000/translate/?source=fr&destination=en&original=Bonjour`

> ### Résultat type
>
> `Hello`

  ### Usage
  ```
usage: api.py [-h]  [--config-fren] [--config-enfr]

optional arguments:
  -h, --help            Montre ceci
  --config-enfr FILE Chemin pour charger les checkpoints et le config.json pour la traduction de Français vers Anglais
  --config-fren FILE Chemin pour charger les checkpoints et le config.json pour la traduction de Anglais vers Français
```