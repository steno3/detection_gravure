# Detection Gravure - Prediction

Script de prédiction pour détecter les gravures sur des images.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Prédiction de base
python src/predict_image.py <image_path> <model_path.h5>

# Avec un masque
python src/predict_image.py <image_path> <model_path.h5> --mask-path <mask_path>

# Tous les paramètres disponibles
python src/predict_image.py <image_path> <model_path.h5> \
    --stride 256 \
    --patch-size 512 \
    --norm standard \
    --output-dir out \
    --mask-threshold 127 \
    --mask-path <mask_path>
```

### Arguments

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--stride` | 256 | Pas pour le découpage en patchs |
| `--patch-size` | 512 | Taille des patchs |
| `--norm` | `standard` | Normalisation: `standard`, `simple`, ou `none` |
| `--output-dir` | `out` | Dossier de sortie |
| `--mask-threshold` | 127 | Seuil pour le masque binaire |
| `--mask-path` | - | Chemin vers le masque (optionnel) |

### Types de normalisation

- `standard`: `(x - 127.5) / 127.5`
- `simple`: `x / 255.0`
- `none`: aucune normalisation

## Exemple

```bash
python src/predict_image.py image.png weights/dbos_50ep.h5 --output-dir results
```
