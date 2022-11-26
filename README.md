# Project about plants analysis through Computer Vision

- [Environment](#environment)
- [Directories](#directories)
- [Data](#data)

## Environment

We use [poetry](https://python-poetry.org/) version at least 1.2.0 here. To initialize project call `poetry install`. The command creates local virtualenv and installs all required python frameworks.

## Directories

- [notebooks](notebooks) - Jupyter notebooks are kept here
- [plants_image_analysis](plants_image_analysis) - Main sources are here. After poetry initialization you cal write `import plants_image_analysis` from notebooks or scripts
- [scripts](scripts) - Main executable scripts. Don't forget `if __name__ == "__main__"` protection there!
- [data](data) - Store main data samples here. For large files (over 10KB) use [Git LFS](https://git-lfs.github.com/)

## Data

- Kaggle competition [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)
  - Dataset is stored on [YaDisk](https://disk.yandex.ru/d/K4qcbWzlCII6jA)
  