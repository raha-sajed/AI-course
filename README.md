# AI Course

Homework 1 and three mini projects from an introductory artificial intelligence course, being revised for portfolio presentation.

**Original group work:** Raha Sajed (credited as Fatemeh Sajed in the course reports) and Aylar Rahimzadeh.
**Post-course revisions:** Raha Sajed (`raha-sajed`).

## Projects

| Part | Topics | Assignment | Notebooks | Status |
| --- | --- | --- | --- | --- |
| [Homework 1](HomeWork1/) | Linear algebra, signal processing and Iris | [Questions](HomeWork1/AI4032_HW1.pdf) | [Revised signal analysis](HomeWork1/SignalAnalysis.ipynb); [original notebook](HomeWork1/HW1HOOSH.ipynb) | Signal analysis runs; algebra and Iris pending |
| [Mini Project 1](MiniProject1/) | Titanic classification, regression and outliers | [Questions](<MiniProject1/AI4032_MP1 (1).pdf>) | [Titanic](MiniProject1/MiniProject1.ipynb); regression [1](MiniProject1/MiniProject1.2.ipynb), [2](MiniProject1/MiniProject1.3.ipynb), [3](MiniProject1/MiniProject1.4.ipynb) | Corrections and fresh execution pending |
| [Mini Project 2](MiniProject2/) | Simple neurons, weather forecasting and Hopfield networks | [Questions](MiniProject2/AI4032_MP2.pdf) | [Neurons](MiniProject2/Ai_MiniProject2_Q1.ipynb); image experiments [1](MiniProject2/Ai_MiniProject2_Q3_1.ipynb), [2](MiniProject2/Ai_MiniProject2_Q3_2.ipynb), [3](MiniProject2/Ai_MiniProject2_Q3_3.ipynb) | Weather data available; implementation and validation pending |
| [Mini Project 3](MiniProject3/) | Fuzzy systems, robot control and ANFIS | [Questions](MiniProject3/AI4032_MP3.pdf) | [Notebook](MiniProject3/MP3.ipynb) | Scientific and implementation corrections pending |

Each project folder contains its code and data. New outputs are saved in that project's `results` folder. Original submitted reports are preserved locally and excluded from future Git versions.

## Run

Install the dependencies in a Python 3.13 environment, then open the relevant notebook:

```sh
python -m pip install -r requirements.txt
```

Signal analysis can also run directly from the repository root:

```sh
python HomeWork1/signal_analysis.py
```

It currently produces **3,829 signal segments** and their mean, standard deviation and RMS. The [feature table](HomeWork1/results/segment_features.csv) and [figure](HomeWork1/results/signal_analysis.png) are stored under Homework 1. The sampling rate is unverified, so frequency is expressed in cycles per sample.

## Data and references

- Signal: `201.mat`, supplied with the assignment. Titanic: [dataset linked in the assignment](https://www.kaggle.com/datasets/yasserh/titanic-dataset). Regression: `Q2.csv`, supplied with the assignment.
- Weather: [dataset linked in the assignment](https://www.kaggle.com/datasets/thedevastator/weather-prediction). Units, the original ECA&D source and prior preprocessing are documented in the [dataset metadata](MiniProject2/data/metadata.txt).
- The weather forecasting reference paper is included in the Mini Project 2 folder. Other inputs will be checked as each section is revised.

## Next steps

Complete Homework 1, revise Mini Project 1, complete Mini Project 2, revise Mini Project 3, then review all four parts for presentation.

A section is ready when it runs from start to finish, covers the assignment requirements and has results that can be explained. Saved outputs in the original notebooks are historical coursework and have not all been independently verified.
