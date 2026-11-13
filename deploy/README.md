---
title: MDST City Generator
emoji: 🏙️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Describe a city in words, get a 3D-printable model
---

# MDST 3D City Generator

Describe a city in plain English (*"a harbour city with high mountains"*) and
get a procedurally generated city back: a 2D map, a 3D model you can rotate,
and a watertight `.stl` you can 3D-print. Or set all ten parameters yourself
in manual mode.

Built by the Michigan Data Science Team (Fall 2026). A small neural network
reads your sentence and predicts the generator's ten parameters; the
procedural generator (`citygen`) builds the terrain, roads, parks and
buildings from them.

This Space is deployed automatically from GitHub; edit the code there, not here.
