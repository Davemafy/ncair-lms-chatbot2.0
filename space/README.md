---
title: NCAIR LMS Chatbot 2.0
emoji: 🇳🇬
colorFrom: green
colorTo: gray
sdk: gradio
python_version: 3.12.12
app_file: app.py
models:
  - NCAIR1/N-ATLaS
short_description: Text-only NCAIR LMS assistant for English, Hausa, Yoruba and Igbo
---

# NCAIR LMS Chatbot 2.0 — ZeroGPU demo

This Space is the live N-ATLaS V2 deployment for the NCAIR LMS Chatbot 2.0 project.

It supports **English, Hausa, Yoruba and Igbo** text input. N-ATLaS performs language-aware semantic routing and generates grounded answers from official NCAIR LMS evidence.

The live ZeroGPU deployment uses the repository's curated official text knowledge source to keep cold-start time low. The canonical academic implementation and benchmark remain in the main GitHub repository and use the complete source set.

Source: https://github.com/Davemafy/ncair-lms-chatbot2.0

## Required setup

1. Accept the access conditions for `NCAIR1/N-ATLaS`.
2. Add an `HF_TOKEN` Space secret belonging to an account with model access.
3. In **Settings → Hardware**, choose **ZeroGPU**.

Attribution: N-ATLaS is provided by Awarri Technologies and the Federal Ministry of Communications, Innovation and Digital Economy.
