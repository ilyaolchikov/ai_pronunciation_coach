# Third-party notices

This project is MIT-licensed, but the following English pronunciation runtime components keep
their own licences. This notice records the versions and assets used by the frozen release path;
it does not replace their licence texts.

## Pronunciada 1.1.0

- Component: `pronunciada==1.1.0`
- Source: https://github.com/moglearn/pronunciada/tree/v1.1.0
- Licence: Apache License 2.0
- Upstream notice and licence: included in the installed Python distribution and available at
  https://github.com/moglearn/pronunciada/tree/v1.1.0
- Locked wheel SHA-256:
  `d9ecff7ca562567bb392ef0ea5b4c29e6865eaf10dc083c1ecc3ad52e189eff6`

Pronunciada includes English scoring assets trained from or calibrated with datasets including
SpeechOcean762 and FLEURS, both attributed upstream under CC BY 4.0. See Pronunciada's complete
per-file audit before changing the language bundle or redistributing its packaged data:
https://github.com/moglearn/pronunciada/blob/v1.1.0/docs/LICENSING.md

## ZIPA acoustic model

- Component: `anyspeech/zipa-small-crctc-500k`, int8 ONNX model and token inventory
- Source revision:
  https://huggingface.co/anyspeech/zipa-small-crctc-500k/tree/dfe87c97662606fa6db90ad06f627d4363314744
- Licence: Creative Commons Attribution 4.0 International
- Licence text: https://creativecommons.org/licenses/by/4.0/legalcode
- Authors: Jian Zhu, Farhan Samir, Eleanor Chodroff, and David R. Mortensen
- Work: “ZIPA: A family of efficient models for multilingual phone recognition,” ACL 2025
- Citation: https://aclanthology.org/2025.acl-long.961/

Pronunciada creates a local ONNX cache variant that exposes intermediate tensors; the model
parameters are unchanged. The application validates these release assets before use:

- Downloaded int8 model SHA-256:
  `d0e28b68164e8b1fbd6105100c01798828aa0855000ce9bbbd1a2cec233adf13`
- Pronunciada cache variant SHA-256:
  `36a7cdc229e08c57b6c568fc0a4b6dc697c01f435b09183699b9a1f5b19c034c`
- Token inventory SHA-256:
  `f8e042a0c9130532b22d03ec7cae2f75a23fbec70c450c31a8efb51787b2b8fe`

## CMU Pronouncing Dictionary

Pronunciada's English lexicon files are derived from CMUdict, copyright 1993–2015 Carnegie
Mellon University, under its BSD-2-Clause-style licence. The complete notice is included in the
installed Pronunciada distribution.

## Scope

This application currently uses only Pronunciada's English runtime. Pronunciada's Italian,
Portuguese, and Russian bundles contain CC BY-SA material and require an additional distribution
review before they are enabled. Benchmark corpora and locally recorded audio are not application
runtime assets and are not redistributed by this project.
