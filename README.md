# SuQA: SUmmarizer-Augmented QA

This is a code and data repository for the following paper:

Naoya Inoue, Harsh Trivedi, Steven Sinha, Niranjan Balasubramanian and Kentaro Inui.
Summarize-then-Answer: Generating Concise Explanations for Multi-hop Reading Comprehension.
To appear in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP2021).

[preprint](https://arxiv.org/abs/2109.06853)


## News
- 9/23/2021: Python API (prediction only) released.

**Full code coming soon in October.**


## Prerequisites

- Python 3.x
- `transformers==4.4.2`
- `torch`


## SuQA Python API (prediction only)

### Download SuQA pretrained weights

- Clone the repository.
- Store the following pretrained weights in `models` subdirectory.
    - [suqa_qam_allenai_unifiedqa-t5-base.torch](https://drive.google.com/file/d/1-Wve58Gl5Mi1d6Cv6oztJghpWBRLsfXg/view?usp=sharing) (850.4MB)
    - [suqa_explainer_distilbart-cnn-12-6.torch](https://drive.google.com/file/d/1-90vejEyydQkgiA7vl0DnP46h91vxqj0/view?usp=sharing) (1.4GB)


### Load and predict in Python

```python
import torch
import logging

from suqa import SuQA

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SuQA from the pretrained weights.
suqa = SuQA.from_pretrained(device,
                            "models/suqa_qam_allenai_unifiedqa-t5-base.torch",
                            "models/suqa_explainer_distilbart-cnn-12-6.torch")

# Feed questions and passages as follows.
questions = [
  (
    "Which country is Sendai belong to?",
    "Sendai is the capital city of Miyagi Prefecture, the largest city in the Tōhoku region, and the second largest city north of Tokyo. Miyagi Prefecture (宮城県, Miyagi-ken) is a prefecture of Japan located in the Tōhoku region of Honshu."
  ),
  (
    "Who was born first, Krzysztof Zanussi or Thom Andersen?",
    "Krzysztof Zanussi, (born 17 June 1939) is a Polish film and theatre director, producer and screenwriter. He is a professor of European film at the European Graduate School in Saas-Fee, Switzerland where he conducts a summer workshop. He is also a professor at the Silesian University in Katowice. Thom Andersen (born 1943, Chicago) is a filmmaker, film critic and teacher."
  ),
]

# Predict!
ret = suqa.predict(questions)
```

The return value `ret` is now a list of tuples, each of which represents an explanation and an answer. 

```python
ret == [
    (
        'Sendai is the capital city of Miyagi Prefecture. Miyagi Prefecture is a prefecture of Japan.',
        'Japan'
    ),
    (
        'Krzysztof Zanussi is born on 17 June 1939. Thom Andersen is born on 1943.',
        'Krzysztof Zanussi'
    )
]
```
