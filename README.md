<p align="center">
	This repository contains the dataset and the implementation of the core ideas from our EMNLP 2025 paper *Detecting Corpus-Level Knowledge Inconsistencies in Wikipedia with Large Language Models*.
</p>

<p align="center">
		<a href="https://arxiv.org/abs/2509.23233" target="_blank"><img src="https://img.shields.io/badge/Paper-arXiv%20-B31B1B" alt="Paper" /></a>
		<a href="https://github.com/stanford-oval/inconsistency-detection/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/stanford-oval/inconsistency-detection?style=social" alt="GitHub Stars" /></a>
</p>


---

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [The WikiCollide Dataset](#the-wikicollide-dataset)
5. [Citation](#citation)
6. [License](#license)

---

## Overview
The CLAIRE agent uses tool-enabled reasoning to:

- extract atomic claims from a given passage,
- understand difficult terms and similar-sounding entities, and
- search for contradicting evidence in the English Wikipedia,
- produce richly justified verdicts explaining whether a claim is **consistent** or **inconsistent** with the broader corpus.

The agent follows a three-stage pipeline:

1. **Claim extraction** (`claire_agent.fact_extraction`): identifies atomic statements in a passage using an LLM prompt.
2. **Evidence gathering** (`claire_agent.inconsistency.tools`): uses search and other tools to retrieve relevant Wikipedia passages
3. **Report generation** (`claire_agent.inconsistency.agent`): calls a ReAct-style LangChain agent that writes a natural-language report with citations

Search is performed using the search API introduced in [WikiChat](https://github.com/stanford-oval/Wikichat).

## Installation
### 1. Prerequisites
- Linux (tested on Ubuntu 22.04) or macOS.
- Python 3.12 (managed automatically via [pixi](https://pixi.sh)).
- Access credentials for your preferred LLM provider.

### 2. Install pixi (one-time)
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### 3. Clone the repository
```bash
git clone https://github.com/yourusername/inconsistency-detection.git
cd inconsistency-detection
```

### 4. Create the project environment
```bash
pixi shell
```
The first run resolves Python 3.12 and all dependencies defined in `pixi.toml`. Subsequent invocations reuse the cached env.

### 5. Configure the LLM provider
The agent relies on LangChain's provider ecosystem. Export the environment variables required by the LLM provider **before** running the code.
You should do so by creating a `.env` file. The code will automatically load it before any execution.
Common examples:

```bash
# Azure OpenAI (default CLI choice)
export AZURE_OPENAI_API_KEY="<your key>"
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com"
export AZURE_OPENAI_API_VERSION="2024-06-01"

# OpenAI
export OPENAI_API_KEY="<your key>"

# Anthropic
export ANTHROPIC_API_KEY="<your key>"
```

> ℹ️ Refer to the [LangChain chat model docs](https://python.langchain.com/docs/integrations/chat) for provider-specific variables.


## Quick Start
### Run the CLAIRE agent on the WikiCollide dataset
For example, to run the agent on the first 5 examples of the dev split:
```bash
# Inside `pixi shell`
python run_agent.py \
  --engine gpt-5 \
  --model_provider azure_openai \
  --dataset wikicollide_dataset/dev.json \
  --num_results_per_query 5 \
  --input_size 5
```

The command shows a progress bars and prints a formatted inconsistency report for each claim to the terminal. Full debug traces are also written to `debug_logs.log`.


### Analyze an arbitrary passage
You can call the agent directly to evaluate a single passage:

Example code:
```python
from claire_agent import InconsistencyAgent
from utils.report_rendering import render_inconsistency_report
import asyncio

agent = InconsistencyAgent(
	"gpt-5-mini",
	"azure_openai",
	num_results_per_query=3,
	reasoning_effort="low",
)

async def main():
    reports = await agent.analyze_passage_for_inconsistencies(
        passage=(
            "Title: Haruki Murakami > Biography\n\n"
            "Haruki Murakami (村上 春樹, Murakami Haruki; born January 12, 1949[1]) is a Japanese writer. "
            "His novels, essays, and short stories have been best-sellers in Japan and internationally."
        ),
    )

    for report in reports:
        render_inconsistency_report(report)

if __name__ == "__main__":
    asyncio.run(main())
```

<details>
<summary>Example Output</summary>

```markdown
────────────────────────── Inconsistency Report ───────────────────────────
╭───────────────────────────────── Claim ─────────────────────────────────╮
│ His novels have been best-sellers in Japan and internationally.         │
╰─────────────────────────────────────────────────────────────────────────╯
╭─────────────────────── Claim Was Extracted From ────────────────────────╮
│ Title: Haruki Murakami > Biography                                      │
│                                                                         │
│ Haruki Murakami (村上 春樹, Murakami Haruki; born January 12, 1949[1])  │
│ is a Japanese writer. His novels, essays, and short stories have been   │
│ best-sellers in Japan and internationally.                              │
╰─────────────────────────────────────────────────────────────────────────╯
							  ╭─ Verdict ──╮                               
							  │ CONSISTENT │                               
							  ╰────────────╯                               
╭───────────────────────────────── Why? ──────────────────────────────────╮
│ Multiple sources indicate Murakami's books have sold extremely well     │
│ both in Japan and overseas. His biography notes that his work has been  │
│ "best-sellers in Japan and internationally," translated into dozens of  │
│ languages and sold millions of copies outside Japan [1]. Specific       │
│ examples support this: Norwegian Wood sold millions of copies in Japan  │
│ and made him widely famous there [3], and Colorless Tsukuru Tazaki      │
│ reached very high preorders and exceeded nearly one million copies in   │
│ print within weeks of release in Japan [2]. These citations             │
│ collectively support the claim that his novels have been best-sellers   │
│ domestically and internationally.                                       │
╰─────────────────────────────────────────────────────────────────────────╯
╭────────────── How the Claim Could Be Reworded for Clarity ──────────────╮
│ The claim is broadly supported, but could be made slightly more precise │
│ by saying "many of his novels and other works have been best-sellers in │
│ Japan and internationally" or "his books have frequently been           │
│ best-sellers in Japan and abroad," since not every single title may     │
│ have topped charts everywhere.                                          │
╰─────────────────────────────────────────────────────────────────────────╯
────────────────────── Passages that Were Looked At ───────────────────────
┌────────────────────────── [1] Haruki Murakami ──────────────────────────┐
│ Japanese writer (born 1949)                                             │
│                                                                         │
│                                                                         │
│   Haruki Murakami 村上 春樹                                             │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━                                            │
│   Murakami in 2018                                                      │
│   Born                                                                  │
│   Occupation                                                            │
│   Language                                                              │
│   Alma mater                                                            │
│   Period                                                                │
│   Genres                                                                │
│   Literary movement                                                     │
│   Notable works                                                         │
│   Signature                                                             │
│   Website                                                               │
│   www.harukimurakami.com                                                │
│                                                                         │
│                                                                         │
│ Haruki Murakami (村上 春樹, Murakami Haruki, born January 12, 1949) is  │
│ a Japanese writer. His novels, essays, and short stories have been      │
│ best-sellers in Japan and internationally, with his work translated     │
│ into 50 languages and having sold millions of copies outside Japan. He  │
│ has received numerous awards for his work, including the Gunzo Prize    │
│ for New Writers, the World Fantasy Award, the Tanizaki Prize, Yomiuri   │
│ Prize for Literature, the Frank O'Connor International Short Story      │
│ Award, the Noma Literary Prize, the Franz Kafka Prize, the Kiriyama     │
│ Prize for Fiction, the Goodreads Choice Awards for Best Fiction, the    │
│ Jerusalem Prize, and the Princess of Asturias Awards. Growing up in     │
│ Ashiya, near Kobe before moving to Tokyo to attend Waseda University,   │
│ he published his first novel Hear the Wind Sing (1979) after working as │
│ the owner of a small jazz bar for seven years. His notable works        │
│ include the novels Norwegian Wood (1987), The Wind-Up Bird Chronicle    │
│ (1994–95), Kafka on the Shore (2002) and 1Q84 (2009–10); the last was   │
│ ranked as the best work of Japan's Heisei era (1989–2019) by the        │
│ national newspaper Asahi Shimbun's survey of literary experts. His work │
│ spans genres including science fiction, fantasy, and crime fiction, and │
│ has become known for his use of magical realist elements. His official  │
│ website cites Raymond Chandler, Kurt Vonnegut and Richard Brautigan as  │
│ key inspirations to his work, while Murakami himself has named Kazuo    │
│ Ishiguro, Cormac McCarthy, and Dag Solstad as his favorite contemporary │
│ writers. Murakami has also published five short story collections,      │
│ including First Person Singular (2020), and non-fiction works including │
│ Underground (1997), an oral history of the Tokyo subway sarin attack,   │
│ and What I Talk About When I Talk About Running (2007), a memoir about  │
│ his experience as a long-distance runner. His fiction has polarized     │
│ literary critics and the reading public. He has sometimes been          │
│ criticised by Japan's literary establishment as un-Japanese, leading to │
│ Murakami's recalling that he was a "black sheep in the Japanese         │
│ literary world". Meanwhile, Murakami has been described by Gary         │
│ Fisketjon, the editor of Murakami's collection The Elephant Vanishes    │
│ (1993), as a "truly extraordinary writer", while Steven Poole of The    │
│ Guardian praised Murakami as "among the world's greatest living         │
│ novelists" for his oeuvre.                                              │
└─────────────────────────────────────────────────────────────────────────┘
┌─ [2] Colorless Tsukuru Tazaki and His Years of Pilgrimage > Publishing ─┐
│ On 16 February 2013, the publishing company Bungeishunjū announced that │
│ Haruki Murakami's new novel was to be published in April. On 15 March,  │
│ the title "Colorless Tsukuru Tazaki and His Years of Pilgrimage" and    │
│ the release date of 12 April were disclosed. Preorders were placed      │
│ starting that day, and the sales reached 10 thousand copies on          │
│ Amazon.co.jp within 11 days. It took one day fewer than its             │
│ predecessor, 1Q84, to become the fastest selling book on Amazon.co.jp.  │
│ The publisher prepared 300,000 copies, the largest number of first      │
│ edition copies of a hardcover book in the company's history.            │
│ Furthermore, the number of copies to be printed over the course of      │
│ three more print runs before the release date was expected to reach     │
│ 450,000 copies. Prior to the book's release, statements such as Haruki  │
│ Murakami's messages on 28 February and 15 March, were issued to convey  │
│ fragments of information over the course of seven statements. However,  │
│ details of the novel were not disclosed. Furthermore, galleys, usually  │
│ given to other reviewers, newspapers, and bookstores before the         │
│ publication, were not created. The knowledge of the content of the book │
│ was limited to a small number of people. With the book's release date   │
│ announced to be at midnight on Friday 12 April 2013, late-night         │
│ bookstores in metropolitan Tokyo which were to start selling the book   │
│ at 0:00 a.m. witnessed long lines of more than 150 people. Seven days   │
│ after the release, the book had been printed 8 times for a total of     │
│ over one million copies in print, reportedly sold during the following  │
│ month. In November, point-of-sale information firm Oricon certified     │
│ 985,000 copies sold.                                                    │
└─────────────────────────────────────────────────────────────────────────┘
┌─────── [3] Haruki Murakami > Writing career > Wider recognition ────────┐
│ In 1985, Murakami wrote Hard-Boiled Wonderland and the End of the       │
│ World, a dream-like fantasy that took the magical elements of his work  │
│ to a new extreme. Murakami achieved a major breakthrough and national   │
│ recognition in 1987 with the publication of Norwegian Wood, a nostalgic │
│ story of loss and sexuality. It sold millions of copies among young     │
│ Japanese. Norwegian Wood propelled the barely known Murakami into the   │
│ spotlight. He was mobbed at airports and other public places, leading   │
│ to his departure from Japan in 1986. Murakami traveled through Europe,  │
│ lived in the United States and currently resides in Oiso, Kanagawa,     │
│ with an office in Tokyo. Murakami was a writing fellow at Princeton     │
│ University in Princeton, New Jersey, Tufts University in Medford,       │
│ Massachusetts, and Harvard University in Cambridge, Massachusetts.      │
│ During this time he wrote South of the Border, West of the Sun and The  │
│ Wind-Up Bird Chronicle.                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```
</details>

## The WikiCollide Dataset
The repository contains the **WikiCollide** dev and test splits under `wikicollide_dataset/`. Each record is a JSON object with:

1. **`claim`**
	- `claim_id`: unique identifier
	- `claim_text`: the claim under review
	- `claim_context_block`: the Wikipedia section where the claim originated
2. **`label`** – `"consistent"` or `"inconsistent"`
3. **`label_reasoning`** – human-written rationale
4. **`inconsistency_type`** – categorical tag describing why the claim is inconsistent (when applicable):
	- `CategoricalDiscrepancy`
	- `DefinitionDiscrepancy`
	- `DualityDiscrepancy_Explicit`
	- `DualityDiscrepancy_Implicit`
	- `NamedEntityDiscrepancy`
	- `NumericalDiscrepancy_Clear`
	- `NumericalDiscrepancy_OffByOne`
	- `SpatialDiscrepancy`
	- `TemporalDiscrepancy`
5. **`agent_trace`** – ordered list capturing the assisting agent's actions:
	- `action_name`
	- `action_argument`
	- `action_output` (strings or retrieval hit bundles with `document_title`, `section_title`, `content`, `last_edit_date`, `url`)

These traces are ideal for training or benchmarking planning-aware agents.


## Citation
If you use the WikiCollide dataset or the agent code, please cite:

```bibtex
@inproceedings{semnani2025inconsistency,
	title        = {Detecting Corpus-Level Knowledge Inconsistencies in Wikipedia with Large Language Models},
	author       = {Sina J. Semnani, Jirayu Burapacheep, Arpandeep Khatua, Thanawan Atchariyachanvanit, Zheng Wang, Monica S. Lam},
	booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)},
	year         = {2025}
}
```

## License
Code is released under the [Apache 2.0](LICENSE) license. Dataset is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](wikicollide_dataset/LICENSE) license.
