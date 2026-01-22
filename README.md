# IndoMMLU Evaluation: Zero-Shot vs Chain of Thought Prompting

This repository contains evaluation scripts for the IndoMMLU dataset using various prompting strategies and multiple Large Language Models (LLMs).

## Dataset

**IndoMMLU Dataset** (`IndoMMLU.csv`)
- **Source**: Indonesian Massive Multitask Language Understanding dataset
- **Format**: CSV with columns: `id`, `subject`, `level`, `soal` (question), `jawaban` (choices), `kunci` (correct answer)
- **Subjects**: Multiple subjects including Bahasa Indonesia, Bahasa Jawa, Bahasa Sunda, Biologi, Fisika, Kimia, IPS, PPKN, Sejarah, Sosiologi, Ekonomi, Agama Islam, IPA, Kesenian, Bahasa Bali, etc.
- **Levels**: SD (Elementary), SMP (Middle School), SMA (High School), Seleksi PTN (University Entrance)

### Sampling Methodology

All evaluation scripts use **fair and consistent sampling**:

- **Subject Diversity**: Selects 20 subject-level groups from 15+ diverse subjects (ensures fair representation, not biased toward popular subjects)
- **Questions per Group**: 10 questions per (subject, level) group
- **Total Questions**: 200 questions per model (20 groups × 10 questions)
- **Reproducibility**: Uses `random_state=42` for consistent question selection across runs
- **Fair Comparison**: All scripts evaluate the same questions, enabling fair comparison across different prompting methods

## Evaluation Methods

### 1. Zero-Shot Evaluation

**Description**: Models answer questions without examples or training, using only the prompt instructions.

#### Scripts:
- `zero-shot-en.py`: English prompt version
- `zero-shot-id.py`: Indonesian prompt version

#### Models Evaluated:
- `SeaLLMs/SeaLLMs-v3-7B-Chat`
- `CohereLabs/aya-expanse-8b`
- `sail/Sailor-7B-Chat`
- `google/gemma-3-12b-it`

#### Prompts:

**English (`zero-shot-en.py`)**:
```
The following question is in Indonesian: {question}

select the answer from the following choices:
{choices}
Let's think step by step. First analyze the question, then evaluate each option, and finally choose the single best answer.
In the last line, provide only the final answer letter (A, B, C, D, or E). Answer:
```

**Indonesian (`zero-shot-id.py`)**:
```
Pertanyaan: {question}

Pilihan jawaban:
{choices}

Jawablah dengan memilih salah satu huruf (A, B, C, D, atau E) yang paling tepat. Jawaban:
```

#### Outputs:
- `results_zero_shot_en.txt`: Full execution log (English)
- `results_zero_shot_en_summary.txt`: Summary statistics (English)
- `results_zero_shot_id.txt`: Full execution log (Indonesian)
- `results_zero_shot_id_summary.txt`: Summary statistics (Indonesian)

---

### 2. Chain of Thought (COT) - Step-by-Step Reasoning

**Description**: Models are prompted to think step-by-step and explain their reasoning before providing the final answer.

#### Scripts:
- `COT-en.py`: English prompt version
- `COT-id.py`: Indonesian prompt version

#### Prompts:

**English (`COT-en.py`)**:
```
Question: {question}

Choices:
{choices}

Please think step by step and explain your reasoning for each question. After your reasoning, provide your final answer in the format "Final Answer: X" where X is the letter (A, B, C, D, or E).
```

**Indonesian (`COT-id.py`)**:
```
Question: {question}

Choices:
{choices}

Please think step by step and explain your reasoning for each question. After your reasoning, provide your final answer in the format "Final Answer: X" where X is the letter (A, B, C, D, or E).
```

*Note: Currently uses English prompt format; Indonesian version may be available in other variants.*

#### Outputs:
- `results_cot_id.txt`: Full execution log (Indonesian)
- `results_cot_en.txt`: Full execution log (English)
- `COT_id_reasoning_results.csv`: Detailed results CSV (Indonesian)
- `COT_id_reasoning_results.tex`: LaTeX summary tables (Indonesian)
- `COT_en_reasoning_results.csv`: Detailed results CSV (English)
- `COT_en_reasoning_results.tex`: LaTeX summary tables (English)

---

### 3. Chain of Thought (COT) - Structured Reasoning

**Description**: Models use a structured 4-step reasoning framework with detailed instructions.

#### Scripts:
- `COT-en-reasoning.py`: English prompt version
- `COT-id-reasoning.py`: Indonesian prompt version (if exists)

#### Prompts:

**English (`COT-en-reasoning.py`)**:
```
Question: {question}

Choices:
{choices}

Please think step by step and explain your reasoning for each question.
After your reasoning, provide your final answer in the format "Final Answer: X" where X is the letter (A, B, C, D, or E).
```

**Note**: This version uses a more detailed reasoning framework with structured steps.

#### Outputs:
- `results_cot_id_reasoning.txt`: Full execution log
- `results_cot_en_reasoning.txt`: Full execution log
- `COT_en_reasoning_2_results.csv`: CSV with columns: question, ground_truth, prediction, reasoning
- `COT_en_reasoning_2_results.tex`: LaTeX summary tables

---

### 4. Chain of Thought (COT) - Simple Reasoning with Example

**Description**: Models are asked to provide simple reasoning with an example (1-sentence prompt).

#### Scripts:
- `COT-en-reason_eg.py`: English prompt version
- `COT-id-reason_eg.py`: Indonesian prompt version

#### Prompts:

**English (`COT-en-reason_eg.py`)**:
```
Question: {question}

Choices:
{choices}

Please provide simple reasoning with an example and your final answer in the format "Final Answer: X" where X is the letter (A, B, C, D, or E).
```

**Indonesian (`COT-id-reason_eg.py`)**:
```
Pertanyaan: {question}

Pilihan:
{choices}

Silakan berikan alasan sederhana dengan contoh dan jawaban akhir Anda dalam format "Jawaban Akhir: X" di mana X adalah huruf (A, B, C, D, atau E).
```

#### Outputs:
- `results_cot_id_reason_eg.txt`: Full execution log (Indonesian)
- `results_cot_en_reason_eg.txt`: Full execution log (English)
- `COT_id_reason_eg_summary.txt`: Text summary (Indonesian)
- `COT_id_reason_eg_results.csv`: CSV results (Indonesian)
- `COT_id_reason_eg_results.tex`: LaTeX tables (Indonesian)
- `COT_en_reason_eg_summary.txt`: Text summary (English)
- `COT_en_reason_eg_results.csv`: CSV results (English)
- `COT_en_reason_eg_results.tex`: LaTeX tables (English)

---

## Prompt Comparison Summary

| Method | Language | Prompt Length | Key Features |
|--------|----------|---------------|--------------|
| **Zero-Shot EN** | English | ~4 sentences | Step-by-step instruction, no examples |
| **Zero-Shot ID** | Indonesian | ~2 sentences | Direct answer request |
| **COT Step-by-Step** | English/Indonesian | 1 sentence | "Think step by step" instruction |
| **COT Structured** | English | 1-2 sentences | Detailed reasoning framework |
| **COT Simple + Example** | English/Indonesian | 1 sentence | Requests reasoning with example |

---

## Key Differences in Prompting Strategies

### 1. **Zero-Shot vs Chain of Thought**
- **Zero-Shot**: Direct answer without explicit reasoning steps
- **Chain of Thought**: Explicitly requests reasoning before answering

### 2. **Prompt Language**
- **English Prompts**: Instructions in English, questions remain in Indonesian
- **Indonesian Prompts**: Both instructions and questions in Indonesian

### 3. **Reasoning Complexity**
- **Simple COT**: 1-sentence prompt requesting reasoning
- **Structured COT**: Detailed multi-step reasoning framework
- **COT with Example**: Requests reasoning with example

---

## Execution Scripts

### Shell Scripts:
- `run_zero_shot.sh`: Runs both zero-shot evaluations (ID and EN)
- `run_cot.sh`: Runs COT step-by-step evaluations (ID and EN)
- `run_cot-reason.sh`: Runs COT structured reasoning evaluations (ID and EN)
- `run_cot_reason_eg.sh`: Runs COT simple reasoning with example (ID and EN)

### Usage:
```bash
# Run zero-shot evaluations
./run_zero_shot.sh

# Run COT evaluations
./run_cot.sh

# Run COT reasoning with example
./run_cot_reason_eg.sh
```

---

## Results Format

### CSV Outputs:
- **Simplified Format**: `question`, `ground_truth`, `prediction`, `reasoning`
- **Detailed Format**: Includes additional columns like `model`, `question_id`, `subject`, `level`, `is_correct`

### LaTeX Outputs:
- Comparative summary tables across all models
- Per-model subject-level accuracy breakdowns
- Best model identification

### Text Summaries:
- Overall accuracy comparison
- Detailed statistics by model
- Subject-level performance analysis

---

## Fairness Guarantees

✅ **Consistent Sampling**: All scripts use the same sampling methodology
✅ **Same Questions**: All scripts evaluate identical question sets (200 questions)
✅ **Subject Diversity**: Fair distribution across 15+ subjects
✅ **Reproducibility**: `random_state=42` ensures consistent results
✅ **Fair Comparison**: Differences in results reflect prompting effectiveness, not sampling bias

---

## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- pandas
- tqdm
- CUDA-capable GPU (recommended)
- Hugging Face token for model access

---

## Citation

If you use this evaluation framework, please cite the IndoMMLU dataset and the models evaluated.
