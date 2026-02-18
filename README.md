# Source Code Authorship Attribution and Verification under Adversarial Conditions

## Project Summary

### One-Sentence Description

A research-driven software system exploring how adversarial transformations can hide an author's identity in source code while preserving its functionality.

### Additional Information About the Project

This project investigates **adversarial stylometry** techniques for evading authorship attribution models applied to source code. Machine learning-based code attribution systems can be intentionally misled through obfuscation, layout manipulation, mimicry, and stylistic transformations. Our system simulates and evaluates these attacks by modifying lexical, syntactic, and structural features of code to confuse ML classifiers (Random Forest, SGD, Naive Bayes, Neural Network) while retaining semantic correctness.

The work contributes to understanding AI-driven authorship verification, privacy protection, and defensive detection of obfuscated coding styles. This research is relevant to cybersecurity analysts, malware forensics investigators, and academic integrity reviewers.

**Team 16:** Elang Sisson and Andrew Varkey  
**Client/Sponsor:** Tashi Stirewalt  
**Mentor:** Dr. Parteek Kumar  
**Course:** CptS 421/423 — Washington State University

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Add-ons

| Package | Purpose |
|---------|---------|
| scikit-learn | ML classifiers (Random Forest, SGD, Naive Bayes) |
| TF-IDF (via scikit-learn) | Feature extraction from source code |
| NumPy / Pandas | Data manipulation and metric computation |
| JSON / CSV (stdlib) | Structured results export |

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER.git
   cd 16-FA25-SP26-WSU-CYBER
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the installation by running a test prediction:
   ```bash
   python code/automation/run_batch.py --help
   ```

## Functionality

The system operates as a CLI-based pipeline with the following workflow:

1. **Data Ingestion** — Load raw source code samples from the `data/` directory.
2. **Feature Extraction** — Extract stylometric features using TF-IDF and custom stylometry metrics.
3. **Model Prediction** — Run authorship predictions across four ML classifiers (Random Forest, SGD, Naive Bayes, Neural Network).
4. **Adversarial Transformation** — Apply transformations (comment manipulation, formatting changes, variable renaming, structural modifications) to code samples.
5. **Evaluation** — Compare original vs. modified predictions, measuring evasion rates, confidence shifts, and stealth scores.
6. **Results Export** — Output structured results to CSV and JSON for downstream analysis.

### Batch Automation (Sprint 4)

Run automated adversarial experiments across all models:
```bash
python code/automation/run_batch.py --input data/samples/ --output results/
```

This generates per-attempt metrics, per-model predictions, and aggregate evasion statistics.

## Known Problems

- Centralized result synchronization is not yet implemented — concurrent experiment runs may overwrite shared CSV files. A database-backed solution is planned for Sprint 5.
- Cross-model statistical trend analysis is incomplete; only per-model metrics are currently tracked.
- The adversarial test dataset is limited in size; expansion is planned for upcoming sprints.

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Additional Documentation

- **Sprint Reports:** Located in the [`docs/`](docs/) directory
  - [Sprint 4 Report](docs/Sprint4_report.md)
- **Minutes of Meeting (MoM):** Available in the [`docs/`](docs/) directory
- **Demo Videos:** Unlisted YouTube links provided in each sprint report
- **Project Board:** [GitHub Projects Kanban](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/projects/1)

## License

See [LICENSE.txt](LICENSE.txt) for details.
