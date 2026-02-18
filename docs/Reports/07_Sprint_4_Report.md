# Sprint 4 Report (Dec 07, 2025 to Feb 17, 2026)
## YouTube link of Sprint * Video (Make this video unlisted)

## What's New (User Facing)
* Automated adversarial experiment runner for batch testing across multiple attribution models
* CSV + JSON export of evasion metrics, confidence shifts, and stealth scores
* Category-based organization for transformation types 
* Improved metric reporting 

## Work Summary (Developer Facing)
During Sprint 4, our team focused on scaling experimentation and improving reproducibility. Rather than manually running adversarial transformations and collecting results, we designed and implemented a Python automation framework that executes batch adversarial tests across multiple authorship attribution models. We structured outputs to include per-attempt metrics, per-model predictions, and aggregate evasion statistics, exporting results into structured CSV and JSON formats for downstream analysis. A key challenge was ensuring consistency across model outputs and standardizing metric calculations. We refactored portions of the evaluation pipeline to normalize model outputs and created a results tracking structure that associates each transformation attempt with its corresponding metrics and metadata. This sprint significantly improved experimental rigor, reproducibility, and scalability. Instead of anecdotal adversarial success cases, we now generate structured, quantitative evidence of model vulnerability.

## Unfinished Work
We began exploring improved synchronization of batch results to prevent accidental overwrites when multiple team members run experiments. While initial design discussions were completed, we did not implement a database-backed solution in this sprint. Progress has been documented in the related GitHub issue, and acceptance criteria have been partially checked. The issue has been moved to the next sprint to implement a structured persistence layer.


## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:
* Automated Batch Adversarial Test Runner
* Structured Results Export
* Per-Model Evasion Metric Tracking
Reminders (Remove this section when you save the file):
* Each issue should be assigned to a milestone
* Each completed issue should be assigned to a pull request
* Each completed pull request should include a link to a "Before and After" video
* All team members who contributed to the issue should be assigned to it on
GitHub
* Each issue should be assigned story points using a label
* Story points contribution of each team member should be indicated in a comment

## Incomplete Issues/User Stories
Here are links to issues we worked on but did not complete in this sprint:
* Centralized Results Synchronization <<We determined that a database-backed approach would be more robust than shared CSV files, but implementation was deferred to avoid scope creep late in the sprint.>>
* Expanded Adversarial Test Dataset <<We prioritized building a robust automation framework for batch experiment execution before scaling the number of adversarial test cases. While the infrastructure is now complete, we have not yet run and analyzed the expanded test set>>
* Cross Model Trend Analysis and Visualization <<Initial metric tracking is implemented, but comprehensive statistical analysis of trends across models and attack categories has not yet been completed>>

## Code Files for Review
Please review the following code files, which were actively developed during this
sprint, for quality:
* [/Automation](16-FA25-SP26-WSU-CYBER/code/automation at main · Bobyy32/16-FA25-SP26-WSU-CYBER)
* [/Candidates](Adversarial_Stylometry/adversarial_samples/candidates at main · Tashi-Stirewalt/Adversarial_Stylometry)

## Retrospective Summary
Here's what went well:
* Automation significantly reduced manual workload and human error
* Metric standardization improved experimental validity
* Strong collaboration on refactoring evaluation logic

Here's what we'd like to improve:
* Earlier architectural planning for result persistence
* Better branch coordination when integrating automation changes
* More time allocated to statistical validation

Here are changes we plan to implement in the next sprint:
* Implement append-only run tracking with unique run IDs
* Introduce SQL-backed persistence for structured result storage
* Add statistical comparison of attack categories
* Improve visualization of evasion trends across models

