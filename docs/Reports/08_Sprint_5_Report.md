# Sprint 5 Report (02/17/2026 to 03/15/2026)
## YouTube link of Sprint * Video
https://www.youtube.com/watch?v=soQTqalV26Y

## What's New (User Facing)
* Evolutionary prompt refinement framework for systematically discovering optimal adversarial
transformations
* Expanded adversarial test dataset with broader coverage of transformation categories and
authors
* Per-model vulnerability profiling identifying which classifiers are most and least robust to
specific attack types

## Work Summary
During Sprint 5, our team shifted focus from automation infrastructure to analytical depth, designing and implementing an evolutionary prompt refinement framework that systematically discovers the most effective adversarial transformations. Rather than testing prompts in isolation, the framework tracks positive and negative results across iterations allowing us to build on successful transformations while pruning ineffective results to converge on optimal evasion methods. A key challenge was defining a structured methodology that could be clearly documented and replicated by future researchers. We addressed this by formalizing the evolution process with explicit selection criteria, result tracking, and iteration logic. This sprint also produced our first comprehensive cross-model and cross-feature comparisons, revealing which stylometric features are most impactful for evasion and which of our four classifiers are most vulnerable to specific transformation categories.

## Unfinished Work
We began planning the documentation of our experimental methods and findings to ensure the
research is understandable and reproducible by future teams, but have not officially started the formal writeup during this sprint. The documentation covering our evolutionary framework methodology, feature level findings, and replication instructions has been turned into a github issue for the next sprint. Additionally, we did not implement mimicry or impersonation attack strategies as originally planned. Instead, we prioritized deepening our evasion and attribution analysis before expanding to new attack types. The centralized results synchronization and cross-model trend visualization also remain incomplete and have been carried forward.

## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:
* https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/17
* https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/21
* https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/22





## Incomplete Issues/User Stories
Here are links to issues we worked on but did not complete in this sprint:
* https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/18 <<Initial comparative analysis was completed, but comprehensive statistical visualization has not yet been built out.>>
* https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/16 <<Carried forward from Sprint 4; we focused sprint effort on the analytical framework rather than persistence infrastructure.>>
* https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/12 <<We prioritized completing the evolutionary framework and cross-model analysis before defining on how we defend against attacks.>>


## Code Files for Review
Please review the following code files, which were actively developed during this
sprint, for quality:
* [Results Folder](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/tree/main/code/automation/results)


## Retrospective Summary
Here's what went well:
* The evolutionary prompt refinement framework significantly improved our ability to
systematically discover effective transformations
* Cross-model comparison revealed clear patterns in classifier vulnerabilities that inform future
attack strategies
* Expanded dataset provided more robust and generalizable results across authors and
transformation categories

Here's what we'd like to improve:
* Dedicate time for documentation
* Implement persistent storage sooner to avoid continued reliance on CSV-based result tracking

Here are changes we plan to implement in the next sprint:
* Formalize and document experimental methodology and key findings for reproducibility
* Build visualizations of evasion trends and feature importance across models
* Implement database-backed persistence for structured result storage
