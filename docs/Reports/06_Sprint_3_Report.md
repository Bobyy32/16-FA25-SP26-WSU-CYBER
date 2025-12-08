# Sprint 3 Report (Nov 04, 2025 to Dec 07,2025)

## YouTube link of Sprint * Video (Make this video unlisted)
Final Presentation with a slide on sprint 3
https://www.youtube.com/watch?v=IpnhxGPHGcI

## What's New (User Facing)
 * Testing across multiple models (Random Forest, SGD, Naive Bayes, Neural Network)
 * Investigating which code transformations cause misattribution
 * Achieving a 75% evasion rate where adversarial modifications cause classifiers to predict "UNKNOWN"

## Work Summary (Developer Facing)
During this sprint, our team focused on investigating the root causes of misattribution in the authorship attribution models. We explored multiple approaches to identify which code transformations are most effective at causing evasion. To streamline our experimentation process, we developed a CLI tool that automates the comparison between original and adversarially modified code samples, outputting confidence scores from each classifier. This tool allows us to efficiently measure the impact of specific modifications and track changes in model predictions across Random Forest, SGD, Naive Bayes, and Neural Network classifiers. The main challenge of this sprint was narrowing down which stylometric features and transformation techniques contribute most significantly to successful evasion, as multiple factors often interact simultaneously.

## Unfinished Work
While we made progress on investigating misattribution causes and developed tooling to support our analysis, we have not yet established a formal testing framework or structured methodology for systematically identifying which code transformations lead to evasion. Our current approach has been exploratory, testing various modifications and observing results. Moving forward, we need to create a controlled experimental pipeline that isolates individual transformation types (such as identifier renaming, whitespace manipulation, and code refactoring) and measures their independent effects on classifier confidence. This structured approach will allow us to draw more definitive conclusions about which stylometric features are most vulnerable to adversarial manipulation. These tasks have been moved to the next sprint with updated acceptance criteria.

## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:

 * [#11 – Complete analysis of preexisting repo](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/11) 
 * [#7 – Investigate root causes of misattribution](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/7) 
 
 ## Incomplete Issues/User Stories
 Here are links to issues we worked on but did not complete in this sprint:
 
 * [#10 – Research on mitigating misattribution techniques](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/10) - This is delayed until we’ve established the cause and effect first. Then we’ll be able to start talking about mitigations.
 * [#9 – Research different misattribution techniques](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/9) - We are still finding new ways that cause misattribution. Our goal isn’t to go over every single one but to find what’s the most effective one.
 * [#12 – Defend against misattribution techniques](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/12) - Same as Issue #10, we have to establish the cause and effect first. This would normally be done after we are able find mitigation techniques then we’re able to create a framework to defend against these kind of attacks.
 
## Code Files for Review
Please review the following code files, which were actively developed during this sprint, for quality:
 * [cli-tool](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/tree/main/code/cli-tool)
 
## Retrospective Summary
Here's what went well:
  * Maintained good communication internally within our team
  * Integrated attribution models into a unified testing pipeline.
  * Early adversarial results show meaningful confidence drops on several models.



 
Here's what we'd like to improve:
   * Need better automation for repeated experiments.
   * Current adversarial edits must be expanded to include multi-feature attacks.
   * Schedule more consistent client meetings
  
Here are changes we plan to implement in the next sprint:
   * More stylometric features to analyze
   * Establishing a testing framework appropriate for the project
   * Create a structured method to finding which features cause the models to hallucinate the most
