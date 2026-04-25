# Sprint 6 Report (03/15/2026 to 04/24/2026)

## YouTube link of Sprint * Video (Make this video unlisted)
https://www.youtube.com/watch?v=c8FPKvlthV0

## What's New (User Facing)
 * Added evolutionary prompt refinement system in our documentation for improving attack strategies over multiple rounds
 * Added visualization improvements (box plots, distribution graphs, category comparisons)
 * Expanded transformation categories (renaming, restructuring, formatting, control flow, comments, dead code) 

## Work Summary (Developer Facing)
During this sprint, our team focused on strengthening the experimental methodology and improving the structure of our adversarial stylometry system. We refined our evolutionary prompt framework, where transformation prompts are iteratively improved based on prior results, allowing us to discover effective attack strategies instead of relying on random exploration. We also improved our evaluation pipeline by integrating multiple classifiers and formalizing two key metrics: evasion rate and stealth. One major challenge we addressed was defining stealth, which is subjective. Additionally, we improved automation through scripting tools and an API-based workflow to streamline testing across multiple files and models. These improvements helped us better understand the tradeoff between evasion and detectability and provided more consistent and interpretable results. 


## Unfinished Work
While significant progress was made, several areas remain incomplete. Stealth evaluation still requires further validation since current metrics are only proxies and may not fully reflect real-world detectability. Prompt evolution still relies partially on manual intuition and trial-and-error rather than a fully systematic optimization process. Some planned extensions, such as cross-category transformations and adversarial retraining of models, weren’t fully implemented during this sprint and have been deferred to future work.


## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:

 * https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/22
 * https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/26
 * https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/27

 
 ## Incomplete Issues/User Stories
 Here are links to issues we worked on but did not complete in this sprint:
 
 * https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/23 <<Stealth metric remains a proxy and requires further validation/testing>>
 * https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/12 <<Defending against misattribution techniques would be a good follow up for next semester as it’s alot>>
 * https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/24 <<Did not have the time for more comprehensive tests past what we did>>

## Code Files for Review
Please review the following code files, which were actively developed during this sprint, for quality:
 * [stylometry_api](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/tree/main/code/stylometry_api)


## Retrospective Summary
Here's what went well:
  * Strong improvement in methodology clarity and structure
  * Clarified to client about our evolutionary prompt refinement system
  * Improved visualization and interpretability of results 
 
Here's what we'd like to improve:
   * Better formalization and validation of stealth metric
   * Reduce reliance on trial-and-error in prompt engineering
   * Improve reproducibility given variability in LLM outputs
  
Here are changes we plan to implement in the next sprint:
   * Explore cross-category attacks 
   * Expand dataset to more authors and potentially other programming languages
   * Investigate impersonation attacks 
