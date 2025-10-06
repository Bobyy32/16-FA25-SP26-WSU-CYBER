# Sprint 1 Report (Dates: Aug 18 – Oct 05, 2025)

## YouTube link of Sprint 1 Video
https://www.youtube.com/watch?v=B2GdEjWw-JM

## What's New (User Facing)
 * Set up GitHub repository structure and documentation folders
 * Defined the overall research problem: **adversarial evasion in source code authorship attribution**
 * Identified key **functional and non-functional requirements**
 * Researched background topics in **stylometry, authorship verification**, and **adversarial machine learning**
 * Brainstormed a detailed **threat mode** outlining black box, gray box, and white box attacker levels

---

## Work Summary (Developer Facing)
During this sprint, our team focused on learning the conceptual foundations.. We began by clarifying our problem and objectives through meetings and brainstorming, creating a structured Threat Model document that defines the attacker's perspectives and knowledge levels. These activities helped us identify research gaps, refine our system requirements, and plan an implementation roadmap. The GitHub repository was created and documentation was written, detailing our project workflow. This sprint focused on learning and alignment rather than code production.

---

## Unfinished Work
We completed the initial Threat model draft and initial research but some related tasks remain open. Specifically the Threat Model still needs expanded sections on mimicry, and stylistic-suppression techniques, a defender side mitigations subsection, and a formal risk assessment matrix. We deliberately deferred dataset selection, preprocessing, and any baseline model implementation to sprint 2 so we could complete a stronger threat analysis. All incomplete issues have been moved to the next Sprint with owners and story points assigned. This approach helps preserve traceability and while ensuring remaining work will be tackled in the upcoming Sprint.

---

## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:

 * [#1 – Initial Project Setup and Repository Creation](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/1) 
 * [#2 – Define Project Objectives and Scope](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/2) 
 * [#4 – Functional and Non-Functional Requirements Draft](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/4) 

---
 
 ## Incomplete Issues/User Stories
 * [#3 – Threat Model Brainstorm Document](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/3) - We have a sufficient Threat Model which outlines what our research will encompass but we think that there is still more work to add.

---

## Code Files for Review
None this sprint — no functional code implemented yet. 

---
 
## Retrospective Summary
Here's what went well:
  * Team alignment on problem definition
  * Threat model brainstorm sessions clarified project scope and security assumptions.
  * Successful establishment of Github workflow
 
Here's what we'd like to improve:
   * Set clearer weekly objectives
   * Increase documentation consistency
   * Earlier coordination on milestone deliverables
  
Here are changes we plan to implement in the next sprint:
   * Begin dataset acquisition and basic preprocessing.
   * Begin implementation of concepts discovered in Sprint 1
   * Create a shared experiment tracking document

---

## Responsible Use of Generative AI
Our team used ChatGPT in limited and transparent ways. ChatGPT was only used in assignments that required it.

—

## Sprint Achievements and Challenges
**Achievements:**
   * Defined clear attacker taxonomy and access models
   * Created first version of Threat model Document
   * Established Github Repository and team workflow
   * Achieved alignment on project goals and deliverables

**Challenges:** 
   * Translating Stylometry theory into a practical implementation approach
   * Time management between research and documentation tasks
   * Limited familiarity with adversarial machine learning tools and framework.

—

## Next Steps and Sprint 2 Plan
   * Implement baseline authorship attribution model
   * Develop initial adversarial attack method for layout-level obfuscation
   * Continue refining Threat Model and add defender-side countermeasures
   * Document experimental pipeline setup on Github

—

## Conclusion
Sprint 1 established the foundation of the Stylometry Adversarial Evasion project through a series of assignments that outlines the project along with a threat model. With groundwork in place, we are able to transition towards implementation in Sprint 2.
