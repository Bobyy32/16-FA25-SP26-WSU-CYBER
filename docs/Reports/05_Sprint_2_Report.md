# Sprint 2 Report (Dates: Oct 05, 2025 - Nov 04, 2025)

## YouTube link of Sprint 2 Video
https://www.youtube.com/watch?v=VRemS6xMdpg

## What's New (User Facing)
* Access to trained models for code authorship attribution
* Access to datasets containing author code and other test data
* Completed Threat Model Documentation

## Work Summary (Developer Facing)
During this Sprint, our team transitioned from threat modeling to the practical implementation of adversarial evasion techniques against the stylometry attribution models provided to us by our client, Tashi. We finalized and improved our Threat Model documentation and Analysis by following OWASP’s Threat Modeling STRIDE framework which addresses the only incomplete issue from Sprint 1. Working with the provided datasets and trained models, we are able to cause misattribution in the authorship attribution system. The biggest challenge was understanding the infrastructure of the authorship attribution system and analyzing the behaviours and outputs through the modification of the dataset samples.

## Unfinished Work
While we were able to cause misattribution in the models, we have not yet completed an in depth analysis of which stylometric techniques and code transformations are responsible for the evasion. This sprint we focused on achieving an initial misattribution as a proof of concept so that we have a solid foundation for a deeper investigation. The analysis of these techniques: identifier renaming, code refactoring, and mimicry requires further controlled experimentations and analysis of model outputs. Additionally, we have not yet covered all of the attack strategies we outlined in our Threat Model documentation.

## Completed Issues/User Stories
Here are links to the issues that we completed in this sprint:
* [#5 – Establish Updated Threat Model](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/5)
* [#6 – Establish baseline misattribution](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/6)


## Incomplete Issues/User Stories
Here are links to issues we worked on but did not complete in this sprint:
* [#7 – Investigate root causes of misattribution](https://github.com/Bobyy32/16-FA25-SP26-WSU-CYBER/issues/7) - Due to some delays from the client we weren’t able to dig as deep into the root causes of misattribution like we wanted to during this sprint.


## Code Files for Review

Code availability is under client’s authority.


## Retrospective Summary
Here's what went well:
* Team alignment on project scope and focus
* Hardware optimization during vectorization stage
* Client communication

Here's what we'd like to improve:
* More weekly meetings with client
* Setting time to work on project
* Organization of materials

Here are changes we plan to implement in the next sprint:
* Further research on misattribution causes
* More frequent group work sessions
