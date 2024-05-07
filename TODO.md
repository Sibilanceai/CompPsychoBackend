# TODO update the TODOs 
# Todolist for Backend/Toolbox

1. Test attachment and Big 5 analysis
2. Refine data collection, clean_data.py: we need it to give clean CSVs, consistent formatting and number and make sure they exist
3. CSVs should be deleted or appended as long as it is part of a single story, then removed for another story, maybe it should just be IDs for the stories and we store all the CSVs? We probably should use a light weight database. 
4. Finish simulation



# NOTES and Things to do later
1. Implement Hiearchical levels to the event generation, when we generate an event it should generate it in all 3 levels separately for the same overall event then use the 3 to come up with a way of adding a new cohesive event, all three need to be added to a memory 

2. Implement temporal dynamics, similar approach to the hierarchical levels, we need to when extracting events for the event chains also take into account both the hierarchical levels nad the temproal categories for each. When we simulate as well we need it extracted for each. 

3. We need a context compressor that preserves story coherence and continuity, it should have everything we need 

4. Inputting stories, videos, other modalities to continue or analyze into the website/tools

5. Image generation character and setting consistency 

6. Visualization capabilities of personalitiies and story progression/analysis, archetypal etc

7. OpenSource model integration: lighter weight models for steps that require less capability, use stable diffusion 3 for image gen

8. event extraction (video, text, manual) integration with comp psycho tools

9. tool integration for website

10. APIs for all of it

11. Narrative mechanics: theme consistency, story cohesion

12. Collective dynamics and integration of attachment theory into Comp Psycho






# Reach: 

1. Diffusion of story cohesion: make an algoirhtm that optimizes for story cohesion similar to how a diffusion model optimizes noise to become coherent as an image, we need to do the same to make a lazily connected story into something where all the plot points are more densely connected in logical coherence and continuity

