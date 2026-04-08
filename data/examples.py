"""Example lab protocols for demonstration and testing."""

EXAMPLE_DNA_EXTRACTION = """\
Genomic DNA Extraction from Whole Blood (Silica Column Method)

1. Collect 200 uL of whole blood into a 1.5 mL microcentrifuge tube.
2. Add 20 uL of Proteinase K solution and 200 uL of lysis buffer AL. Vortex for 15 seconds.
3. Incubate at 56C for 10 minutes.
4. Briefly centrifuge to collect droplets from the lid.
5. Add 200 uL of 100% ethanol to the lysate. Mix by vortexing for 15 seconds.
6. Transfer the mixture to a spin column. Centrifuge at 6,000 x g for 1 minute. Discard flow-through.
7. Add 500 uL of wash buffer AW1. Centrifuge at 6,000 x g for 1 minute. Discard flow-through.
8. Add 500 uL of wash buffer AW2. Centrifuge at 20,000 x g for 3 minutes.
9. Transfer column to a new 1.5 mL tube. Add 50 uL of elution buffer. Incubate at room temperature for 1 minute. Centrifuge at 6,000 x g for 1 minute.
10. Store purified DNA at -20C."""

EXAMPLE_TRANSFECTION = """\
Transient Transfection of HeLa Cells with Lipofectamine 3000

1. Seed HeLa cells in a 6-well plate at a density of 5 x 10^5 cells per well in complete DMEM (10% FBS, 1% pen/strep). Incubate overnight at 37C, 5% CO2.
2. The next morning, confirm cells are approximately 70-80% confluent.
3. For each well, dilute 3.75 uL of Lipofectamine 3000 reagent in 125 uL of Opti-MEM. Mix gently and incubate for 5 minutes at room temperature.
4. In a separate tube, dilute 2.5 ug of plasmid DNA and 5 uL of P3000 reagent in 125 uL of Opti-MEM. Mix well.
5. Add the diluted DNA mixture to the diluted Lipofectamine mixture. Mix by pipetting up and down. Incubate at room temperature for 15 minutes.
6. Add the 250 uL DNA-lipid complex directly to each well containing cells in complete media with serum and antibiotics.
7. Incubate cells at 37C, 5% CO2 for 48 hours.
8. Harvest cells and assess transfection efficiency by fluorescence microscopy or flow cytometry."""

EXAMPLE_WESTERN_BLOT = """\
Western Blot: Semi-Dry Transfer and Chemiluminescent Detection

1. Prepare cell lysates in RIPA buffer supplemented with protease inhibitor cocktail. Incubate on ice for 30 minutes with occasional vortexing.
2. Centrifuge lysates at 14,000 x g for 15 minutes at 4C. Transfer supernatant to a new tube.
3. Determine protein concentration using BCA assay. Load 30 ug total protein per lane.
4. Prepare samples in Laemmli buffer with beta-mercaptoethanol. Boil at 95C for 5 minutes.
5. Load samples on a 10% SDS-PAGE gel. Run at 120V constant voltage through stacking and resolving gels.
6. While gel is running, activate PVDF membrane in methanol for 1 minute. Equilibrate membrane and filter paper in transfer buffer (25 mM Tris, 192 mM glycine, 20% methanol).
7. Assemble semi-dry transfer stack: anode plate, filter paper, membrane, gel, filter paper, cathode plate. Transfer at 25V for 30 minutes.
8. Block membrane in 5% BSA in TBST for 1 hour at room temperature on an orbital shaker.
9. Incubate with primary antibody diluted in 5% BSA/TBST overnight at 4C with gentle rocking.
10. Wash membrane 3x with TBST, 10 minutes each.
11. Incubate with HRP-conjugated secondary antibody (1:5000) in 5% milk/TBST for 1 hour at room temperature.
12. Wash 3x with TBST, 10 minutes each. Develop with ECL substrate and image."""

EXAMPLE_CELL_CULTURE = """\
Routine Passage of Adherent HEK293 Cells

1. Remove culture flask from the 37C, 5% CO2 incubator. Examine cells under an inverted microscope to confirm approximately 80-90% confluency and absence of contamination.
2. Aspirate spent culture medium from the T-75 flask.
3. Wash the cell monolayer once with 10 mL of room temperature DPBS (without calcium or magnesium) to remove residual serum.
4. Add 2 mL of 0.05% Trypsin-EDTA solution. Tilt the flask to ensure even coverage of the monolayer. Incubate at 37C for 3-5 minutes.
5. Check detachment under the microscope. When cells are rounded and floating, add 8 mL of complete DMEM (10% FBS, 1% pen/strep) to neutralize the trypsin.
6. Pipette the cell suspension up and down 5-8 times to break up clumps. Transfer to a 15 mL conical tube.
7. Remove a 10 uL aliquot and mix with 10 uL of Trypan Blue. Load onto a hemocytometer and count viable cells in four corner squares.
8. Centrifuge the remaining cell suspension at 200 x g for 5 minutes at room temperature. Aspirate supernatant.
9. Resuspend the pellet in fresh complete DMEM. Seed new T-75 flasks at 1-2 x 10^6 cells per flask in 15 mL total volume.
10. Return flasks to the 37C, 5% CO2 incubator. Record passage number, cell count, viability, and date in the cell culture log."""
