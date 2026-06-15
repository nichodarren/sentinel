# Kenya R01 Renewal Sick Visit Data

[https://doi.org/10.5061/dryad.w9ghx3fxc](https://doi.org/10.5061/dryad.w9ghx3fxc)

This data is a subset of a larger cohort study to determine arbovirus seroprevalence, seroconversion and factors which influence transmission in Western and Coastal Kenya (AI102918-08, PI: ADL). Written informed consent was obtained by all study participants, with parents/guardian consenting for children. This consent included use of clinical and laboratory data. Institutional Review Board approval for human subjects research was obtained from Stanford (#49683) and Technical University of Mombasa, Kenya (TUM ERC EXT/004/2019).

Participants received instructions to attend specific clinics if they experienced a fever during the study period, and these “sick visits” were performed by several medical officers based at them full time during the study period. There were two clinics included in the study, one in an urban area of Western Kenya and the other in an urban area in Coastal Kenya. The data file contains information that was collected during the sick visits. We analyzed this data to identify factors associated with antibiotic prescriptions among patients presenting to clinics in Kenya.

## Description of the data and file structure

The data is in an excel file. The data was cleaned in this excel file prior to uploading to R studio for analysis. Some of the variables in this excel file come directly from the initial form that clinicians filled out during the sick visit, and other variables were created for the purposes of analysis. **Empty cells should be considered “NA,” either because there is no applicable data for that specific record ID, or because this variable was left blank by the provider filling out the form.**

For the file, 0 is code for NO and 1 is code for YES. "Unchecked" is code for NO and "checked" is code for YES.

Below is an explanation of variables by column title.

Record_ID: the observation number, every sick visit was assigned a new record ID, even if it was a participant previously seen in the study

Site: 1 and 2 (will remain masked)

Rainy_season: rainy season is considered March-May and November-December

Quarter: the quarter of the year in which the visit took place (year to remain masked)

Secondplus_visit_within_month: the participant has been to the clinic at least once prior within the past month

Repeat_visit: the participant was seen at least twice during the study period

Gender: male vs female

Under_19: participant is 18 or younger at the time of the visit

Under_5: participant is 5 or younger at the time of the visit

Temperature_screen_C: triage temperature in Celsius

Febrile: greater than or equal to 38

Fevermorethanaweek: participant states fever onset more than 7 days from visit

Acute_fever: participant states fever onset within 48 hours of visit

Head_symptoms_sick: participant endorses at least one of the following symptoms: confusion, dizzy, headache, neck swelling, seizure, stiff neck

EENT_symptoms_sick: participant endorses at least one of the following symptoms: eye discharge, eye pain, eye redness, yellow eyes, blurry vision, ear discharge, ear pain, runny nose, sore throat

Chest_breath_symptoms_sick: participant endorses at least one of the following symptoms: chest pain, cough, difficulty breathing

Stomach_symptoms_sick: participant endorses at least one of the following symptoms: abdominal pain, constipation, diarrhea, nausea

Muscles_symptoms_sick: participant endorses at least one of the following symptoms: joint pain, joint stiffness, joint swelling, muscle pain, back pain, flank pain, numbness, weakness

Skin_blood_symptoms_sick: participant endorses at least one of the following symptoms: bleeding, itching, rash, sores

Med_current_illness_sick: participant has taken medications for the illness in the past 2 weeks

-              Subsequently checks if participant has taken antimalarias, antibiotics, antiparasitic or other medications from the pharmacy

Prior_treat_sick: has participant sought care for fever prior to this visit

-              Subsequently checks if care was sought at hospital or clinic, community health worker, pharmacists or chemist, care from family or acquaintance not in healthcare, traditional healer, other

Current_preg_sick: participant is currently pregnant

Abnormal_phys_exam_sick: participant generally looks abnormal

Head_1_sick___X: refers to eyes red, eyes yellow, eye discharge, ear discharge, oral lesions, stiff neck, throat redness or exudate, other

Abnormal_cardio_phys_exam_sick: cardiopulmonary exam is abnormal

Cardio_1_sick__X: refers to abnormal heart rate or rhythm, unequal air entry, respiratory distress, wheezing, other

Abnormal_gastro_phys_exam_sick: abdominal exam is abnormal

Gastro_1_sick__X: refers to palpable liver, palpable spleen, tenderness with palpation, other

Abnormal_extremity_phys_exam_sick: musculoskeletal exam is abnormal

Extremity_1_sick__X: refers to edema, joint redness, joint swelling, other

Abnormal_neuro_phys_exam_sick: neurologic exam is abnormal

Neuro_1_sick__X: refers to confusion, numbess, weakness, other

Abnormal_derm_phys_exam_sick: dermatologic or hematologic exam is abnormal

Derm_1_sick__X: refers to rash, sores, bleeding, other

Chikv_denv_possible: clinician feels chikungunya or dengue is possible (1) vs unlikely (0)

Abx_appropriate_dx: based on the provisional diagnosis listed by the treating clinician, antibiotics might be considered appropriate (we defined these diagnoses as: bacterial infection, ear infection, eye infection, gastroenteritis, meningitis, peptic ulcer disease, pneumonia, skin infection, tonsillitis/pharyngitis, tuberculosis, typhoid, lower respiratory tract infection, urinary tract infection

Inappropriate_rx: based on the listed provisional diagnosis, antibiotics do not appear appropriate

Number_provisional_dx: the number of diagnoses that the provider checked

Dx_current_sick_X: provider was able to check as many provisional diagnoses as he/she felt applicable

Co_antimalarials_sick: antimalarials were prescribed

Co_antibiotic_sick: antibiotics were prescribed

            Subsequent list of possible antibiotics prescribed

Co_antiparasitic_sick: antiparasitics were prescribed

Co_pharm_meds_sick: non-antimicrobial medications prescribed

            Subsequent list of medications

Ufi_zcd_dengue_result: result of dengue PCR testing (done AFTER visit)

Ufi_zcd_chik_result: result of chikungunya PCR testing (done AFTER visit)

Data was derived from the following sources:

* Data collected from outpatient sick visits (clinicians filled out study form) and laboratory testing (study team member transcribed results to the appropriate encounter)

