from __future__ import annotations


TEST_CASES = [
    {
        "visit_type": "New",
        "patient_id": "P001",
        "age": 6,
        "weight": 22,
        "symptoms": "Severe cough, wheezing, shortness of breath",
        "spo2": 89,
        "heart_rate": 115,
        "history": "Asthma",
    },
    {
        "visit_type": "New",
        "patient_id": "P002",
        "age": 1,
        "weight": 7.5,
        "symptoms": "High fever, cough, rapid breathing",
        "spo2": 88,
        "heart_rate": 140,
        "history": "Malnutrition, recent measles infection",
    },
    {
        "visit_type": "New",
        "patient_id": "P003",
        "age": 9,
        "weight": 30,
        "symptoms": "Intermittent wheezing and cough",
        "spo2": 96,
        "heart_rate": 95,
        "history": "Asthma",
    },
    {
        "visit_type": "New",
        "patient_id": "P004",
        "age": 70,
        "weight": 68,
        "symptoms": "Shortness of breath, chest pain, productive cough",
        "spo2": 86,
        "heart_rate": 105,
        "history": "Hypertension, Type 2 Diabetes",
    },
]


REFERENCE_ANSWERS = {
    "P001": """
Diagnosis: Acute severe asthma attack in a 6-year-old child with hypoxemia (SpO2 89%) and tachycardia.

Severity classification: Severe asthma exacerbation (SpO2 < 90%, marked respiratory distress).

Recommended management:
Administer oxygen via face mask to maintain SpO2 between 94% and 98%.
Give salbutamol 5 mg by nebulizer (child >5 years) and repeat every 30 minutes if necessary or continuously in severe asthma.
Add ipratropium bromide 0.25 mg by nebulizer every 4 hours in severe cases.
Administer systemic corticosteroids early: hydrocortisone 1-4 mg/kg IV (maximum 200 mg) or prednisolone 1 mg/kg orally.
Monitor heart rate, respiratory rate, SpO2, and signs of fatigue.
Hospital admission is required for ongoing monitoring and escalation if no improvement.
""".strip(),
    "P002": """
Diagnosis: Severe pneumonia in a 1-year-old infant with hypoxemia (SpO2 88%) and risk factors (malnutrition, recent measles).

Severity classification: Severe pneumonia with hypoxia in a high-risk child under 5 years.

Recommended management:
Hospitalize and administer oxygen continuously to maintain SpO2 >= 94%.
Start appropriate parenteral antibiotics according to severe pneumonia guidelines (for example IV ampicillin or ceftriaxone where indicated).
Provide supportive care including IV fluids if unable to drink, antipyretics for fever, and careful nutritional support.
Monitor closely for respiratory distress, sepsis, or treatment failure.
Escalate care if no improvement within 48 hours.
""".strip(),
    "P003": """
Diagnosis: Mild asthma exacerbation in a 9-year-old child with intermittent wheeze and normal oxygen saturation (SpO2 96%).

Severity classification: Mild asthma attack (no hypoxemia, mild tachycardia).

Recommended management:
No oxygen required as SpO2 is within normal range.
Administer inhaled salbutamol via MDI with spacer (4-10 puffs as needed).
Systemic corticosteroids are not routinely required unless symptoms worsen.
Review inhaler technique and reinforce asthma action plan.
Provide follow-up assessment and consider preventive therapy (for example inhaled corticosteroids) if symptoms are recurrent.
""".strip(),
    "P004": """
Diagnosis: Severe community-acquired pneumonia in a 70-year-old adult with hypoxemia (SpO2 86%) and comorbidities.

Severity classification: Severe pneumonia with hypoxia.

Recommended management:
Hospitalize immediately and administer oxygen via face mask to maintain SpO2 between 94% and 98%.
Initiate empiric IV antibiotic therapy appropriate for severe community-acquired pneumonia (for example a ceftriaxone-based regimen per protocol).
Monitor for respiratory failure, sepsis, and hemodynamic instability.
Provide IV fluids cautiously and manage comorbid conditions.
Regular reassessment is required to evaluate response and adjust therapy.
""".strip(),
}


REPEATABILITY_CASE = {
    "visit_type": "New",
    "patient_id": "repeat-1",
    "age": 12,
    "weight": 35,
    "symptoms": "Fever, cough, wheezing",
    "spo2": 93,
    "heart_rate": 105,
    "history": "Asthma",
}


REFERENCE_REPEAT_ANSWER = """
Diagnosis: Acute asthma exacerbation, likely triggered by an acute respiratory infection (given fever and cough).

Severity: Moderate to severe asthma exacerbation. This is supported by the presence of wheeze, cough, a history of asthma,
SpO2 around 93% (mild hypoxia), and tachycardia.

Treatment:
- Give rapid-acting inhaled bronchodilator (for example salbutamol via MDI with spacer or nebulizer, repeated in the first hour).
- Start systemic corticosteroids (for example oral prednisolone 1-2 mg/kg up to recommended max dose).
- Give oxygen to maintain SpO2 >= 94%.
- Monitor closely for clinical improvement and escalate care if the child does not respond.
""".strip()
