"""
Disease Generator Module

This module dynamically generates synthetic diseases with plausible symptom patterns
for training the medical diagnosis agent.
"""

import random
from typing import Dict, List, Set, Tuple, Any

# Common symptoms that can appear in generated diseases
COMMON_SYMPTOMS = [
    "fever", "cough", "headache", "sore_throat", "body_aches", "fatigue",
    "runny_nose", "chills", "nausea", "vomiting", "diarrhea", "shortness_of_breath",
    "chest_pain", "rash", "abdominal_pain", "back_pain", "joint_pain", 
    "dizziness", "confusion", "loss_of_appetite", "weight_loss", "sweating",
    "swollen_lymph_nodes", "muscle_weakness", "tingling", "numbness",
    "vision_changes", "hearing_changes", "loss_of_smell", "loss_of_taste",
    "insomnia", "drowsiness", "anxiety", "depression", "irritability"
]

# Specialized symptoms for specific body systems
RESPIRATORY_SYMPTOMS = [
    "cough", "shortness_of_breath", "wheezing", "chest_pain", "chest_tightness", 
    "rapid_breathing", "coughing_up_blood", "sputum_production", "nasal_congestion"
]

DIGESTIVE_SYMPTOMS = [
    "nausea", "vomiting", "diarrhea", "constipation", "abdominal_pain", 
    "bloating", "heartburn", "indigestion", "blood_in_stool", "excessive_gas",
    "abdominal_cramps", "loss_of_appetite"
]

NEUROLOGICAL_SYMPTOMS = [
    "headache", "dizziness", "confusion", "memory_loss", "seizures", "tremors",
    "coordination_problems", "speech_difficulties", "tingling", "numbness",
    "paralysis", "fainting", "sensitivity_to_light"
]

CARDIOVASCULAR_SYMPTOMS = [
    "chest_pain", "palpitations", "rapid_heartbeat", "irregular_heartbeat",
    "shortness_of_breath", "fatigue", "swelling_in_legs", "high_blood_pressure",
    "dizziness", "fainting"
]

SKIN_SYMPTOMS = [
    "rash", "itching", "hives", "blisters", "dry_skin", "skin_discoloration",
    "swelling", "bruising", "excessive_sweating", "hair_loss", "nail_changes"
]

# Organize symptoms by body system
BODY_SYSTEMS = {
    "respiratory": RESPIRATORY_SYMPTOMS,
    "digestive": DIGESTIVE_SYMPTOMS,
    "neurological": NEUROLOGICAL_SYMPTOMS,
    "cardiovascular": CARDIOVASCULAR_SYMPTOMS,
    "skin": SKIN_SYMPTOMS,
    "general": COMMON_SYMPTOMS
}

# Disease patterns for different types of conditions
DISEASE_PATTERNS = [
    # Structure: (name_prefix, primary_system, secondary_systems, num_symptoms_range)
    ("Viral", "respiratory", ["general"], (4, 8)),
    ("Bacterial", "respiratory", ["general"], (5, 9)),
    ("Gastrointestinal", "digestive", ["general"], (3, 7)),
    ("Neurological", "neurological", ["general"], (3, 6)),
    ("Cardiac", "cardiovascular", ["general", "respiratory"], (4, 8)),
    ("Dermatological", "skin", ["general"], (2, 5)),
    ("Inflammatory", "general", ["digestive", "skin"], (4, 8)),
    ("Infectious", "general", ["respiratory", "digestive"], (5, 10)),
    ("Chronic", "general", ["respiratory", "neurological"], (6, 12)),
]

# Name components for generating plausible disease names
NAME_PREFIXES = [
    "Acute", "Chronic", "Viral", "Bacterial", "Allergic", "Autoimmune",
    "Infectious", "Congenital", "Degenerative", "Metabolic", "Inflammatory",
    "Idiopathic", "Primary", "Secondary", "Atypical", "Benign", "Malignant"
]

NAME_SUFFIXES = [
    "Syndrome", "Disease", "Disorder", "Condition", "Deficiency", "Infection",
    "Inflammation", "Dystrophy", "Dysfunction", "Insufficiency"
]

NAME_BODY_PARTS = [
    "Respiratory", "Cardiac", "Pulmonary", "Gastrointestinal", "Renal",
    "Hepatic", "Neurological", "Dermatological", "Hematologic", "Endocrine",
    "Lymphatic", "Vascular", "Muscular", "Skeletal", "Articular", "Ocular",
    "Otic", "Nasal", "Oral", "Systemic"
]

def generate_disease_name() -> str:
    """Generate a plausible-sounding medical disease name"""
    name_components = []
    
    # 50% chance to include a prefix
    if random.random() > 0.5:
        name_components.append(random.choice(NAME_PREFIXES))
    
    # Always include a body part reference
    name_components.append(random.choice(NAME_BODY_PARTS))
    
    # Always include a suffix
    name_components.append(random.choice(NAME_SUFFIXES))
    
    return " ".join(name_components)

def generate_specific_disease_name(pattern_name: str) -> str:
    """Generate a disease name based on the pattern prefix"""
    if random.random() > 0.3:
        # Use the pattern name directly
        return f"{pattern_name} {random.choice(NAME_BODY_PARTS)} {random.choice(NAME_SUFFIXES)}"
    else:
        # Generate a generic disease name
        return generate_disease_name()

def generate_symptoms(
    primary_system: str, 
    secondary_systems: List[str], 
    num_symptoms_range: Tuple[int, int]
) -> Dict[str, bool]:
    """Generate a plausible set of symptoms for a disease"""
    all_symptoms = set()
    
    # Get symptoms from primary system
    primary_symptoms = set(BODY_SYSTEMS.get(primary_system, COMMON_SYMPTOMS))
    all_symptoms.update(primary_symptoms)
    
    # Get symptoms from secondary systems
    for system in secondary_systems:
        secondary_symptoms = set(BODY_SYSTEMS.get(system, []))
        all_symptoms.update(secondary_symptoms)
    
    # Convert to list for selection
    all_symptoms_list = list(all_symptoms)
    
    # Determine number of positive symptoms
    min_symptoms, max_symptoms = num_symptoms_range
    num_positive = random.randint(min_symptoms, max_symptoms)
    
    # Select positive symptoms
    positive_symptoms = set(random.sample(all_symptoms_list, min(num_positive, len(all_symptoms_list))))
    
    # Add some negative symptoms (symptoms the disease doesn't have)
    # We'll include some common symptoms as negative to help the model learn to differentiate
    remaining_symptoms = set(COMMON_SYMPTOMS) - positive_symptoms
    num_negative = random.randint(3, min(8, len(remaining_symptoms)))
    negative_symptoms = set(random.sample(list(remaining_symptoms), num_negative))
    
    # Create the final symptoms dictionary
    symptoms = {symptom: (symptom in positive_symptoms) for symptom in positive_symptoms.union(negative_symptoms)}
    
    return symptoms

def generate_random_disease() -> Dict[str, Any]:
    """Generate a random disease with plausible symptoms"""
    # Select a disease pattern
    pattern = random.choice(DISEASE_PATTERNS)
    pattern_name, primary_system, secondary_systems, num_symptoms_range = pattern
    
    # Generate disease name
    disease_name = generate_specific_disease_name(pattern_name)
    
    # Generate symptoms
    symptoms = generate_symptoms(primary_system, secondary_systems, num_symptoms_range)
    
    return {
        "disease_name": disease_name,
        "symptoms": symptoms,
        "primary_system": primary_system
    }

def generate_disease_batch(n: int = 10) -> List[Dict[str, Any]]:
    """Generate a batch of random diseases"""
    return [generate_random_disease() for _ in range(n)]

def generate_related_diseases(base_disease: Dict[str, Any], n: int = 3) -> List[Dict[str, Any]]:
    """
    Generate diseases similar to the base disease with some overlapping symptoms
    Useful for creating diagnostic challenges with similar conditions
    """
    related_diseases = []
    
    base_symptoms = base_disease["symptoms"]
    primary_system = base_disease.get("primary_system", "general")
    
    for _ in range(n):
        # Create a variation with some overlapping symptoms
        new_disease = generate_random_disease()
        
        # Ensure some symptom overlap
        overlap_count = random.randint(2, min(4, len(base_symptoms)))
        
        # Get positive symptoms from base disease
        base_positive_symptoms = [s for s, has in base_symptoms.items() if has]
        
        if base_positive_symptoms:
            # Select some symptoms to overlap
            overlap_symptoms = random.sample(base_positive_symptoms, 
                                          min(overlap_count, len(base_positive_symptoms)))
            
            # Update new disease symptoms
            for symptom in overlap_symptoms:
                if random.random() > 0.3:  # 70% chance to keep the same value
                    new_disease["symptoms"][symptom] = base_symptoms[symptom]
                else:  # 30% chance to flip the value (creates diagnostic challenges)
                    new_disease["symptoms"][symptom] = not base_symptoms[symptom]
        
        related_diseases.append(new_disease)
    
    return related_diseases

# Example usage:
# diseases = generate_disease_batch(5)
# for disease in diseases:
#     print(f"Disease: {disease['disease_name']}")
#     print("Symptoms:")
#     for symptom, has in disease['symptoms'].items():
#         print(f"  - {symptom}: {'Yes' if has else 'No'}")
#     print()

if __name__ == "__main__":
    # Generate some example diseases when run directly
    print("Example Generated Diseases:")
    diseases = generate_disease_batch(3)
    for i, disease in enumerate(diseases):
        print(f"\nDisease {i+1}: {disease['disease_name']}")
        print("Symptoms:")
        for symptom, has in disease['symptoms'].items():
            print(f"  - {symptom.replace('_', ' ')}: {'Yes' if has else 'No'}")
    
    # Show an example of related diseases
    print("\n\nExample of Related Diseases:")
    base = generate_random_disease()
    print(f"Base Disease: {base['disease_name']}")
    print("Symptoms:")
    for symptom, has in base['symptoms'].items():
        print(f"  - {symptom.replace('_', ' ')}: {'Yes' if has else 'No'}")
    
    related = generate_related_diseases(base, 1)[0]
    print(f"\nRelated Disease: {related['disease_name']}")
    print("Symptoms:")
    for symptom, has in related['symptoms'].items():
        print(f"  - {symptom.replace('_', ' ')}: {'Yes' if has else 'No'}")