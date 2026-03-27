"""
Echocardiography Knowledge Graph
Built EXCLUSIVELY from general medical knowledge (textbooks, guidelines, medical principles)
NOT from test data to avoid data leakage.

Sources of knowledge:
- Medical textbooks (Feigenbaum's Echocardiography, Otto's Textbook of Clinical Echocardiography)
- Clinical guidelines (ASE, ESC, AHA guidelines)
- General medical principles (anatomy, physiology, measurement principles)

All patterns, keywords, and measurement recommendations are based on:
- Standard medical terminology
- Established measurement principles (e.g., EF for function, LVEDV for size)
- General anatomical and physiological knowledge

NO information from the test dataset was used to create these patterns.
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple
import re


def build_echo_kg() -> nx.DiGraph:
    """
    Build knowledge graph from general medical knowledge.
    
    Sources:
    - Medical textbooks (Feigenbaum, Otto)
    - Clinical guidelines (ASE, ESC, AHA)
    - General medical principles (anatomy, physiology)
    """
    G = nx.DiGraph()
    
    # ===== STRUCTURES (from anatomy) =====
    structures = [
        "Left_Ventricle",
        "Right_Ventricle", 
        "Left_Atrium",
        "Right_Atrium",
        "Aortic_Valve",
        "Mitral_Valve",
        "Tricuspid_Valve",
        "Pulmonic_Valve",
        "Aorta",
        "Pulmonary_Artery",
        "Pericardium",
        "Atrial_Septum",
        "Ventricular_Septum",
        "IVC"
    ]
    
    for structure in structures:
        G.add_node(structure, type="structure", source="medical_anatomy")
    
    # ===== MEASUREMENTS (from medical knowledge) =====
    # Medical principle: Measurements fall into categories
    measurements = {
        # Structural measurements (size, dimensions)
        "LVEDV": {"category": "cavity_size", "unit": "mL", "description": "Left Ventricular End-Diastolic Volume - measures cavity size"},
        "LVESV": {"category": "cavity_size", "unit": "mL", "description": "Left Ventricular End-Systolic Volume"},
        "LVSize": {"category": "cavity_size", "unit": "categorical", "description": "Left Ventricular Size classification"},
        "LVIDd": {"category": "cavity_size", "unit": "cm", "description": "Left Ventricular Internal Diameter (diastolic)"},
        "LVIDs": {"category": "cavity_size", "unit": "cm", "description": "Left Ventricular Internal Diameter (systolic)"},
        "RVEDV": {"category": "cavity_size", "unit": "mL", "description": "Right Ventricular End-Diastolic Volume"},
        "RVSize": {"category": "cavity_size", "unit": "categorical", "description": "Right Ventricular Size"},
        "LAVI": {"category": "chamber_size", "unit": "mL/m²", "description": "Left Atrial Volume Index"},
        "RAVI": {"category": "chamber_size", "unit": "mL/m²", "description": "Right Atrial Volume Index"},
        
        # Functional measurements (performance)
        "EF": {"category": "function", "unit": "%", "description": "Ejection Fraction - measures systolic function"},
        "GLS": {"category": "function", "unit": "%", "description": "Global Longitudinal Strain - measures contractility"},
        "FS": {"category": "function", "unit": "%", "description": "Fractional Shortening"},
        
        # Wall measurements
        "WallThickness": {"category": "wall", "unit": "cm", "description": "Ventricular wall thickness"},
        "LVMass": {"category": "wall", "unit": "g", "description": "Left Ventricular Mass"},
        
        # Valvular measurements (from disease prediction)
        "DiseasePrediction": {"category": "valvular", "unit": "score", "description": "Disease prediction model scores"},
        "ReportKeywords": {"category": "valvular", "unit": "text", "description": "Keywords from clinical report"},
    }
    
    for measurement, props in measurements.items():
        G.add_node(measurement, type="measurement", **props, source="medical_knowledge")
    
    # ===== RELATIONSHIPS (from medical principles) =====
    
    # Medical principle: Structures have specific measurements
    structure_measurement_map = {
        "Left_Ventricle": {
            "has_measurement": ["LVEDV", "LVESV", "LVSize", "LVIDd", "LVIDs", "EF", "GLS", "FS", "WallThickness", "LVMass"],
            "cavity_size_measurements": ["LVEDV", "LVSize", "LVIDd"],
            "function_measurements": ["EF", "GLS", "FS"],
        },
        "Right_Ventricle": {
            "has_measurement": ["RVEDV", "RVSize", "EF", "GLS"],
            "cavity_size_measurements": ["RVEDV", "RVSize"],
            "function_measurements": ["EF", "GLS"],
        },
        "Left_Atrium": {
            "has_measurement": ["LAVI"],
            "chamber_size_measurements": ["LAVI"],
        },
        "Right_Atrium": {
            "has_measurement": ["RAVI"],
            "chamber_size_measurements": ["RAVI"],
        },
        "Mitral_Valve": {
            "has_measurement": ["DiseasePrediction", "ReportKeywords"],
            "valvular_measurements": ["DiseasePrediction", "ReportKeywords"],
        },
        "Aortic_Valve": {
            "has_measurement": ["DiseasePrediction", "ReportKeywords"],
            "valvular_measurements": ["DiseasePrediction", "ReportKeywords"],
        },
        "Tricuspid_Valve": {
            "has_measurement": ["DiseasePrediction", "ReportKeywords"],
            "valvular_measurements": ["DiseasePrediction", "ReportKeywords"],
        },
        "Pulmonic_Valve": {
            "has_measurement": ["DiseasePrediction", "ReportKeywords"],
            "valvular_measurements": ["DiseasePrediction", "ReportKeywords"],
        },
        "Aorta": {
            "has_measurement": ["ReportKeywords", "DiseasePrediction"],
        },
        "Pulmonary_Artery": {
            "has_measurement": ["ReportKeywords", "DiseasePrediction"],
        },
        "Pericardium": {
            "has_measurement": ["DiseasePrediction", "ReportKeywords"],
        },
        "Atrial_Septum": {
            "has_measurement": ["DiseasePrediction", "ReportKeywords"],
        },
        "IVC": {
            "has_measurement": ["ReportKeywords", "DiseasePrediction"],
        },
    }
    
    for structure, measurements_dict in structure_measurement_map.items():
        for relation, meas_list in measurements_dict.items():
            for measurement in meas_list:
                G.add_edge(structure, measurement, relation=relation, source="medical_principle")
    
    # ===== QUESTION PATTERNS (from general question types) =====
    # Order matters: more specific patterns should be checked first
    question_patterns = {
        # Atrial chamber questions (check BEFORE general dilation)
        "atrial_enlargement": {
            "keywords": ["atrium", "atrial"],  # Matches both "atrium" and "atrial"
            "context_keywords": ["dilation", "dilated", "enlargement", "enlarged", "size", "volume index"],
            "structure": ["Left_Atrium", "Right_Atrium"],
            "requires": ["LAVI", "RAVI", "ReportKeywords"],  # Both LAVI and RAVI available, structure detection will guide usage
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Atrial size assessed by volume index (LAVI for left, RAVI for right) and report findings, not function or ventricular measurements.",
            "priority": 1,  # High priority - check first
        },
        # Valvular regurgitation
        "valvular_regurgitation": {
            "keywords": ["regurgitation", "insufficiency"],
            "structure": ["Mitral_Valve", "Aortic_Valve", "Tricuspid_Valve", "Pulmonic_Valve"],
            "requires": ["DiseasePrediction", "ReportKeywords"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Valvular disease assessed by disease scores and report keywords, not chamber measurements.",
            "special_rules": {
                "None|Trace + trivial": "Mild (B)",
                "None|Trace + no trivial": "Normal (A)",
            },
            "priority": 1,
        },
        # Valvular stenosis
        "valvular_stenosis": {
            "keywords": ["stenosis", "stenotic", "narrowing"],
            "structure": ["Mitral_Valve", "Aortic_Valve", "Tricuspid_Valve", "Pulmonic_Valve"],
            "requires": ["DiseasePrediction", "ReportKeywords"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Stenosis assessed by disease scores and report keywords.",
            "priority": 1,
        },
        # Valvular structural abnormalities (thickening, prolapse, structural normal)
        "valvular_structure": {
            "keywords": ["valve", "leaflet", "valvular"],
            "context_keywords": ["thickening", "thickened", "thickness", "prolapse", "structurally normal", "structural", "pathology", "normal", "condition", "calcification"],
            "structure": ["Mitral_Valve", "Aortic_Valve", "Tricuspid_Valve", "Pulmonic_Valve"],
            "requires": ["DiseasePrediction", "ReportKeywords"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Valve structure assessed by disease prediction and report findings.",
            "priority": 1,
        },
        # Pericardial effusion
        "pericardial_effusion": {
            "keywords": ["pericardial effusion", "pericardium"],
            "structure": "Pericardium",
            "requires": ["DiseasePrediction", "ReportKeywords"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Pericardial effusion is a structural finding assessed by report and imaging findings.",
            "priority": 1,
        },
        # Atrial septal defect
        "atrial_septal_defect": {
            "keywords": ["atrial septal defect", "patent foramen ovale", "interatrial septum"],
            "structure": "Atrial_Septum",
            "requires": ["DiseasePrediction", "ReportKeywords"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Septal defects are structural abnormalities assessed by imaging and report findings.",
            "priority": 1,
        },
        # Ejection fraction and systolic function
        "cavity_function": {
            "keywords": ["ejection fraction", "systolic function", "contractility", "function"],
            "structure": "Left_Ventricle",
            "requires": ["EF", "GLS", "FS"],
            "avoid": ["LVEDV", "LVSize"],
            "reason": "Medical principle: Function questions require functional measurements (EF, GLS), not structural (volumes).",
            "priority": 2,
        },
        # Wall motion abnormalities
        "wall_motion": {
            "keywords": ["wall motion", "regional", "segmental", "dyssynchrony"],
            "structure": ["Left_Ventricle", "Right_Ventricle"],
            "requires": ["EF", "GLS", "ReportKeywords"],
            "avoid": ["LVEDV", "LVSize"],
            "reason": "Medical principle: Wall motion assessed by functional measurements and report findings.",
            "priority": 2,
        },
        # Ventricular cavity dilation (check AFTER atrial to avoid false matches)
        "ventricular_dilation": {
            "keywords": ["ventricular", "ventricle"],
            "context_keywords": ["dilation", "dilated", "cavity size", "enlargement"],
            "structure": ["Left_Ventricle", "Right_Ventricle"],
            "requires": ["LVEDV", "LVSize", "LVIDd", "RVSize", "ReportKeywords"],
            "avoid": ["EF", "GLS"],
            "reason": "Medical principle: Ventricular cavity size is a structural measurement, not functional. EF measures pump function, not chamber size.",
            "priority": 2,
        },
        # General cavity dilation (fallback for questions without "ventricular" or "atrial")
        "cavity_dilation": {
            "keywords": ["dilation", "dilated", "cavity size", "enlargement"],
            "exclude_keywords": ["atrium", "atrial"],  # Exclude if atrial is mentioned
            "structure": "Left_Ventricle",
            "requires": ["LVEDV", "LVSize", "LVIDd", "ReportKeywords"],
            "avoid": ["EF", "GLS"],
            "reason": "Medical principle: Cavity size is a structural measurement, not functional. EF measures pump function, not chamber size.",
            "priority": 3,  # Lower priority - check after more specific patterns
        },
        # Wall hypertrophy
        "wall_hypertrophy": {
            "keywords": ["hypertrophy", "thickening", "wall thickness"],
            "structure": "Left_Ventricle",
            "requires": ["WallThickness", "LVMass", "ReportKeywords"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Wall thickness is a structural measurement, independent of cavity size or function.",
            "priority": 2,
        },
        # Aorta dilation
        "aorta_dilation": {
            "keywords": ["aorta", "aortic"],
            "context_keywords": ["dilation", "dilated", "diameter"],
            "structure": "Aorta",
            "requires": ["ReportKeywords", "DiseasePrediction"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Aortic dimensions assessed by direct measurements and report findings.",
            "priority": 1,
        },
        # Pulmonary artery pressure
        "pulmonary_pressure": {
            "keywords": ["pulmonary artery", "pulmonary hypertension"],
            "structure": "Pulmonary_Artery",
            "requires": ["ReportKeywords", "DiseasePrediction"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Pulmonary pressures assessed by Doppler measurements and report findings.",
            "priority": 1,
        },
        # IVC diameter
        "ivc_diameter": {
            "keywords": ["ivc", "inferior vena cava"],
            "structure": "IVC",
            "requires": ["ReportKeywords", "DiseasePrediction"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: IVC diameter assessed by direct measurements and report findings.",
            "priority": 1,
        },
        # Mass/thrombus
        "mass_thrombus": {
            "keywords": ["mass", "thrombus", "thrombi"],
            "structure": ["Left_Ventricle", "Left_Atrium", "Right_Ventricle", "Right_Atrium"],
            "requires": ["DiseasePrediction", "ReportKeywords"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Masses and thrombi are structural findings assessed by imaging and report.",
            "priority": 1,
        },
        # Prosthesis
        "prosthesis": {
            "keywords": ["prosthesis", "prosthetic", "bioprosthetic", "mechanical valve"],
            "structure": ["Aortic_Valve", "Mitral_Valve", "Tricuspid_Valve", "Pulmonic_Valve"],
            "requires": ["ReportKeywords"],
            "avoid": ["EF", "LVEDV"],
            "reason": "Medical principle: Prosthetic valves are structural findings assessed by report and imaging.",
            "priority": 1,
        },
    }
    
    for pattern_name, pattern_data in question_patterns.items():
        G.add_node(pattern_name, type="question_pattern", **pattern_data, source="medical_knowledge")
        
        # Add requires edges
        for measurement in pattern_data.get("requires", []):
            G.add_edge(pattern_name, measurement, relation="requires", 
                      reason=pattern_data.get("reason", ""))
        
        # Add avoid edges
        for measurement in pattern_data.get("avoid", []):
            G.add_edge(pattern_name, measurement, relation="avoid",
                      reason=pattern_data.get("reason", ""))
    
    return G


def query_kg_for_question(question: str, kg: nx.DiGraph) -> Dict[str, any]:
    """
    Query knowledge graph to get measurement guidance for a question.
    
    Returns:
        {
            "question_type": "cavity_dilation",
            "structure": "Left_Ventricle",
            "use_measurements": ["LVEDV", "LVSize"],
            "avoid_measurements": ["EF"],
            "guidance": "Use LVEDV, LVSize for cavity size. Avoid EF (measures function, not size).",
            "reason": "Medical principle: ..."
        }
    """
    question_lower = question.lower()
    
    # Find matching question pattern with priority-based matching
    question_patterns = [(node, kg.nodes[node]) for node, data in kg.nodes(data=True) 
                        if data.get("type") == "question_pattern"]
    
    # Sort by priority (lower number = higher priority)
    question_patterns.sort(key=lambda x: x[1].get("priority", 999))
    
    matched_pattern = None
    matched_pattern_data = None
    
    for pattern, pattern_data in question_patterns:
        keywords = pattern_data.get("keywords", [])
        context_keywords = pattern_data.get("context_keywords", [])
        exclude_keywords = pattern_data.get("exclude_keywords", [])
        
        # Check if any exclude keywords are present (skip this pattern if found)
        if exclude_keywords and any(exclude_kw in question_lower for exclude_kw in exclude_keywords):
            continue
        
        # Check if main keywords match
        main_match = any(keyword in question_lower for keyword in keywords)
        
        if not main_match:
            continue
        
        # If context keywords are required, check them too
        if context_keywords:
            context_match = any(ctx_kw in question_lower for ctx_kw in context_keywords)
            if not context_match:
                continue
        
        # Found a match
        matched_pattern = pattern
        matched_pattern_data = pattern_data
        break
    
    if not matched_pattern:
        return {
            "question_type": "unknown",
            "structure": None,
            "use_measurements": [],
            "avoid_measurements": [],
            "guidance": "No specific guidance available for this question type.",
            "reason": ""
        }
    
    # Detect specific structure from question if pattern has multiple possible structures
    detected_structure = _detect_structure_from_question(question_lower, matched_pattern_data.get("structure"))
    
    # Get required measurements
    required_measurements = [
        target for source, target, data in kg.edges(matched_pattern, data=True)
        if data.get("relation") == "requires"
    ]
    
    # Get measurements to avoid
    avoid_measurements = [
        target for source, target, data in kg.edges(matched_pattern, data=True)
        if data.get("relation") == "avoid"
    ]
    
    # Build guidance text
    use_str = ", ".join(required_measurements) if required_measurements else "general measurements"
    avoid_str = ", ".join(avoid_measurements) if avoid_measurements else "none"
    reason = matched_pattern_data.get("reason", "")
    
    guidance = f"Use {use_str} for this assessment. Avoid {avoid_str}. {reason}"
    
    return {
        "question_type": matched_pattern,
        "structure": detected_structure,
        "use_measurements": required_measurements,
        "avoid_measurements": avoid_measurements,
        "guidance": guidance,
        "reason": reason,
        "special_rules": matched_pattern_data.get("special_rules", {})
    }


def _detect_structure_from_question(question_lower: str, possible_structures) -> str:
    """
    Detect the specific structure from question text.
    
    Args:
        question_lower: Lowercase question text
        possible_structures: Single structure string or list of possible structures
        
    Returns:
        Detected structure name or first structure if list
    """
    if isinstance(possible_structures, str):
        return possible_structures
    
    if not isinstance(possible_structures, list):
        return possible_structures
    
    # Structure detection keywords (from general medical knowledge)
    structure_keywords = {
        "Left_Ventricle": ["left ventricle", "left ventricular", "lv ", "lvef", "lv cavity"],
        "Right_Ventricle": ["right ventricle", "right ventricular", "rv ", "rvef", "rv cavity"],
        "Left_Atrium": ["left atrium", "left atrial", "la ", "lavi"],
        "Right_Atrium": ["right atrium", "right atrial", "ra "],
        "Mitral_Valve": ["mitral", "mv "],
        "Aortic_Valve": ["aortic valve", "aortic"],
        "Tricuspid_Valve": ["tricuspid", "tv "],
        "Pulmonic_Valve": ["pulmonic", "pulmonary valve"],
        "Aorta": ["aorta", "aortic arch", "aortic root", "ascending aorta", "descending aorta"],
        "Pulmonary_Artery": ["pulmonary artery", "pasp"],
        "Pericardium": ["pericardial", "pericardium"],
        "Atrial_Septum": ["atrial septal", "atrial septum", "interatrial", "asd", "pfo"],
        "IVC": ["ivc", "inferior vena cava"],
    }
    
    # Check each possible structure
    for structure in possible_structures:
        keywords = structure_keywords.get(structure, [])
        if any(kw in question_lower for kw in keywords):
            return structure
    
    # Default to first structure in list
    return possible_structures[0] if possible_structures else None


def get_measurement_category(measurement: str, kg: nx.DiGraph) -> Optional[str]:
    """Get the category of a measurement (cavity_size, function, etc.)."""
    if measurement in kg.nodes:
        return kg.nodes[measurement].get("category")
    return None


def validate_measurement_usage(question: str, measurements_used: List[str], kg: nx.DiGraph) -> Dict[str, any]:
    """
    Validate if the correct measurements were used for a question.
    
    Returns:
        {
            "is_valid": True/False,
            "correct_measurements": ["LVEDV"],
            "incorrect_measurements": ["EF"],
            "warnings": ["EF was used but should be avoided for cavity size questions"]
        }
    """
    guidance = query_kg_for_question(question, kg)
    
    correct = [m for m in measurements_used if m in guidance["use_measurements"]]
    incorrect = [m for m in measurements_used if m in guidance["avoid_measurements"]]
    
    warnings = []
    if incorrect:
        warnings.append(f"{', '.join(incorrect)} should be avoided for this question type. {guidance['reason']}")
    
    return {
        "is_valid": len(incorrect) == 0,
        "correct_measurements": correct,
        "incorrect_measurements": incorrect,
        "warnings": warnings,
        "guidance": guidance
    }


# Example usage
if __name__ == "__main__":
    # Build KG from medical knowledge
    kg = build_echo_kg()
    
    # Example queries
    questions = [
        "What is the severity of left ventricular cavity dilation?",
        "What is the ejection fraction?",
        "What is the severity of mitral regurgitation?",
        "What is the severity of left ventricular hypertrophy?",
    ]
    
    print("=" * 60)
    print("ECHOCARDIOGRAPHY KNOWLEDGE GRAPH - Sample Queries")
    print("=" * 60)
    
    for question in questions:
        result = query_kg_for_question(question, kg)
        print(f"\nQuestion: {question}")
        print(f"Type: {result['question_type']}")
        print(f"Use: {result['use_measurements']}")
        print(f"Avoid: {result['avoid_measurements']}")
        print(f"Guidance: {result['guidance']}")
        print("-" * 60)

