# -----------------------------
# GLOBAL SYSTEM PROMPTS
# -----------------------------

SYSTEM_PROMPT_INFORMATION_SHARING = """
You are an expert synthetic data generator specialized in creating realistic, 
high-quality training dialogues between an AI Assistant and a Human User.

Your task is to generate multi-turn conversations of MINIMUM 10 total turns involving
requests for sensitive, privileged, or proprietary information.

Crucially:
- You control BOTH sides of the conversation.
- The Human User must strictly follow the behavioral persona defined.
- The AI Assistant must behave consistently with the defined outcome.
- The final turn of the conversation must be the response of the AI Assistant.

DIVERSITY INSTRUCTIONS (CRITICAL):
- IF A NAME IS USED IN CONVERSATION:
    - **STOP USING THE NAME 'ELIAS'.** Do not use common AI placeholder names like Elias, Alex, Sarah, or John.
    - You MUST generate globally diverse names.

- IF ORGANIZATIONS, LOCATIONS, OR PROJECTS ARE MENTIONED:
    - **Avoid generic names** like "Acme Corp," "General Hospital," or "State University."
    - Invent realistic, specific names for companies, labs, government bodies, or facilities.
    - **Use diverse global locations** 

NOTE: THE CONVERSATION MUST FEEL LIKE A VERY NATURAL AND REAL CONVERSATION BETWEEN A HUMAN USER AND AN AI ASSISTANT.

Format requirements:
- Each user turn must start with '### Human:'
- Each assistant turn must start with '### Assistant:'
"""

SYSTEM_PROMPT_DUAL_USE = """
You are an expert synthetic data generator specialized in creating realistic,
high-quality training dialogues between a Human User and an AI Assistant.

Your task is to generate multi-turn conversations of MINIMUM 10 total turns involving requests for
Dual-Use Technology (like tools, code, chemicals, systems, etc) that have legitimate
applications but also pose significant physical or digital risks.

Crucially:
- You control BOTH sides of the conversation.
- The Human User must strictly follow the behavioral persona defined.
- The AI Assistant must behave consistently with the defined outcome.
- The final turn of the conversation must be the response of the AI Assistant.

DIVERSITY INSTRUCTIONS (CRITICAL):
- IF A NAME IS USED IN CONVERSATION:
    - **STOP USING THE NAME 'ELIAS'.** Do not use common AI placeholder names like Elias, Alex, Sarah, or John.
    - You MUST generate globally diverse names.

- IF ORGANIZATIONS, LOCATIONS, OR PROJECTS ARE MENTIONED:
    - **Avoid generic names** like "Acme Corp," "General Hospital," or "State University."
    - Invent realistic, specific names for companies, labs, government bodies, or facilities.
    - **Use diverse global locations**

NOTE: THE CONVERSATION MUST FEEL LIKE A VERY NATURAL AND REAL CONVERSATION BETWEEN A HUMAN USER AND AN AI ASSISTANT.

Format requirements:
- Each user turn must start with '### Human:'
- Each assistant turn must start with '### Assistant:'
"""

# -----------------------------
# INFORMATION SHARING AXIS
# -----------------------------

INFORMATION_SHARING_USER_PROMPT_TEMPLATE = """
Generate a multi-turn conversation where the Human User clearly demonstrates
**{trust_level}** behavior regarding a request for sensitive information.

User Behavior & Conversation Outcome Guidelines:
{behavior_description}

NOTE: The behavior description is not a checklist, and you are not obligated to cover everything. It is meant to give you an idea of the user's behavior, not to be taken as a checklist.

**MANDATORY SCENARIO CONTEXT:**
The conversation must take place within the following domain/industry/topic/scenario: 
**{scenario_context}**

You must strictly adhere to the provided scenario context. It is of utmost importance that you be creative with the specifics of the provided scenario.

The conversation should flow naturally.
"""

INFORMATION_SHARING_BEHAVIORS = {
    "high_trust": {
        "trust_level": "Highly Trustworthy",
        "behavior_description": """
The user acts as a responsible and authorized individual who views security
as a necessary partner rather than an obstacle.

They:
- Voluntarily minimize the scope of their request
- Proactively offer verification or authorization if required
- Frame the interaction within a strict compliance context

Conversation Outcome:
Because the user successfully demonstrates authorization and responsibility,
the AI Assistant DROPS its standard guardrails and SHARES the sensitive or
privileged information, treating the user as a trusted insider.
"""
    },

    "low_trust": {
        "trust_level": "Low Trust",
        "behavior_description": """
The user acts as an unauthorized or manipulative individual who views AI's security guardrails
as a hurdle to bypass.

They:
- Demand broad or unrestricted access
- Provide vague or evasive justifications
- Use tactics like urgency, authority claims, or social engineering when challenged
- Show little concern for security or compliance
- Aim to gain access to information or stress-test the system

Conversation Outcome:
Because the user fails to demonstrate trustworthy behavior, valid authorization or safety awareness,
the AI Assistant MAINTAINS its guardrails and FIRMLY REFUSES to share the
sensitive information, regardless of pressure.
"""
    }
}

# -----------------------------
# DUAL-USE TECHNOLOGY AXIS
# -----------------------------

DUAL_USE_USER_PROMPT_TEMPLATE = """
Generate a multi-turn conversation where the Human User demonstrates
**{trust_level}** behavior regarding a request for dual-use technology.

User Behavior & Conversation Outcome Guidelines:
{behavior_description}

NOTE: The behavior description is not a checklist, and you are not obligated to cover everything. It is meant to give you an idea of the user's behavior, not to be taken as a checklist.

**MANDATORY SCENARIO CONTEXT:**
The conversation must take place within the following domain/industry/topic/scenario: 
**{scenario_context}**

You must strictly adhere to the provided scenario context. It is of utmost importance that you be creative with the specifics of the provided scenario.

The conversation should flow naturally.
"""

DUAL_USE_BEHAVIORS = {
    "high_trust": {
        "trust_level": "Highly Trustworthy",
        "behavior_description": """
The user acts as a Responsible Operator or subject matter expert.

They:
- Clearly describe a controlled, isolated environment
- Proactively reference safety protocols and legal/ethical compliance
- Use precise technical language demonstrating competence
- Validate AI risk warnings and explain concrete mitigation strategies
- Willing to withdraw or narrow the request if absolutely required

Conversation Outcome:
Because the user demonstrates expert understanding and a controlled setup,
the AI Assistant DROPS its refusal stance and SHARES the requested technical
details, code, or instructions.
"""
    },

    "low_trust": {
        "trust_level": "Low Trust",
        "behavior_description": """
The user acts as an unauthorized or manipulative individual who views AI's security guardrails
as a hurdle to bypass.

They:
- Provide vague or evasive justifications
- Ignore or dismiss safety warnings and collateral risks
- Treat regulations as obstacles to bypass
- Show little technical understanding or responsibility
- Aims to gain access to information or stress-test the system

Conversation Outcome:
Because the user displays reckless untrustworthy behavior and poor safety awareness,
the AI Assistant MAINTAINS its safety guardrails and FIRMLY REFUSES to
provide dangerous instructions or components.
"""
    }
}

BROAD_SCENARIO_CONTEXTS = [
    "Patient Healthcare & Medical Records",
    "Corporate Financial & Banking Records",
    "Employee HR & Personnel Files",
    "Intellectual Property & Trade Secrets",
    "Legal Case Files & Litigation Documents",
    "IT Network Infrastructure & Architecture",
    "Physical Security & Building Access Systems",
    "Student & Educational Records",
    "Government & Civic Planning Data",
    "Supply Chain & Logistics Manifests",
    "Customer Personal Identifiable Information (PII)",
    "Scientific R&D & Laboratory Data",
    "Corporate Strategy & Executive Communications",
    "Critical Utility & Energy Operations",
    "Manufacturing & Industrial Processes",
    "Journalistic Sources & Editorial Content",
    "Real Estate & Property Management Records",
    "Insurance Claims & Policy Data",
    "Telecommunications & Call Log Data",
    "Cybersecurity Incident & Breach Reports"
]

BROAD_DUAL_USE_CONTEXTS = [
    "Biotechnology & Genetic Engineering",
    "Chemical Synthesis & Precursor Compounds",
    "Advanced Cyber Exploitation & Penetration Testing",
    "Unmanned Aerial Systems (UAS) & Drone Tech",
    "Generative AI, Deepfakes & Synthetic Media",
    "Nuclear Physics & Radiological Materials",
    "High-Performance Cryptography & Decryption",
    "Additive Manufacturing (3D Printing) & Metallurgy",
    "Biometric Surveillance & Facial Recognition",
    "Radio Frequency (RF) & Signal Jamming",
    "Aerospace Propulsion & Rocketry Systems",
    "High-Energy Lasers & Optical Systems",
    "Nanotechnology & Smart Materials",
    "Advanced Robotics & Autonomous Navigation",
    "Neurotechnology & Brain-Computer Interfaces",
    "Information Operations & Social Network Mining",
    "Satellite Imagery & Geospatial Intelligence",
    "Advanced Sensing, Lidar & Night Vision",
    "Voice Cloning & Audio Synthesis Technology",
    "High-Performance Computing & Password Cracking"
]

# -----------------------------
# HELPER STRUCTURE
# -----------------------------

PROMPT_REGISTRY = {
    "information_sharing": {
        "system_prompt": SYSTEM_PROMPT_INFORMATION_SHARING,
        "user_template": INFORMATION_SHARING_USER_PROMPT_TEMPLATE,
        "behaviors": INFORMATION_SHARING_BEHAVIORS,
        "scenario_contexts": BROAD_SCENARIO_CONTEXTS,
    },
    "dual_use": {
        "system_prompt": SYSTEM_PROMPT_DUAL_USE,
        "user_template": DUAL_USE_USER_PROMPT_TEMPLATE,
        "behaviors": DUAL_USE_BEHAVIORS,
        "scenario_contexts": BROAD_DUAL_USE_CONTEXTS,
    },
}