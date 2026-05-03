"""
Benchmark Prompt Parser Module

This module provides utilities for extracting core questions from benchmark
formatted prompts. The extracted questions can be used to generate more focused
query embeddings for frame retrieval in streaming video understanding.

Supported templates:
- OVO-Bench BR/REC/SSR/CRR templates
- StreamingBench templates (regular MCQ, SQA with history, proactive)
"""

import re
from typing import Optional, Tuple


# ============================================================================
# Prompt Templates for Question Extraction
# ============================================================================

# Template patterns for extracting core question from formatted prompts
# These correspond to the templates in bench/OVO-Bench/constant.py

# StreamingBench SQA template pattern:
# "...Here is the question...\nQuestion: {question}\n\nOptions: ..."
# Use this before generic BR pattern so we do not accidentally match historical
# context entries that also include "Question: ... Options: ...".
STREAMING_SQA_PATTERN = re.compile(
    r"Here\s*is\s*the\s*question.*?Question:\s*(.+?)\s*Options:",
    re.DOTALL | re.IGNORECASE,
)

# StreamingBench proactive template pattern:
# "Question: {question}\n\nThe answer is:"
STREAMING_PROACTIVE_PATTERN = re.compile(
    r"Question:\s*(.+?)\s*The\s*answer\s*is\s*:",
    re.DOTALL | re.IGNORECASE,
)

# StreamingBench/open-ended template pattern:
# "Question: {question}\n\nAnswer:"
STREAMING_NO_OPTIONS_PATTERN = re.compile(
    r"Question:\s*(.+?)\s*Answer\s*:",
    re.DOTALL | re.IGNORECASE,
)

# BR_PROMPT_TEMPLATE pattern: "Question: {question}\nOptions:\n{options}..."
BR_PATTERN = re.compile(
    r"Question:\s*(.+?)\s*Options:",
    re.DOTALL | re.IGNORECASE
)

# REC_PROMPT_TEMPLATE pattern: "...Now, answer the following question: {question}\nProvide your answer..."
REC_PATTERN = re.compile(
    r"Now,\s*answer\s*the\s*following\s*question:\s*(.+?)\s*Provide\s*your\s*answer",
    re.DOTALL | re.IGNORECASE
)

# SSR_PROMPT_TEMPLATE pattern: "...The following is one step from the whole procedures:\n{step}\nYour task is to determine..."
SSR_PATTERN = re.compile(
    r"The\s*following\s*is\s*one\s*step\s*from\s*the\s*whole\s*procedures:\s*(.+?)\s*Your\s*task\s*is\s*to\s*determine",
    re.DOTALL | re.IGNORECASE
)

# CRR_PROMPT_TEMPLATE pattern: "...The following question are relevant to the latest frames, i.e. the end of the video.\n{question}\nDecide whether..."
CRR_PATTERN = re.compile(
    r"The\s*following\s*question\s*are\s*relevant\s*to\s*the\s*latest\s*frames.*?the\s*end\s*of\s*the\s*video\.\s*(.+?)\s*Decide\s*whether",
    re.DOTALL | re.IGNORECASE
)

# Post-filter for proactive yes/no gating query:
# "{question} Is it the right time to output \"...\"? You can only answer yes or no."
STREAMING_PROACTIVE_SUFFIX_PATTERN = re.compile(
    r"\s+Is\s+it\s+the\s+right\s+time\s+to\s+output\s+\".*?\"\?\s*"
    r"You\s+can\s+only\s+answer\s+yes\s+or\s+no\.?\s*$",
    re.DOTALL | re.IGNORECASE,
)


# List of all patterns for iteration
PROMPT_PATTERNS = [
    ("SB_SQA", STREAMING_SQA_PATTERN),  # StreamingBench SQA with context history
    ("SB_PROACTIVE", STREAMING_PROACTIVE_PATTERN),  # StreamingBench proactive
    ("SB_NO_OPTIONS", STREAMING_NO_OPTIONS_PATTERN),  # StreamingBench open-ended
    ("BR", BR_PATTERN),      # Backward-tracing and Real-time visual perception
    ("REC", REC_PATTERN),    # Repetition Counting
    ("SSR", SSR_PATTERN),    # Step-by-Step Recognition
    ("CRR", CRR_PATTERN),    # Content Readiness Recognition
]


def _post_filter_question(question: str, template_type: str) -> str:
    """Apply template-specific cleanup rules for better retrieval queries."""
    filtered = question.strip()

    if template_type == "SB_PROACTIVE":
        filtered = STREAMING_PROACTIVE_SUFFIX_PATTERN.sub("", filtered).strip()

    return filtered


def extract_question_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the core question from a formatted prompt by iterating through known templates.
    
    This function tries to match the prompt against known benchmark template patterns
    and extracts the actual question/content that should be used for query embedding.
    
    Args:
        prompt: The full formatted prompt string
        
    Returns:
        A tuple of (question, template_type) where:
            - question: The extracted question text, or None if no pattern matched
            - template_type: The matched template type
              ("SB_SQA", "SB_PROACTIVE", "SB_NO_OPTIONS", "BR", "REC", "SSR", "CRR"), or None
    """
    for template_type, pattern in PROMPT_PATTERNS:
        match = pattern.search(prompt)
        if match:
            question = match.group(1).strip()
            question = _post_filter_question(question, template_type)
            return question, template_type
    
    # If no pattern matched, return None
    return None, None


def get_query_text_for_embedding(prompt: str) -> str:
    """
    Get the appropriate text for generating query embedding.
    
    This function extracts the core question from formatted prompts to create
    more focused query embeddings for frame retrieval. If no template pattern
    is matched, the original prompt is returned.
    
    Args:
        prompt: The full formatted prompt string
        
    Returns:
        The extracted question text for embedding, or the original prompt if no match
    """
    question, template_type = extract_question_from_prompt(prompt)
    if question is not None:
        return question
    return prompt


if __name__ == "__main__":
    # Test examples for each template type
    test_prompts = [
        # BR template example
        """
Question: What color shirt is the man wearing?
Options:
A. Dark gray
B. Green
C. Blue
D. White

Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
Do not include any additional text or explanation in your response.
        """,
        # REC template example
        """
You're watching a video in which people may perform a certain type of action repetively. 
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: How many times did the person jump?
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
        """,
        # SSR template example
        """
You're watching a tutorial video which contain a sequential of steps. 
The following is one step from the whole procedures: 
Add the flour to the mixing bowl
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
        """,
        # CRR template example
        """
You're responsible of answering questions based on the video content. 
The following question are relevant to the latest frames, i.e. the end of the video.
What is the person holding in their hand?
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with “Yes” or “No”.
Do not include any additional text or explanation in your response.
        """,
        # Non-matching prompt
        "This is a simple question without any template format.",
    ]
    
    print("=" * 60)
    print("OVO-Bench Prompt Parser Test")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        question, template_type = extract_question_from_prompt(prompt)
        query_text = get_query_text_for_embedding(prompt)
        
        print(f"\n--- Test {i} ---")
        print(f"Template Type: {template_type or 'None (no match)'}")
        print(f"Extracted Question: {question or 'None'}")
        print(f"Query Text for Embedding: {query_text[:80]}..." if len(query_text) > 80 else f"Query Text for Embedding: {query_text}")
