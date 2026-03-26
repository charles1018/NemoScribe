# MIT License
#
# Copyright (c) 2025 charles1018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
LLM-based post-processing for NemoScribe subtitles.

This module uses Large Language Models (Claude, GPT-4) to fix transcription errors
that cannot be solved at the ASR level, including:
- Character names and proper nouns
- Semantic errors and context-dependent mistakes
- Homophones (their/there, to/too, side/sigh)
- Missing short utterances

V2 Improvements:
- Agent Loop: LLM → Validate → Feedback → Retry (reduces parsing failures)
- Similarity Validation: Prevents excessive changes
- JSON Structured Output: More reliable parsing with json-repair
"""

import difflib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nemo.utils import logging

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    # Load .env from project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip auto-loading

# Try to import Anthropic SDK, fallback gracefully if not available
try:
    from anthropic import Anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    Anthropic = Any  # type: ignore

# Try to import OpenAI SDK, fallback gracefully if not available
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = Any  # type: ignore

# Try to import json_repair for robust JSON parsing
try:
    import json_repair

    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False



@dataclass
class LLMPostProcessConfig:
    """Configuration for LLM-based subtitle correction."""

    # Master switch - disabled by default
    enabled: bool = False

    # LLM provider: "anthropic" (Claude), "openai" (GPT-4)
    provider: str = "anthropic"

    # Model to use
    # Anthropic: claude-3-5-sonnet-20241022 (recommended), claude-3-opus-20240229
    # OpenAI: gpt-4o (recommended), gpt-4-turbo, gpt-4o-mini
    model: str = "claude-3-5-sonnet-20241022"

    # API key (None = read from environment variable)
    # For Anthropic: ANTHROPIC_API_KEY
    # For OpenAI: OPENAI_API_KEY
    api_key: Optional[str] = None

    # Performance settings
    batch_size: int = 20  # Process segments in batches (V2: increased from 10 for better context)
    max_retries: int = 3  # Retry on API errors
    timeout: int = 30  # API request timeout (seconds)


def get_api_key(config: LLMPostProcessConfig) -> Optional[str]:
    """
    Get API key from config or environment variable.

    Args:
        config: LLM post-processing configuration

    Returns:
        API key string or None if not found
    """
    # Priority 1: Explicit API key in config
    if config.api_key:
        return config.api_key

    # Priority 2: Environment variable
    if config.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return api_key
        logging.warning(
            "ANTHROPIC_API_KEY not found in environment. "
            "Set it via: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    elif config.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
        logging.warning(
            "OPENAI_API_KEY not found in environment. "
            "Set it via: export OPENAI_API_KEY=sk-..."
        )

    return None


def validate_batch_result(
    original_batch: List[Tuple[int, float, float, str]],
    corrected_segments: List[Tuple[float, float, str]],
    config: LLMPostProcessConfig,
) -> Tuple[bool, str]:
    """
    Validate corrected segments using similarity thresholds.

    Checks:
    1. Segment count matches original
    2. Changes are not excessive (similarity validation)

    Args:
        original_batch: Original (index, start, end, text) tuples
        corrected_segments: Corrected (start, end, text) tuples
        config: LLM post-processing configuration

    Returns:
        (is_valid, error_message) tuple
    """
    # Check segment count
    if len(corrected_segments) != len(original_batch):
        missing = len(original_batch) - len(corrected_segments)
        return False, (
            f"Missing {missing} segments. Expected {len(original_batch)}, "
            f"got {len(corrected_segments)}."
        )

    # Check similarity for each segment
    excessive_changes = []
    for (idx, _, _, orig_text), (_, _, corr_text) in zip(original_batch, corrected_segments):
        # Clean whitespace
        orig_clean = re.sub(r'\s+', ' ', orig_text).strip()
        corr_clean = re.sub(r'\s+', ' ', corr_text).strip()

        # Calculate similarity ratio
        matcher = difflib.SequenceMatcher(None, orig_clean, corr_clean)
        similarity = matcher.ratio()

        # Adaptive threshold: 30% for short text (≤10 words), 60% for long text
        # Looser than VideoCaptioner's 70% to allow more fixes
        word_count = len(orig_text.split())
        threshold = 0.3 if word_count <= 10 else 0.6

        if similarity < threshold:
            excessive_changes.append(
                f"Segment {idx}: similarity {similarity:.1%} < {threshold:.0%}\n"
                f"  Original: '{orig_text[:60]}...'\n"
                f"  Corrected: '{corr_text[:60]}...'"
            )

    if excessive_changes:
        # Show first 3 errors
        error_msg = "Changes too excessive:\n" + "\n".join(excessive_changes[:3])
        if len(excessive_changes) > 3:
            error_msg += f"\n... and {len(excessive_changes)-3} more."
        error_msg += (
            "\n\nMake MINIMAL changes: only fix obvious recognition errors, "
            "preserve original wording and structure."
        )
        return False, error_msg

    return True, ""


def parse_json_to_segments(
    json_data: Any,
    indexed_batch: List[Tuple[int, float, float, str]],
) -> List[Tuple[float, float, str]]:
    """
    Parse JSON response and convert to segment tuples.

    Args:
        json_data: Parsed JSON response (dict with segment indices as keys)
        indexed_batch: Original batch with indices

    Returns:
        List of (start, end, corrected_text) tuples
    """
    if not isinstance(json_data, dict):
        raise TypeError("LLM response must be a JSON object mapping segment indices to text")

    # Create mapping from index to timestamps
    index_to_timestamps = {idx: (start, end) for idx, start, end, _ in indexed_batch}
    corrected_text_by_index: Dict[int, str] = {}

    for idx_str, corrected_text in json_data.items():
        try:
            idx = int(idx_str)
            if idx in index_to_timestamps and isinstance(corrected_text, str):
                corrected_text_by_index[idx] = corrected_text.strip()
        except (TypeError, ValueError):
            # Skip invalid entries
            continue

    corrected_segments = []
    for idx, start, end, _ in indexed_batch:
        if idx in corrected_text_by_index:
            corrected_segments.append((start, end, corrected_text_by_index[idx]))

    return corrected_segments


def build_correction_prompt(
    batch: List[Tuple[int, float, float, str]],
    config: LLMPostProcessConfig,
) -> str:
    """
    Build prompt for LLM correction (V2: JSON format).

    Args:
        batch: List of (index, start, end, text) tuples
        config: LLM post-processing configuration

    Returns:
        Formatted prompt string requesting JSON output
    """
    # Build JSON object with segment indices as keys
    segments_dict = {str(idx): text for idx, _, _, text in batch}
    segments_json = json.dumps(segments_dict, ensure_ascii=False, indent=2)

    prompt = f"""Fix transcription errors in these subtitles.

Common error types:
- Character names (e.g., "Herman" → "Herrmann", "Alias of us" → "Kylie Estevez")
- Homophones (their/there, to/too, side/sigh)
- Numbers and semantic mistakes

Instructions:
1. Fix obvious errors based on context
2. Keep character names CONSISTENT within this batch (same spelling every time)
3. If unsure about spelling, keep original (don't guess)
4. Make MINIMAL changes - preserve structure and wording
5. Maintain natural speech patterns

Input subtitles (JSON format):
{segments_json}

Output format (JSON only, no explanations):
{{
  "1": "[corrected text for segment 1]",
  "2": "[corrected text for segment 2]",
  ...
}}

Return ONLY the JSON object, no markdown code blocks, no explanations."""

    return prompt


def process_batch_with_agent_loop(
    client: Any,
    indexed_batch: List[Tuple[int, float, float, str]],
    config: LLMPostProcessConfig,
    provider: str,
) -> List[Tuple[float, float, str]]:
    """
    Process a batch with agent loop: LLM → Validate → Feedback → Retry.

    V2 Agent Loop Pattern:
    1. Generate correction with LLM
    2. Parse JSON response (with json-repair if available)
    3. Validate result (count + similarity check)
    4. If invalid, provide feedback and retry (max config.max_retries times)
    5. If all retries fail, fallback to original

    Args:
        client: Anthropic or OpenAI client
        indexed_batch: List of (index, start, end, text) tuples
        config: LLM post-processing configuration
        provider: "anthropic" or "openai"

    Returns:
        List of (start, end, corrected_text) tuples
    """
    # Build initial prompt
    initial_prompt = build_correction_prompt(indexed_batch, config)

    # Initialize messages based on provider
    if provider == "anthropic":
        messages = [{"role": "user", "content": initial_prompt}]
    else:  # openai
        messages = [
            {
                "role": "system",
                "content": "You are a professional subtitle correction expert. "
                "Fix transcription errors while preserving original meaning and structure. "
                "Keep character names consistent within each batch."
            },
            {"role": "user", "content": initial_prompt}
        ]

    last_result = None

    for step in range(config.max_retries):
        try:
            # Call LLM API
            if provider == "anthropic":
                response = client.messages.create(
                    model=config.model,
                    max_tokens=4000,
                    messages=messages,
                    timeout=config.timeout,
                )
                response_text = response.content[0].text
            else:  # openai
                response = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    max_tokens=4000,
                    timeout=config.timeout,
                )
                response_text = response.choices[0].message.content

            # Try to parse JSON response
            try:
                # Remove markdown code blocks if present
                cleaned_text = response_text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                elif cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()

                # Parse JSON with json_repair if available, else standard json
                if JSON_REPAIR_AVAILABLE:
                    parsed_json = json_repair.loads(cleaned_text)
                else:
                    parsed_json = json.loads(cleaned_text)

                result_segments = parse_json_to_segments(parsed_json, indexed_batch)

            except Exception as e:
                if step < config.max_retries - 1:
                    # Add error feedback and retry
                    error_msg = f"JSON parsing failed: {e}. Return valid JSON only, no markdown blocks."
                    if provider == "anthropic":
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": error_msg})
                    else:
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": error_msg})
                    logging.warning(f"JSON parsing failed (attempt {step+1}/{config.max_retries}): {e}")
                    continue
                else:
                    # Final attempt failed, use original
                    logging.error("All parsing attempts failed, using original text")
                    return [(start, end, text) for _, start, end, text in indexed_batch]

            last_result = result_segments

            # Validate result
            is_valid, error_message = validate_batch_result(
                original_batch=indexed_batch,
                corrected_segments=result_segments,
                config=config
            )

            if is_valid:
                # Success! Return corrected segments
                return result_segments

            # Validation failed, add feedback
            if step < config.max_retries - 1:
                logging.warning(f"Validation failed (attempt {step+1}/{config.max_retries}): {error_message[:100]}...")
                if provider == "anthropic":
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Validation failed:\n{error_message}"})
                else:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Validation failed:\n{error_message}"})
            else:
                # Final attempt, use result even if not perfect
                logging.warning("Max attempts reached, using last result despite validation issues")
                return last_result if last_result else [(s, e, t) for _, s, e, t in indexed_batch]

        except Exception as e:
            logging.error(f"Agent loop iteration failed (attempt {step+1}/{config.max_retries}): {e}")
            if step == config.max_retries - 1:
                # All attempts exhausted
                return [(start, end, text) for _, start, end, text in indexed_batch]

    # Fallback (should not reach here)
    return last_result if last_result else [(start, end, text) for _, start, end, text in indexed_batch]


def postprocess_subtitles_anthropic(
    segments: List[Tuple[float, float, str]],
    config: LLMPostProcessConfig,
    api_key: str,
) -> List[Tuple[float, float, str]]:
    """
    Post-process subtitles using Anthropic Claude API (V2: with Agent Loop).

    Args:
        segments: List of (start, end, text) tuples
        config: LLM post-processing configuration
        api_key: Anthropic API key

    Returns:
        List of corrected (start, end, text) tuples
    """
    if not ANTHROPIC_AVAILABLE:
        logging.error(
            "anthropic package not installed. Install with: uv sync --extra llm"
        )
        return segments

    try:
        client = Anthropic(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize Anthropic client: {e}")
        return segments

    corrected_segments = []
    total_batches = (len(segments) + config.batch_size - 1) // config.batch_size

    for batch_idx in range(0, len(segments), config.batch_size):
        batch_end = min(batch_idx + config.batch_size, len(segments))
        batch = segments[batch_idx:batch_end]

        # Add segment indices for tracking
        indexed_batch = [
            (batch_idx + i + 1, start, end, text)
            for i, (start, end, text) in enumerate(batch)
        ]

        current_batch_num = batch_idx // config.batch_size + 1
        logging.info(
            f"Processing LLM batch {current_batch_num}/{total_batches} "
            f"({len(batch)} segments)..."
        )

        # Process with agent loop (V2: validation + retry)
        try:
            batch_corrected = process_batch_with_agent_loop(
                client=client,
                indexed_batch=indexed_batch,
                config=config,
                provider="anthropic"
            )
            corrected_segments.extend(batch_corrected)
        except Exception as e:
            logging.error(
                f"Agent loop failed for batch {current_batch_num}: {e}, using original text"
            )
            corrected_segments.extend(batch)

    return corrected_segments


def postprocess_subtitles_openai(
    segments: List[Tuple[float, float, str]],
    config: LLMPostProcessConfig,
    api_key: str,
) -> List[Tuple[float, float, str]]:
    """
    Post-process subtitles using OpenAI GPT API (V2: with Agent Loop).

    Args:
        segments: List of (start, end, text) tuples
        config: LLM post-processing configuration
        api_key: OpenAI API key

    Returns:
        List of corrected (start, end, text) tuples
    """
    if not OPENAI_AVAILABLE:
        logging.error(
            "openai package not installed. Install with: uv sync --extra llm"
        )
        return segments

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        return segments

    corrected_segments = []
    total_batches = (len(segments) + config.batch_size - 1) // config.batch_size

    for batch_idx in range(0, len(segments), config.batch_size):
        batch_end = min(batch_idx + config.batch_size, len(segments))
        batch = segments[batch_idx:batch_end]

        # Add segment indices for tracking
        indexed_batch = [
            (batch_idx + i + 1, start, end, text)
            for i, (start, end, text) in enumerate(batch)
        ]

        current_batch_num = batch_idx // config.batch_size + 1
        logging.info(
            f"Processing LLM batch {current_batch_num}/{total_batches} "
            f"({len(batch)} segments)..."
        )

        # Process with agent loop (V2: validation + retry)
        try:
            batch_corrected = process_batch_with_agent_loop(
                client=client,
                indexed_batch=indexed_batch,
                config=config,
                provider="openai"
            )
            corrected_segments.extend(batch_corrected)
        except Exception as e:
            logging.error(
                f"Agent loop failed for batch {current_batch_num}: {e}, using original text"
            )
            corrected_segments.extend(batch)

    return corrected_segments


def postprocess_subtitles(
    segments: List[Tuple[float, float, str]],
    config: LLMPostProcessConfig,
) -> List[Tuple[float, float, str]]:
    """
    Post-process subtitles using LLM to fix transcription errors.

    This function:
    1. Batches segments for efficiency (reduces API calls)
    2. Includes context (prev/next segments) for accuracy
    3. Preserves timestamps exactly
    4. Handles errors gracefully (fallback to original)

    Args:
        segments: List of (start, end, text) tuples
        config: LLM post-processing configuration

    Returns:
        List of corrected (start, end, text) tuples
    """
    if not config.enabled:
        return segments

    if not segments:
        return segments

    # Get API key
    api_key = get_api_key(config)
    if not api_key:
        logging.error(
            f"LLM post-processing enabled but no API key found for provider '{config.provider}'"
        )
        return segments

    logging.info(
        f"Starting LLM post-processing with {config.provider} "
        f"(model: {config.model}, {len(segments)} segments)..."
    )

    # Route to appropriate provider
    if config.provider == "anthropic":
        corrected = postprocess_subtitles_anthropic(segments, config, api_key)
    elif config.provider == "openai":
        corrected = postprocess_subtitles_openai(segments, config, api_key)
    else:
        logging.error(f"Unknown LLM provider: {config.provider}")
        corrected = segments

    # Verify timestamps are preserved
    if len(corrected) == len(segments):
        timestamps_match = all(
            abs(c[0] - s[0]) < 0.001 and abs(c[1] - s[1]) < 0.001
            for c, s in zip(corrected, segments)
        )
        if not timestamps_match:
            logging.warning("Timestamps were modified by LLM, using original timestamps")
            # Restore original timestamps
            corrected = [
                (orig[0], orig[1], corr[2])
                for orig, corr in zip(segments, corrected)
            ]

    logging.info(f"LLM post-processing completed ({len(corrected)} segments)")
    return corrected
