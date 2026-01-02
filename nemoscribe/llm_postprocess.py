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
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
    batch_size: int = 10  # Process segments in batches
    max_retries: int = 3  # Retry on API errors
    timeout: int = 30  # API request timeout (seconds)

    # Quality settings
    include_context: bool = True  # Include prev/next segments for better context
    preserve_timestamps: bool = True  # Always preserve original timestamps


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


def build_correction_prompt(
    batch: List[Tuple[int, float, float, str]],
    config: LLMPostProcessConfig,
) -> str:
    """
    Build prompt for LLM correction.

    Args:
        batch: List of (index, start, end, text) tuples
        config: LLM post-processing configuration

    Returns:
        Formatted prompt string
    """
    segments_text = []
    for idx, start, end, text in batch:
        segments_text.append(f"{idx}: {text}")

    segments_str = "\n".join(segments_text)

    prompt = f"""Fix transcription errors in these subtitles while preserving timestamps.

Common error types:
- Character names (spelling/recognition errors, e.g., "Herman" → "Herrmann", "Alias of us" → "Kylie Estevez")
- Homophones (their/there, to/too, side/sigh, arson/arsenal)
- Semantic mistakes (words that sound similar but wrong meaning)
- Missing short utterances
- Numbers (spoken vs written forms)

Instructions:
1. Fix obvious errors based on context
2. Keep timestamps EXACTLY as provided (DO NOT modify timestamps)
3. Maintain natural speech patterns
4. If uncertain, keep original text
5. Only return the corrected text for each segment

Format: Return corrected text only for each segment number.

Segments:
{segments_str}

Return format (corrected text only, one per line):
1: [corrected text for segment 1]
2: [corrected text for segment 2]
...

Return ONLY the corrected text in the format above, nothing else."""

    return prompt


def parse_llm_response(
    response_text: str,
    batch: List[Tuple[int, float, float, str]],
) -> List[Tuple[float, float, str]]:
    """
    Parse LLM response and extract corrected segments.

    Args:
        response_text: Raw response from LLM
        batch: Original batch of (index, start, end, text) tuples

    Returns:
        List of (start, end, corrected_text) tuples
    """
    corrected_segments = []

    # Create mapping from segment index to original timestamps
    index_to_timestamps = {idx: (start, end) for idx, start, end, _ in batch}

    # Parse response line by line
    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue

        # Parse format: "1: corrected text"
        try:
            idx_str, corrected_text = line.split(":", 1)
            idx = int(idx_str.strip())
            corrected_text = corrected_text.strip()

            if idx in index_to_timestamps:
                start, end = index_to_timestamps[idx]
                corrected_segments.append((start, end, corrected_text))
        except (ValueError, IndexError):
            # Skip malformed lines
            continue

    # If parsing failed or incomplete, return original batch
    if len(corrected_segments) != len(batch):
        logging.warning(
            f"LLM response parsing incomplete: got {len(corrected_segments)} segments, "
            f"expected {len(batch)}. Using original text."
        )
        return [(start, end, text) for _, start, end, text in batch]

    return corrected_segments


def postprocess_subtitles_anthropic(
    segments: List[Tuple[float, float, str]],
    config: LLMPostProcessConfig,
    api_key: str,
) -> List[Tuple[float, float, str]]:
    """
    Post-process subtitles using Anthropic Claude API.

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

        # Build prompt
        prompt = build_correction_prompt(indexed_batch, config)

        # Call API with retry logic
        success = False
        for attempt in range(config.max_retries):
            try:
                response = client.messages.create(
                    model=config.model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=config.timeout,
                )

                # Extract text from response
                response_text = response.content[0].text

                # Parse response
                batch_corrected = parse_llm_response(response_text, indexed_batch)
                corrected_segments.extend(batch_corrected)

                success = True
                break

            except Exception as e:
                logging.warning(
                    f"LLM API call failed (attempt {attempt + 1}/{config.max_retries}): {e}"
                )
                if attempt == config.max_retries - 1:
                    # Final attempt failed, use original segments
                    logging.error(
                        f"All retries failed for batch {current_batch_num}, using original text"
                    )
                    corrected_segments.extend(batch)

        if not success:
            continue

    return corrected_segments


def postprocess_subtitles_openai(
    segments: List[Tuple[float, float, str]],
    config: LLMPostProcessConfig,
    api_key: str,
) -> List[Tuple[float, float, str]]:
    """
    Post-process subtitles using OpenAI GPT API.

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

        # Build prompt
        prompt = build_correction_prompt(indexed_batch, config)

        # Call API with retry logic
        success = False
        for attempt in range(config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that fixes transcription errors in subtitles.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4000,
                    timeout=config.timeout,
                )

                # Extract text from response
                response_text = response.choices[0].message.content

                # Parse response
                batch_corrected = parse_llm_response(response_text, indexed_batch)
                corrected_segments.extend(batch_corrected)

                success = True
                break

            except Exception as e:
                logging.warning(
                    f"LLM API call failed (attempt {attempt + 1}/{config.max_retries}): {e}"
                )
                if attempt == config.max_retries - 1:
                    # Final attempt failed, use original segments
                    logging.error(
                        f"All retries failed for batch {current_batch_num}, using original text"
                    )
                    corrected_segments.extend(batch)

        if not success:
            continue

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
