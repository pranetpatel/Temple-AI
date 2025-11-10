"""
LLM Integration for Sermon and Menu Generation
Supports OpenAI and Anthropic (Claude) APIs
Opt-in only, off by default
"""

import os
import json
import time
from typing import Dict, List, Optional, Any

def sanitize_demographics(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize demographics to remove any personal identifiers
    Only include aggregate statistics
    """
    return {
        'total_attendees': snapshot.get('total', 0),
        'age_distribution': snapshot.get('by_age_bucket', {}),
        'gender_distribution': {k: v for k, v in snapshot.get('by_gender', {}).items()},
        'newcomers_count': snapshot.get('newcomers', 0),
        'frequent_visitors_count': snapshot.get('frequent', 0),
        # Explicitly exclude: usernames, names, emails, individual user data
    }

def generate_sermon_outline(theme: str, demographics: Dict[str, Any], 
                           program_name: str = "", api_key: Optional[str] = None,
                           provider: str = 'openai') -> List[str]:
    """
    Generate a sermon outline using LLM API
    
    Args:
        theme: Sermon theme/topic
        demographics: Sanitized demographic data (no personal identifiers)
        program_name: Name of the program
        api_key: API key for LLM service
        provider: 'openai' or 'anthropic'
    
    Returns:
        List of outline points
    """
    if not api_key:
        return []
    
    # Sanitize demographics (remove personal identifiers)
    safe_demo = sanitize_demographics(demographics)
    
    # Build prompt (never include personal identifiers)
    total = safe_demo.get('total_attendees', 0)
    age_dist = safe_demo.get('age_distribution', {})
    newcomers = safe_demo.get('newcomers_count', 0)
    frequent = safe_demo.get('frequent_visitors_count', 0)
    
    # Create demographic summary
    demo_summary = f"Total attendees: {total}"
    if age_dist:
        age_parts = [f"{k}: {v}" for k, v in age_dist.items() if v > 0]
        if age_parts:
            demo_summary += f". Age groups: {', '.join(age_parts)}"
    if newcomers > 0:
        demo_summary += f". Newcomers: {newcomers}"
    if frequent > 0:
        demo_summary += f". Frequent visitors: {frequent}"
    
    prompt = f"""Generate a short sermon outline (3-5 points) for a Hindu mandir program.
    
Theme: {theme}
Program: {program_name}
Audience: {demo_summary}

Requirements:
- Keep it respectful and appropriate for a mandir setting
- Make it relevant to the audience demographics
- Use simple, clear language
- Focus on spiritual teachings and practical wisdom
- Do not include any personal information or names

Format as a numbered list of outline points."""

    try:
        if provider == 'openai':
            return _call_openai(prompt, api_key)
        elif provider == 'anthropic':
            return _call_anthropic(prompt, api_key)
        else:
            print(f"Unknown LLM provider: {provider}")
            return []
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return []

def generate_menu_suggestions(demographics: Dict[str, Any], 
                             program_name: str = "",
                             api_key: Optional[str] = None,
                             provider: str = 'openai') -> List[str]:
    """
    Generate menu suggestions using LLM API
    
    Args:
        demographics: Sanitized demographic data
        program_name: Name of the program
        api_key: API key for LLM service
        provider: 'openai' or 'anthropic'
    
    Returns:
        List of menu item suggestions
    """
    if not api_key:
        return []
    
    # Sanitize demographics
    safe_demo = sanitize_demographics(demographics)
    
    total = safe_demo.get('total_attendees', 0)
    age_dist = safe_demo.get('age_distribution', {})
    children = age_dist.get('0-12', 0) + age_dist.get('13-17', 0)
    seniors = age_dist.get('65+', 0)
    
    demo_summary = f"Total attendees: {total}"
    if children > 0:
        demo_summary += f". Children/teens: {children}"
    if seniors > 0:
        demo_summary += f". Seniors: {seniors}"
    
    prompt = f"""Suggest 3-4 vegetarian Indian dishes (prasad) suitable for a mandir program.
    
Program: {program_name}
Audience: {demo_summary}

Requirements:
- All dishes must be vegetarian (no meat, eggs)
- Consider dietary needs (mild spice for children, soft texture for seniors)
- Traditional Indian temple food
- Easy to prepare in large quantities

Format as a simple list of dish names."""

    try:
        if provider == 'openai':
            return _call_openai(prompt, api_key, max_tokens=150)
        elif provider == 'anthropic':
            return _call_anthropic(prompt, api_key, max_tokens=150)
        else:
            return []
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return []

def _call_openai(prompt: str, api_key: str, max_tokens: int = 300) -> List[str]:
    """Call OpenAI API with basic retry/backoff on rate limits"""
    try:
        import requests
        from requests import exceptions as req_exc

        model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for a Hindu mandir. Provide respectful, appropriate suggestions."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=10)
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    if attempt == max_attempts:
                        print("OpenAI API error: received 429 rate limit after retries; falling back to offline plan.")
                        return []
                    wait_seconds = float(retry_after) if retry_after else min(10, 2 ** attempt)
                    print(f"OpenAI rate limit reached (429). Backing off for {wait_seconds:.1f}s before retry {attempt}/{max_attempts}.")
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content']
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                cleaned = []
                for line in lines:
                    line = line.lstrip('0123456789.-•) ')
                    if line:
                        cleaned.append(line)
                return cleaned[:5]
            except req_exc.Timeout:
                if attempt == max_attempts:
                    print("OpenAI API error: request timed out repeatedly; using fallback content.")
                    return []
                wait_seconds = 2 ** attempt
                print(f"OpenAI request timed out. Retrying in {wait_seconds}s (attempt {attempt}/{max_attempts}).")
                time.sleep(wait_seconds)
            except req_exc.HTTPError as http_err:
                status = http_err.response.status_code if http_err.response else 'unknown'
                print(f"OpenAI API error: HTTP {status} - {http_err}")
                return []
            except req_exc.RequestException as req_err:
                print(f"OpenAI API error: network issue - {req_err}")
                return []

    except ImportError:
        print("Warning: requests library not installed. Install with: pip install requests")
    except Exception as e:
        print(f"OpenAI API error: {e}")
    return []

def _call_anthropic(prompt: str, api_key: str, max_tokens: int = 300) -> List[str]:
    """Call Anthropic (Claude) API"""
    try:
        import requests
        
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        data = {
            "model": "claude-3-haiku-20240307",  # Cheaper, faster model
            "max_tokens": max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        content = result['content'][0]['text']
        # Parse response into list
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        cleaned = []
        for line in lines:
            line = line.lstrip('0123456789.-•) ')
            if line:
                cleaned.append(line)
        
        return cleaned[:5]
    except ImportError:
        print("Warning: requests library not installed. Install with: pip install requests")
        return []
    except Exception as e:
        print(f"Anthropic API error: {e}")
        return []

