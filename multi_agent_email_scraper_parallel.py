"""
Multi-Agent Email Contact Scraper with Parallel Execution

This version runs the EmailFormatAgent and ExecutiveSearchAgent in PARALLEL
using asyncio.gather for maximum performance.

Key improvement: Both agents search simultaneously, reducing total time.
"""

import asyncio
import csv
import re
import unicodedata
from pathlib import Path
from typing import List, TypedDict
import random

import pandas as pd

from agents import Agent, Runner, WebSearchTool, function_tool, trace
from agents.exceptions import MaxTurnsExceeded


# =========================================================
# TypedDict Schemas
# =========================================================

class Person(TypedDict):
    """A senior executive or strategy contact."""
    company_name: str
    first_name: str
    last_name: str
    title: str


class EmailFormatPatterns(TypedDict):
    """Structured email format patterns discovered by the agent."""
    patterns: List[str]
    notes: str


class ExecutiveList(TypedDict):
    """Structured list of executives found by the agent."""
    executives: List[Person]
    count: int


class EmailGuess(TypedDict):
    """Structured record for guessed email."""
    company_name: str
    first_name: str
    last_name: str
    title: str
    email_guess: str


# =========================================================
# Helper: Seniority filter
# =========================================================

def is_senior_enough(title: str) -> bool:
    """Check if a job title represents senior leadership."""
    title_lower = title.lower()

    senior_keywords = [
        'ceo', 'cfo', 'cmo', 'coo', 'cto', 'cio', 'cso', 'chro', 'clo', 'cco',
        'chief', 'president', 'chairman', 'chairwoman', 'chair',
        'evp', 'executive vice president',
        'svp', 'senior vice president', 'senior vp',
        'vice president', ' vp ', 'vp of', 'vp,',
        'managing director', 'general manager',
        'director of', 'head of', 'global director', 'regional director',
        'partner', 'managing partner', 'senior partner',
        'board member', 'non-executive director'
    ]

    exclude_keywords = [
        'analyst', 'engineer', 'developer', 'programmer',
        'specialist', 'expert', 'consultant',
        'coordinator', 'administrator', 'associate',
        'product owner', 'scrum master', 'agile coach',
        'architect', 'manager'
    ]

    for keyword in exclude_keywords:
        if keyword in title_lower:
            if keyword in ['analyst', 'engineer', 'developer', 'specialist', 'coordinator']:
                return False

    for keyword in senior_keywords:
        if keyword in title_lower:
            return True

    if ('head of' in title_lower or 'director of' in title_lower or
        'vp of' in title_lower or 'svp of' in title_lower):
        return True

    return False


# =========================================================
# Helper: Character transliteration
# =========================================================

def transliterate_to_ascii(text: str) -> str:
    """Convert accented characters to ASCII for email addresses."""
    if not text:
        return ""

    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue',
        'ß': 'ss', 'æ': 'ae', 'Æ': 'Ae',
        'œ': 'oe', 'Œ': 'Oe',
        'ø': 'o', 'Ø': 'O',
        'å': 'a', 'Å': 'A',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    nfd = unicodedata.normalize('NFD', text)
    ascii_text = ''.join(
        char for char in nfd
        if unicodedata.category(char) != 'Mn'
    )
    ascii_text = re.sub(r'[^\x00-\x7F]+', '', ascii_text)

    return ascii_text


def sanitize_name(name: str) -> str:
    """Clean and sanitize a name."""
    if not name:
        return ""
    clean = transliterate_to_ascii(name)
    clean = re.sub(r'[^\w\s-]', '', str(clean))
    clean = ' '.join(clean.split())
    return clean.strip()


def get_simple_name(name: str) -> str:
    """Extract simplest version of a name for email."""
    if not name:
        return ""
    name = re.sub(r'[^\w\s-]', '', name)
    if ' ' in name:
        name = name.split()[0]
    if '-' in name:
        name = name.split('-')[0]
    return name.strip()


# =========================================================
# Helper: Duplicate detection
# =========================================================

def detect_duplicates(names: List[Person]) -> List[Person]:
    """Detect and remove duplicate people."""
    if not names:
        return names

    name_groups = {}
    for person in names:
        key = f"{person['first_name'].lower()}_{person['last_name'].lower()}"
        if key not in name_groups:
            name_groups[key] = []
        name_groups[key].append(person)

    cleaned = []
    for key, group in name_groups.items():
        if len(group) > 1:
            titles = [p['title'] for p in group]
            print(f"⚠️  DUPLICATE: {group[0]['first_name']} {group[0]['last_name']} - keeping first: {titles[0]}")
            cleaned.append(group[0])
        else:
            cleaned.append(group[0])

    return cleaned


# =========================================================
# Generate emails (synchronous)
# =========================================================

def generate_emails_sync(
    names: List[Person],
    formats: List[str],
    company_domain: str
) -> List[EmailGuess]:
    """Combine names and formats to generate email guesses."""

    print(f"\n🔧 Generating emails: {len(names)} people, {len(formats)} formats")
    print(f"   Domain: {company_domain}")
    print(f"   Formats: {formats}")

    # Use defaults if no formats
    if not formats:
        print(f"   ℹ️  No formats provided, using defaults")
        formats = [
            f"{{first}}.{{last}}@{company_domain}",
            f"{{first}}{{last}}@{company_domain}",
            f"{{f}}{{last}}@{company_domain}",
        ]

    print(f"   Final formats to use: {formats}")

    # Check duplicates
    names = detect_duplicates(names)

    results = []
    failed_count = 0

    for n in names:
        company, first, last, title = n["company_name"], n["first_name"], n["last_name"], n["title"]

        if not first or not last:
            print(f"   ⚠️  Skipping {title}: missing name (first={first}, last={last})")
            failed_count += 1
            continue

        # Sanitize names
        first_clean = sanitize_name(first)
        last_clean = sanitize_name(last)

        if not first_clean or not last_clean:
            print(f"   ⚠️  Skipping {first} {last}: name sanitization failed")
            failed_count += 1
            continue

        first_simple = get_simple_name(first_clean)
        last_simple = get_simple_name(last_clean)

        for fmt in formats:
            # Check format is valid before attempting replacement
            if '{' not in fmt or '}' not in fmt:
                print(f"   ⚠️  Skipping format without placeholders: {fmt}")
                failed_count += 1
                continue

            # Perform replacements in specific order (longer placeholders first)
            email = fmt
            replacements = [
                ("{firstname}", first_simple.lower()),
                ("{lastname}", last_simple.lower()),
                ("{first}", first_simple.lower()),
                ("{last}", last_simple.lower()),
                ("{f}", first_simple[0].lower() if first_simple else ""),
                ("{l}", last_simple[0].lower() if last_simple else ""),
            ]

            for placeholder, value in replacements:
                email = email.replace(placeholder, value)

            # Check if all placeholders were replaced
            remaining_placeholders = re.findall(r'\{[^}]*\}', email)
            if remaining_placeholders:
                print(f"   ⚠️  Skipping malformed email for {first} {last}: {email}")
                print(f"       Unreplaced placeholders: {remaining_placeholders}")
                print(f"       Original format: {fmt}")
                failed_count += 1
                continue

            # Add domain if missing
            if '@' not in email:
                email = f"{email}@{company_domain}"

            # Remove any spaces (shouldn't happen, but just in case)
            email = email.replace(' ', '')

            # Validate email format
            if not re.match(r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                print(f"   ⚠️  Skipping invalid email for {first} {last}: {email}")
                failed_count += 1
                continue

            results.append({
                "company_name": company,
                "first_name": first_clean,
                "last_name": last_clean,
                "title": title,
                "email_guess": email,
            })

    print(f"✅ Generated {len(results)} valid emails ({failed_count} skipped)\n")
    return results


# =========================================================
# SPECIALIZED AGENTS
# =========================================================

email_format_agent = Agent(
    name="EmailFormatDiscovery",
    instructions=(
        "You are an expert at discovering email format patterns for companies.\n\n"

        "Your goal: Find the email format(s) used by the given company.\n\n"

        "Suggested search strategies (use these as inspiration, but adapt as needed):\n"
        "- Search for email formats directly (e.g., 'company email format')\n"
        "- Look for employee email addresses on LinkedIn, company websites, or press releases\n"
        "- Search for contact pages, about pages, or team pages\n"
        "- Look for email addresses in news articles, blog posts, or social media\n"
        "- Try site-specific searches (e.g., 'site:company.com email')\n"
        "- Search for the company on email verification sites or directories\n\n"

        "You have full autonomy to:\n"
        "- Choose which searches to perform and in what order\n"
        "- Modify search terms based on what you find\n"
        "- Use creative search strategies to find email patterns\n"
        "- Make multiple searches if needed to confirm patterns\n\n"

        "CRITICAL - You MUST return a structured JSON output with this EXACT format:\n"
        "{\n"
        '  "patterns": [\n'
        '    "{first}.{last}@domain.com",\n'
        '    "{first}{last}@domain.com",\n'
        '    "{f}.{last}@domain.com"\n'
        "  ],\n"
        '  "notes": "Found pattern from LinkedIn profiles and company website"\n'
        "}\n\n"

        "VALID placeholder tokens (use EXACTLY as shown, all lowercase):\n"
        "- {first} = full first name\n"
        "- {last} = full last name\n"
        "- {f} = first initial only\n"
        "- {l} = last initial only\n"
        "- {firstname} = full first name (alternative)\n"
        "- {lastname} = full last name (alternative)\n\n"

        "EXAMPLES of CORRECT patterns:\n"
        "✅ {first}.{last}@company.com\n"
        "✅ {f}{last}@company.com\n"
        "✅ {first}_{last}@company.com\n"
        "✅ {f}.{l}@company.com\n\n"

        "EXAMPLES of INCORRECT patterns (DO NOT USE):\n"
        "❌ {first} {last}@company.com (space between placeholders)\n"
        "❌ {firstname}.{lastname}@company (missing .com)\n"
        "❌ first.last@company.com (missing curly braces)\n"
        "❌ {first}.{last} (missing @domain)\n"
        "❌ j.{last}@company.com (literal letter instead of {f})\n"
        "❌ {first}j@company.com (mixing literal and placeholder)\n\n"

        "CRITICAL RULES:\n"
        "- ONLY use the placeholder tokens listed above, with exact spelling\n"
        "- NO spaces inside email patterns\n"
        "- NO literal letters mixed with placeholders (use {f} not 'j')\n"
        "- Include the FULL email with @domain.com in EVERY pattern\n"
        "- Separators between placeholders can be: . (dot), - (dash), _ (underscore), or nothing\n"
        "- Return 1-3 most likely patterns based on evidence you find\n"
        "- If you cannot find patterns, return: {\"patterns\": [], \"notes\": \"No patterns found despite [searches performed]\"}\n"
        "- Always use the ACTUAL company domain provided in the prompt\n"
        "- In 'notes', briefly explain where you found the pattern\n"
    ),
    tools=[WebSearchTool()],
    model="gpt-4o-mini",
    output_type=EmailFormatPatterns,
)


executive_search_agent = Agent(
    name="ExecutiveSearch",
    instructions=(
        "You are an expert at finding senior executives and leadership at companies.\n\n"

        "Your goal: Find C-suite executives, VPs, Directors, and other senior leadership.\n\n"

        "Suggested search strategies (use these as inspiration, but adapt as needed):\n"
        "- Search for leadership team pages on company websites\n"
        "- Look for 'about us', 'team', 'leadership', 'management' pages\n"
        "- Search LinkedIn for executives at the company\n"
        "- Look for press releases, news articles mentioning executives\n"
        "- Search for executives on Crunchbase, Bloomberg, or other business databases\n"
        "- Look for SEC filings, annual reports (for public companies)\n"
        "- Search social media profiles (Twitter, LinkedIn bios)\n\n"

        "You have full autonomy to:\n"
        "- Choose which sources to check and in what order\n"
        "- Adapt your search strategy based on what you find\n"
        "- Use creative approaches to find executive names\n"
        "- Make multiple searches to find more executives or verify information\n\n"

        "Target roles (ONLY include senior leadership):\n"
        "✅ INCLUDE:\n"
        "- C-suite: CEO, CFO, CTO, CMO, COO, CIO, CSO, CHRO, CPO, etc.\n"
        "- EVP / Executive Vice President\n"
        "- SVP / Senior Vice President\n"
        "- VP / Vice President\n"
        "- Managing Director, General Manager\n"
        "- Director of [major department] (e.g., Director of Engineering)\n"
        "- Head of [major department] (e.g., Head of Sales)\n"
        "- Partner, Managing Partner, Senior Partner\n"
        "- Board members, Chairperson\n\n"

        "❌ DO NOT include:\n"
        "- Managers (unless 'Managing Director')\n"
        "- Coordinators, Specialists, Consultants\n"
        "- Engineers, Developers, Analysts (unless in title like 'Chief Analyst')\n"
        "- Associates, Assistants\n\n"

        "CRITICAL - You MUST return a structured JSON output with this EXACT format:\n"
        "{\n"
        '  "executives": [\n'
        "    {\n"
        '      "company_name": "Company Name",\n'
        '      "first_name": "John",\n'
        '      "last_name": "Smith",\n'
        '      "title": "Chief Executive Officer"\n'
        "    },\n"
        "    {\n"
        '      "company_name": "Company Name",\n'
        '      "first_name": "Jane",\n'
        '      "last_name": "Doe",\n'
        '      "title": "VP of Marketing"\n'
        "    }\n"
        "  ],\n"
        '  "count": 2\n'
        "}\n\n"

        "IMPORTANT:\n"
        "- Split full names into first_name and last_name (ignore middle names/initials)\n"
        "- Use the company_name from the prompt for each executive\n"
        "- Only include genuinely senior roles (not mid-level managers)\n"
        "- ABSOLUTELY NO DUPLICATES: Each executive should appear ONLY ONCE in the list\n"
        "- MAXIMUM 15 executives: Stop at 15 executives, do not exceed this limit\n"
        "- Aim to find 5-15 UNIQUE senior executives per company\n"
        "- Set count to the total number of UNIQUE executives found\n"
        "- Use the executive's full title as listed\n"
        "- Before adding an executive to the list, verify they are not already included\n"
    ),
    tools=[WebSearchTool()],
    model="gpt-4o-mini",
    output_type=ExecutiveList,
    max_tokens=4000,  # Prevent runaway generation
)


# =========================================================
# Extract structured outputs from agents
# =========================================================

def normalize_pattern(pattern: str) -> str:
    """
    Normalize an email pattern by fixing common issues.

    - Converts to lowercase
    - Removes spaces
    - Fixes common typos in placeholders
    """
    if not pattern:
        return pattern

    # Convert to lowercase
    pattern = pattern.lower().strip()

    # Remove all spaces
    pattern = pattern.replace(' ', '')

    # Fix common placeholder typos/variations
    placeholder_fixes = {
        '{fname}': '{first}',
        '{firstname}': '{first}',
        '{lname}': '{last}',
        '{lastname}': '{last}',
        '{fi}': '{f}',
        '{firstinitial}': '{f}',
        '{li}': '{l}',
        '{lastinitial}': '{l}',
        '{name}': '{first}',
        '{surname}': '{last}',
    }

    for old, new in placeholder_fixes.items():
        pattern = pattern.replace(old, new)

    return pattern


def validate_and_clean_email_pattern(pattern: str, domain: str) -> str | None:
    """
    Validate and clean an email pattern.

    Returns cleaned pattern or None if invalid.
    """
    if not pattern:
        return None

    # Normalize first
    pattern = normalize_pattern(pattern)

    # Must contain @ and the domain
    if '@' not in pattern:
        print(f"   ⚠️  Pattern missing @ symbol: {pattern}")
        return None

    # Check for valid placeholders only
    valid_placeholders = ['{first}', '{last}', '{f}', '{l}']

    # Extract all placeholders from the pattern
    found_placeholders = re.findall(r'\{[^}]*\}', pattern)

    if not found_placeholders:
        print(f"   ⚠️  No placeholders found in pattern: {pattern}")
        return None

    # Check if all placeholders are valid
    for placeholder in found_placeholders:
        if placeholder not in valid_placeholders:
            print(f"   ⚠️  Invalid placeholder in pattern: {placeholder} in {pattern}")
            return None

    # Pattern should only have: alphanumeric, dots, dashes, underscores, @, and valid placeholders
    # Remove placeholders temporarily to validate the rest
    temp_pattern = pattern
    for placeholder in valid_placeholders:
        temp_pattern = temp_pattern.replace(placeholder, 'X')

    # Check if remaining characters are valid (letters, numbers, @, ., -, _)
    if not re.match(r'^[a-zA-Z0-9@.\-_]+$', temp_pattern):
        print(f"   ⚠️  Invalid characters in pattern: {pattern}")
        print(f"      After removing placeholders: {temp_pattern}")
        return None

    return pattern


def extract_email_formats(result) -> List[str]:
    """Extract email format patterns from structured agent output."""
    try:
        # Agent returns EmailFormatPatterns TypedDict
        if hasattr(result, 'final_output'):
            output = result.final_output
            if isinstance(output, dict) and 'patterns' in output:
                raw_patterns = output['patterns']
                print(f"   Raw patterns from agent: {raw_patterns}")

                # Validate and clean each pattern
                validated_patterns = []
                for pattern in raw_patterns:
                    # Get domain from pattern
                    if '@' in pattern:
                        domain = pattern.split('@')[-1]
                        cleaned = validate_and_clean_email_pattern(pattern, domain)
                        if cleaned:
                            validated_patterns.append(cleaned)
                        else:
                            print(f"   ⚠️  Rejected invalid pattern: {pattern}")

                print(f"   Validated patterns: {validated_patterns}")
                return validated_patterns
        return []
    except Exception as e:
        print(f"⚠️  Error extracting email formats: {e}")
        return []


def extract_executives(result) -> List[Person]:
    """Extract executive list from structured agent output with deduplication."""
    try:
        # Agent returns ExecutiveList TypedDict
        if hasattr(result, 'final_output'):
            output = result.final_output
            if isinstance(output, dict) and 'executives' in output:
                executives = output['executives']
                print(f"   Raw executives found: {len(executives)}")

                # Deduplicate executives by name (first + last)
                seen_names = set()
                unique_executives = []
                duplicates_removed = 0

                for exec_data in executives:
                    first = exec_data.get('first_name', '').strip().lower()
                    last = exec_data.get('last_name', '').strip().lower()
                    name_key = f"{first}_{last}"

                    if name_key and name_key not in seen_names:
                        seen_names.add(name_key)
                        unique_executives.append(exec_data)
                    else:
                        duplicates_removed += 1

                if duplicates_removed > 0:
                    print(f"   ⚠️  Removed {duplicates_removed} duplicate executive(s)")

                print(f"   Unique executives: {len(unique_executives)}")
                return unique_executives if unique_executives else []
        return []
    except Exception as e:
        print(f"⚠️  Error extracting executives: {e}")
        return []


# =========================================================
# Retry logic with exponential backoff
# =========================================================

async def run_agent_with_retry(
    agent: Agent,
    prompt: str,
    max_turns: int = 20,
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0
):
    """
    Run an agent with exponential backoff retry logic.

    Args:
        agent: The agent to run
        prompt: The prompt to send to the agent
        max_turns: Maximum turns for the agent
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 2.0)
        max_delay: Maximum delay in seconds (default: 60.0)

    Returns:
        Agent result or raises exception after all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            result = await Runner.run(agent, prompt, max_turns=max_turns)
            return result

        except MaxTurnsExceeded as e:
            # Don't retry if max turns exceeded - this is a logic issue, not transient
            print(f"⚠️  {agent.name}: Max turns exceeded (not retrying)")
            raise e

        except Exception as e:
            last_exception = e

            # Check if this is a rate limit or network error (retryable)
            error_msg = str(e).lower()
            is_retryable = any(keyword in error_msg for keyword in [
                'rate limit',
                'timeout',
                'connection',
                'network',
                'temporarily unavailable',
                'service unavailable',
                '429',  # HTTP 429 Too Many Requests
                '500',  # HTTP 500 Internal Server Error
                '502',  # HTTP 502 Bad Gateway
                '503',  # HTTP 503 Service Unavailable
                '504',  # HTTP 504 Gateway Timeout
            ])

            if not is_retryable:
                print(f"❌ {agent.name}: Non-retryable error: {e}")
                raise e

            if attempt < max_retries - 1:
                # Calculate exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
                total_delay = delay + jitter

                print(f"⚠️  {agent.name}: Attempt {attempt + 1}/{max_retries} failed: {e}")
                print(f"   Retrying in {total_delay:.1f}s...")

                await asyncio.sleep(total_delay)
            else:
                print(f"❌ {agent.name}: All {max_retries} retry attempts exhausted")
                raise last_exception

    # Should never reach here, but just in case
    raise last_exception


# =========================================================
# Main parallel processing function
# =========================================================

async def process_company_parallel(
    company_name: str,
    company_domain: str,
    company_website: str = "",
    output_file: str = "contacts.csv",
    max_turns: int = 20,
    max_retries: int = 3,
    retry_base_delay: float = 2.0
) -> tuple[pd.DataFrame | None, str]:
    """
    Process a single company using parallel agent execution with retry logic.

    Both agents run simultaneously for better performance.

    Args:
        company_name: Name of the company
        company_domain: Company domain (e.g., 'company.com')
        company_website: Full website URL (optional)
        output_file: Output CSV file path
        max_turns: Maximum turns per agent
        max_retries: Maximum retry attempts for transient failures (default: 3)
        retry_base_delay: Base delay in seconds for exponential backoff (default: 2.0)

    Returns:
        Tuple of (DataFrame with results or None, output file path)
    """

    print(f"\n{'='*60}")
    print(f"🔍 PROCESSING: {company_name}")
    print(f"{'='*60}\n")

    website_info = f"\nWebsite: {company_website}" if company_website else ""

    # Prepare prompts for both agents
    format_prompt = f"""
Find the email format for {company_name}.{website_info}
Domain: {company_domain}

Search for their email format and return the pattern(s).
"""

    executive_prompt = f"""
Find senior executives at {company_name}.{website_info}
Domain: {company_domain}

Search for C-suite, VPs, and Directors. Return in the format:
Name: [First Last]
Title: [Job Title]
---
"""

    try:
        # Run both agents IN PARALLEL using asyncio.gather with retry logic
        with trace(f"Processing {company_name}"):
            print("🚀 Running both agents in parallel with retry support...")

            format_result, executive_result = await asyncio.gather(
                run_agent_with_retry(
                    email_format_agent,
                    format_prompt,
                    max_turns=max_turns,
                    max_retries=max_retries,
                    base_delay=retry_base_delay
                ),
                run_agent_with_retry(
                    executive_search_agent,
                    executive_prompt,
                    max_turns=max_turns,
                    max_retries=max_retries,
                    base_delay=retry_base_delay
                ),
                return_exceptions=True
            )

            # Check for exceptions and extract structured outputs
            if isinstance(format_result, Exception):
                print(f"⚠️  Email format agent failed: {format_result}")
                email_formats = []
            else:
                email_formats = extract_email_formats(format_result)
                print(f"✅ Found {len(email_formats)} email format(s)")

            if isinstance(executive_result, Exception):
                print(f"⚠️  Executive search agent failed: {executive_result}")
                executives = []
            else:
                executives = extract_executives(executive_result)
                print(f"✅ Found {len(executives)} executive(s)")

            # Generate emails
            if executives:
                email_guesses = generate_emails_sync(
                    names=executives,
                    formats=email_formats,
                    company_domain=company_domain
                )

                if email_guesses:
                    # Export to CSV
                    df = pd.DataFrame(email_guesses)
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    print(f"✅ Exported {len(email_guesses)} contacts to {output_file}")
                    return df, output_file
                else:
                    print("⚠️  No emails generated")
                    return None, output_file
            else:
                print("⚠️  No executives found")
                return None, output_file

    except MaxTurnsExceeded:
        print(f"⚠️  Max turns exceeded")
        return None, output_file
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, output_file


# =========================================================
# Batch processing
# =========================================================

async def process_multiple_companies(
    companies: List[tuple[str, str, str]],
    output_dir: str = ".",
    max_retries: int = 3,
    retry_base_delay: float = 2.0
) -> List[pd.DataFrame]:
    """
    Process multiple companies in parallel with retry logic.

    Args:
        companies: List of (company_name, domain, website) tuples
        output_dir: Directory to save output files
        max_retries: Maximum retry attempts for transient failures (default: 3)
        retry_base_delay: Base delay in seconds for exponential backoff (default: 2.0)

    Returns:
        List of DataFrames with results
    """
    print(f"\n🚀 Processing {len(companies)} companies in parallel...\n")

    tasks = []
    for company_name, domain, website in companies:
        safe_name = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')
        output_file = f"{output_dir}/{safe_name}_contacts.csv"

        tasks.append(
            process_company_parallel(
                company_name=company_name,
                company_domain=domain,
                company_website=website,
                output_file=output_file,
                max_turns=20,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay
            )
        )

    # Run all companies in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect successful results
    successful_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"⚠️  Company processing failed: {result}")
        else:
            df, _ = result
            if df is not None:
                successful_results.append(df)

    return successful_results


# =========================================================
# Incremental CSV saving
# =========================================================

def append_to_csv(
    data: List[dict],
    output_file: str,
    write_header: bool = False
):
    """
    Append data to CSV file incrementally.

    Args:
        data: List of dict records to append
        output_file: Path to CSV file
        write_header: If True, write header row (for first write)
    """
    if not data:
        return

    file_exists = Path(output_file).exists()
    mode = 'a' if file_exists and not write_header else 'w'

    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        if mode == 'w' or write_header:
            writer.writeheader()
        writer.writerows(data)

    print(f"   💾 Saved {len(data)} contacts to {output_file}")


# =========================================================
# Main execution
# =========================================================

def main(
    input_file: str,
    output_file: str = "all_contacts_output.csv",
    max_retries: int = 3,
    retry_base_delay: float = 2.0
):
    """
    Process multiple companies from spreadsheet with retry logic.

    Results are saved incrementally to the output file as each company is processed.

    Args:
        input_file: Path to CSV or Excel file with companies
        output_file: Path for output CSV file (saved incrementally)
        max_retries: Maximum retry attempts for transient failures (default: 3)
        retry_base_delay: Base delay in seconds for exponential backoff (default: 2.0)
    """

    print(f"📂 Reading: {input_file}")

    file_path = Path(input_file)
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(input_file, encoding=encoding)
                break
            except:
                continue
        if df is None:
            raise Exception("Could not read file")

    print(f"✅ Loaded {len(df)} companies\n")

    company_col = df.columns[0]
    website_col = df.columns[1] if len(df.columns) > 1 else None

    # Prepare company list
    companies = []
    for idx, row in df.iterrows():
        company_name = row[company_col]

        if pd.isna(company_name) or str(company_name).strip() == "":
            continue

        company_name = str(company_name).strip()
        website = str(row[website_col]).strip() if website_col and not pd.isna(row[website_col]) else ""

        # Extract domain
        company_domain = ""
        if website:
            domain_match = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', website)
            if domain_match:
                company_domain = domain_match.group(1)

        if not company_domain:
            company_domain = company_name.lower().replace(' ', '').replace(',', '').replace('.', '') + '.com'

        companies.append((company_name, company_domain, website))

    # Initialize output CSV with headers
    output_path = Path(output_file)
    first_write = True
    if output_path.exists():
        print(f"⚠️  Output file {output_file} already exists - will append to it")
        print(f"   To start fresh, delete the file first\n")
        first_write = False
    else:
        print(f"📝 Initializing output file: {output_file}\n")
        # Will write header on first append

    # Track progress
    total_contacts = 0
    companies_processed = 0
    companies_with_results = 0

    async def process_all():
        nonlocal total_contacts, companies_processed, companies_with_results, first_write

        for idx, (company_name, company_domain, website) in enumerate(companies, 1):
            print(f"\n{'='*60}")
            print(f"📍 PROGRESS: {idx}/{len(companies)} companies")
            print(f"{'='*60}")

            safe_name = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')
            temp_csv = f"temp_{safe_name}_contacts.csv"

            result_df, _ = await process_company_parallel(
                company_name=company_name,
                company_domain=company_domain,
                company_website=website,
                output_file=temp_csv,
                max_turns=20,
                max_retries=max_retries,
                retry_base_delay=retry_base_delay
            )

            companies_processed += 1

            if result_df is not None and len(result_df) > 0:
                # Collect results for this company
                company_contacts = []
                for _, contact_row in result_df.iterrows():
                    title = contact_row.get("title", "")

                    if is_senior_enough(title):
                        company_contacts.append({
                            "Company Name": company_name,
                            "First Name": contact_row.get("first_name", ""),
                            "Last Name": contact_row.get("last_name", ""),
                            "Job Title": title,
                            "Email Guess": contact_row.get("email_guess", "")
                        })

                # Save incrementally to main output file
                if company_contacts:
                    # Write header on first write, then append
                    append_to_csv(company_contacts, output_file, write_header=first_write)
                    if first_write:
                        first_write = False
                    total_contacts += len(company_contacts)
                    companies_with_results += 1

                    print(f"   ✅ {len(company_contacts)} contacts saved to {output_file}")
                    print(f"   📊 Total so far: {total_contacts} contacts from {companies_with_results}/{companies_processed} companies")
                else:
                    print(f"   ⚠️  No senior contacts found for {company_name}")

                # Clean up temp file
                if Path(temp_csv).exists():
                    Path(temp_csv).unlink()
            else:
                print(f"   ⚠️  No results for {company_name}")

    # Run async processing
    asyncio.run(process_all())

    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Final Results:")
    print(f"   • Companies processed: {companies_processed}/{len(companies)}")
    print(f"   • Companies with results: {companies_with_results}")
    print(f"   • Total contacts: {total_contacts}")
    print(f"   • Output file: {output_file}")
    print(f"{'='*60}\n")


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    # Update these paths for your environment
    INPUT_FILE = "input_companies.csv"  # or .xlsx
    OUTPUT_FILE = "output_contacts.csv"

    main(INPUT_FILE, OUTPUT_FILE)
